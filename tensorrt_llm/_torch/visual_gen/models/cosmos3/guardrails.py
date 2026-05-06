from __future__ import annotations

import os
import warnings
from typing import Callable

import cv2
import numpy as np
import torch
import torch.nn as nn

from tensorrt_llm.logger import logger

TextGuardrailFn = Callable[[str], tuple[bool, str]]
VideoGuardrailFn = Callable[[np.ndarray], np.ndarray]

GUARDRAIL_HF_REPO = "nvidia/Cosmos-Guardrail1"
GUARDRAIL_HF_REVISION = "d6d4bfa899a71454a700907664f3e88f503950cf"
CUTOFF_UNSAFE_FRAMES_PERCENT = 10


# ---------------------------------------------------------------------------
# Video safety classifier (matches reference: SigLIP so400m + 3-layer head)
# ---------------------------------------------------------------------------
class SafetyClassifier(nn.Module):
    """3-layer classifier with BatchNorm (1152 → 512 → 256 → 7)."""

    def __init__(self, input_size: int = 1152, num_classes: int = 7):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


CLASS_IDX_TO_NAME = {
    0: "Safe",
    1: "Sexual_Content",
    3: "Drugs",
    4: "Child_Abuse",
    5: "Hate_and_Harassment",
    6: "Self-Harm",
}


# ---------------------------------------------------------------------------
# Face pixelation utility
# ---------------------------------------------------------------------------
def _pixelate_face(face_img: np.ndarray, blocks: int = 5) -> np.ndarray:
    h, w = face_img.shape[:2]
    if h == 0 or w == 0:
        return face_img
    temp = cv2.resize(face_img, (blocks, blocks), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)


# ---------------------------------------------------------------------------
# Default guardrail builders
# ---------------------------------------------------------------------------
def download_guardrail_checkpoint() -> str:
    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(
            GUARDRAIL_HF_REPO,
            revision=GUARDRAIL_HF_REVISION,
            local_files_only=True,
        )
    except FileNotFoundError:
        logger.warning(
            f"Guardrail checkpoint not found, downloading from {GUARDRAIL_HF_REPO} {GUARDRAIL_HF_REVISION}"
        )
        return snapshot_download(
            GUARDRAIL_HF_REPO,
            revision=GUARDRAIL_HF_REVISION,
        )


def build_text_guardrail(guardrail_ckpt_dir: str) -> TextGuardrailFn:
    checkers: list[Callable[[str], tuple[bool, str]]] = []

    # 1. Blocklist
    try:
        import nltk
        from better_profanity import profanity as profanity_filter

        blocklist_dir = os.path.join(guardrail_ckpt_dir, "blocklist")
        nltk.data.path.append(os.path.join(blocklist_dir, "nltk_data"))

        def _read_keywords(dirpath: str) -> list[str]:
            words: list[str] = []
            if not os.path.isdir(dirpath):
                return words
            for fname in sorted(os.listdir(dirpath)):
                fpath = os.path.join(dirpath, fname)
                if os.path.isfile(fpath):
                    with open(fpath) as f:
                        words.extend(line.strip() for line in f if line.strip())
            return words

        blocklist_words = _read_keywords(os.path.join(blocklist_dir, "custom"))
        whitelist_words = _read_keywords(os.path.join(blocklist_dir, "whitelist"))
        profanity_filter.load_censor_words(
            custom_words=blocklist_words, whitelist_words=whitelist_words
        )

        def _blocklist_check(prompt: str) -> tuple[bool, str]:
            if profanity_filter.contains_profanity(prompt):
                return False, "Blocked by keyword filter"
            return True, ""

        checkers.append(_blocklist_check)
        logger.info("Blocklist guardrail loaded (%d keywords)", len(blocklist_words))
    except ImportError:
        logger.warning("better-profanity or nltk not installed; skipping blocklist guardrail")

    # 2. Qwen3Guard
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_id = "Qwen/Qwen3Guard-Gen-0.6B"
        qwen_tokenizer = AutoTokenizer.from_pretrained(model_id)
        qwen_model = (
            AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
            )
            .to("cuda")
            .eval()
        )

        def _qwen_check(prompt: str) -> tuple[bool, str]:
            conversations = [{"role": "user", "content": prompt}]
            input_ids = qwen_tokenizer.apply_chat_template(
                conversations,
                tokenize=True,
                return_tensors="pt",
                add_generation_prompt=True,
            ).to("cuda")
            with torch.no_grad():
                output_ids = qwen_model.generate(input_ids, max_new_tokens=128)
            response = qwen_tokenizer.decode(
                output_ids[0][input_ids.shape[1] :],
                skip_special_tokens=True,
            )
            if "unsafe" in response.lower():
                return False, f"Qwen3Guard: {response.strip()}"
            return True, ""

        checkers.append(_qwen_check)
        logger.info("Qwen3Guard guardrail loaded")
    except ImportError:
        logger.warning("transformers not installed; skipping Qwen3Guard")

    def text_guardrail(prompt: str) -> None:
        for checker in checkers:
            is_safe, msg = checker(prompt)
            if not is_safe:
                return is_safe, msg
        return True, ""

    return text_guardrail


def build_video_guardrail(guardrail_ckpt_dir: str) -> VideoGuardrailFn:
    safety_checker: Callable[[np.ndarray], tuple[bool, str]] | None = None
    face_blurrer: Callable[[np.ndarray], np.ndarray] | None = None

    # 1. Video content safety filter: SigLIP so400m + SafetyClassifier
    try:
        from PIL import Image
        from transformers import SiglipModel, SiglipProcessor

        siglip_id = "google/siglip-so400m-patch14-384"
        siglip_model = SiglipModel.from_pretrained(siglip_id).to("cuda", dtype=torch.float32).eval()
        siglip_processor = SiglipProcessor.from_pretrained(siglip_id)

        classifier = SafetyClassifier(input_size=1152, num_classes=7)
        ckpt_path = os.path.join(
            guardrail_ckpt_dir, "video_content_safety_filter", "safety_filter.pt"
        )
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        state = {k.removeprefix("network."): v for k, v in checkpoint["model"].items()}
        classifier.load_state_dict(state)
        classifier = classifier.to("cuda", dtype=torch.float32).eval()

        def _safety_check(frames: np.ndarray) -> tuple[bool, str]:
            nonlocal siglip_model, classifier

            unsafe_count = 0
            total = len(frames)
            for frame in frames:
                img = Image.fromarray(frame)
                inputs = siglip_processor(images=img, return_tensors="pt").to(
                    "cuda", dtype=torch.float32
                )
                with torch.no_grad():
                    features = siglip_model.get_image_features(**inputs)
                    features = features / features.norm(dim=-1, keepdim=True)
                    logits = classifier(features)
                    pred = logits.argmax(dim=-1).item()
                class_name = CLASS_IDX_TO_NAME.get(pred, "Unknown")
                if class_name != "Safe":
                    unsafe_count += 1

            if unsafe_count / total > CUTOFF_UNSAFE_FRAMES_PERCENT / 100:
                return False, f"Video content safety: {unsafe_count}/{total} frames unsafe"
            return True, ""

        safety_checker = _safety_check
        logger.info("Video content safety filter loaded (SigLIP so400m + classifier)")
    except (ImportError, FileNotFoundError) as e:
        logger.warning("Could not load video safety filter: %s", e)

    # 2. Face blur: RetinaFace + pixelation
    try:
        from retinaface.data import cfg_re50
        from retinaface.layers.functions.prior_box import PriorBox
        from retinaface.models.retinaface import RetinaFace
        from retinaface.utils.nms.py_cpu_nms import py_cpu_nms

        face_ckpt = os.path.join(guardrail_ckpt_dir, "face_blur_filter", "Resnet50_Final.pth")
        if not os.path.exists(face_ckpt):
            raise FileNotFoundError(face_ckpt)

        cfg = dict(cfg_re50)
        cfg["pretrain"] = False
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            retinaface_net = RetinaFace(cfg=cfg, phase="test")

        # Load weights (strip 'module.' prefix if present)
        pretrained_dict = torch.load(face_ckpt, map_location="cpu", weights_only=True)
        if "state_dict" in pretrained_dict:
            pretrained_dict = pretrained_dict["state_dict"]
        pretrained_dict = {
            k.replace("module.", "", 1) if k.startswith("module.") else k: v
            for k, v in pretrained_dict.items()
        }
        retinaface_net.load_state_dict(pretrained_dict, strict=False)
        retinaface_device = "cuda"
        retinaface_net = retinaface_net.to(retinaface_device, dtype=torch.float32).eval()

        CONF_THRESH = 0.7
        NMS_THRESH = 0.4
        TOP_K = 5000
        KEEP_TOP_K = 750

        def _decode_batch(loc, priors, variances):
            batch_size = loc.size(0)
            p = priors.unsqueeze(0).expand(batch_size, -1, -1)
            boxes = torch.cat(
                (
                    p[:, :, :2] + loc[:, :, :2] * variances[0] * p[:, :, 2:],
                    p[:, :, 2:] * torch.exp(loc[:, :, 2:] * variances[1]),
                ),
                dim=2,
            )
            boxes[:, :, :2] -= boxes[:, :, 2:] / 2
            boxes[:, :, 2:] += boxes[:, :, :2]
            return boxes

        def _face_blur(frames: np.ndarray) -> np.ndarray:
            nonlocal retinaface_net

            prior_data = None
            scale = None
            result_frames = []

            for frame in frames:
                frame_t = torch.from_numpy(frame).to("cuda", dtype=torch.float32)
                frame_t = frame_t.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
                frame_t = frame_t[:, [2, 1, 0], :, :]  # RGB → BGR
                means = torch.tensor(
                    [104.0, 117.0, 123.0], device="cuda", dtype=torch.float32
                ).view(1, 3, 1, 1)
                frame_t = frame_t - means

                h, w = frame_t.shape[2], frame_t.shape[3]
                if prior_data is None:
                    priorbox = PriorBox(cfg, image_size=(h, w))
                    prior_data = priorbox.forward().to("cuda", dtype=torch.float32)
                if scale is None:
                    scale = torch.tensor([w, h, w, h], device="cuda", dtype=torch.float32)

                with torch.no_grad():
                    loc, conf, _ = retinaface_net(frame_t)

                boxes = _decode_batch(loc, prior_data, cfg["variance"])
                boxes = (boxes * scale).squeeze(0).cpu().numpy()
                scores = conf.squeeze(0)[:, 1].cpu().numpy()

                # Filter by confidence
                inds = np.where(scores > CONF_THRESH)[0]
                boxes_f = boxes[inds]
                scores_f = scores[inds]
                order = scores_f.argsort()[::-1][:TOP_K]
                boxes_f = boxes_f[order]
                scores_f = scores_f[order]

                # NMS
                dets = np.hstack((boxes_f, scores_f[:, np.newaxis])).astype(np.float32)
                keep = py_cpu_nms(dets, NMS_THRESH)
                dets = dets[keep][:KEEP_TOP_K]

                out_frame = frame.copy()
                for det in dets:
                    x1, y1, x2, y2 = map(int, det[:4])
                    if x2 - x1 < 20 or y2 - y1 < 20:
                        continue
                    max_h, max_w = out_frame.shape[:2]
                    y1c, y2c = max(y1, 0), min(y2, max_h)
                    x1c, x2c = max(x1, 0), min(x2, max_w)
                    out_frame[y1c:y2c, x1c:x2c] = _pixelate_face(out_frame[y1c:y2c, x1c:x2c])

                result_frames.append(out_frame)

            return np.array(result_frames)

        face_blurrer = _face_blur
        logger.info("Face blur filter loaded (RetinaFace Resnet50)")
    except (ImportError, FileNotFoundError) as e:
        logger.warning("Could not load face blur filter: %s", e)

    def video_guardrail(frames: np.ndarray) -> np.ndarray | None:
        if safety_checker is not None:
            is_safe, msg = safety_checker(frames)
            if not is_safe:
                logger.warning(f"Video content safety: {msg}")
                return None
        if face_blurrer is not None:
            frames = face_blurrer(frames)
        return frames

    return video_guardrail


def check_video_safety(
    video_tensor: torch.Tensor, video_guardrail: VideoGuardrailFn
) -> torch.Tensor | None:
    v = video_tensor.detach().cpu()
    if v.dim() == 5:
        v = v[0]
    frames_np = v.numpy()
    frames_np = video_guardrail(frames_np)
    if frames_np is None:
        return None

    result = torch.from_numpy(frames_np)
    if video_tensor.dim() == 4:
        result = result.unsqueeze(0)
    return result.to(video_tensor.device)
