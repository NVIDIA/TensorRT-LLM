#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Dict

import torch
import torch.export
import torch.nn as nn
from transformers import AutoConfig, LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)


class Eagle3Attention(LlamaAttention):
    def __init__(self, config: LlamaConfig, layer_idx: int = 0):
        super().__init__(config, layer_idx)
        # Patching the projection layers to accept 8192 (2 * hidden_size)
        # Your checkpoint: model.midlayer.self_attn.q_proj.weight [4096, 8192]
        self.q_proj = nn.Linear(
            2 * config.hidden_size, config.num_attention_heads * config.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            2 * config.hidden_size, config.num_key_value_heads * config.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            2 * config.hidden_size, config.num_key_value_heads * config.head_dim, bias=False
        )
        # o_proj remains [4096, 4096]


class Eagle3DecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int = 0):
        super().__init__(config, layer_idx)
        self.self_attn = Eagle3Attention(config, layer_idx=layer_idx)
        # Checkpoint expects: model.midlayer.hidden_norm.weight
        self.hidden_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states,
        embeds,
        position_ids,
        attention_mask,
        position_embeds,
        past_key_value=None,
    ):
        residual = hidden_states
        hidden_states = self.hidden_norm(hidden_states)

        embeds = self.input_layernorm(embeds)
        hidden_states = torch.cat([embeds, hidden_states], dim=-1)

        hidden_states = self.self_attn(
            position_ids=position_ids,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeds,
        )[0]

        print(f"residual: {residual}, hidden_states: {hidden_states}")
        hidden_states = residual + hidden_states

        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        # TODO: Need to capture hidden states
        return hidden_states, residual


class EagleModel(nn.Module):
    def __init__(self, config: LlamaConfig, eh_proj_before_attn: bool = False):
        super().__init__()
        # 1. Vocab Mappings
        self.register_buffer("t2d", torch.zeros(config.vocab_size, dtype=torch.long))
        self.register_buffer("d2t", torch.zeros(config.draft_vocab_size, dtype=torch.long))

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        if config.draft_vocab_size is not None and config.draft_vocab_size != config.vocab_size:
            print(
                f"ℹ️  Patching config: vocab_size {config.vocab_size} -> {config.draft_vocab_size}"
            )
            config.vocab_size = config.draft_vocab_size

        # 2. Input Feature Fusion (12288 -> 4096)
        # This matches your: model.fc.weight [4096, 12288]
        self.fc = nn.Linear(config.hidden_size * 3, config.hidden_size, bias=False)

        self.rotary_emb = LlamaRotaryEmbedding(config=config)

        # 3. The Transformer Layer (Midlayer)
        self.midlayer = Eagle3DecoderLayer(config, layer_idx=0)

        if eh_proj_before_attn:
            self.enorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.eh_proj = nn.Linear(
                config.hidden_size * 2, config.hidden_size, bias=False, dtype=config.torch_dtype
            )

        # 4. Output Head
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self, input_ids, position_ids, hidden_states, attention_mask=None, past_key_value=None
    ):
        # before: hidden_states: [Batch, Seq, 3* hidden_size]
        # after: hidden_states: [Batch, Seq, hidden_size]
        hidden_states = self.fc(hidden_states)

        input_embeds = self.embed_tokens(input_ids)

        cos, sin = self.rotary_emb(hidden_states, position_ids)

        position_embeds = (cos, sin)

        # EAGLE logic usually concatenates the projection with original hidden state
        # to create the [Batch, Seq, 8192] input for the midlayer
        # (Assuming x is current prediction and we concat with base hidden state)
        # This is a placeholder for the exact concatenation logic:
        # x_attn = torch.cat([x, base_hidden_state], dim=-1)

        # For pure loading/inference testing:
        out = self.midlayer(
            hidden_states=hidden_states,
            embeds=input_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            position_embeds=position_embeds,
        )[0]
        logits = self.lm_head(self.norm(out))
        return logits

    def load_weights(self, weights: Dict):
        missing, unexpected = self.load_state_dict(weights, strict=False)
        if missing:
            print(f"⚠️ Missing keys (initialized randomly): {missing}")
        if unexpected:
            print(f"⚠️ Unexpected keys: {unexpected}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="Path to local config.json")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--seq", type=int, default=16)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--dtype", type=str, default="float16")
    # We ignore --hf_repo for loading logic now, relying on local files
    ap.add_argument("--hf_repo", type=str, default="", help="Ignored in this version")
    args = ap.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    # 1. Setup Device & Dtype
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    # 2. Load Config & Patch Layers
    print(f"Loading config from: {config_path}")
    cfg = AutoConfig.from_pretrained(config_path, attn_implementation="eager")

    # CRITICAL FIX: The config says 32 layers (Llama), but Eagle is 1 layer.
    # We must overwrite this before creating the model.
    if cfg.num_hidden_layers > 1:
        print(f"ℹ️  Patching config: num_hidden_layers {cfg.num_hidden_layers} -> 1")
        cfg.num_hidden_layers = 1

    print(f"Config: {cfg}")

    # 4. Manually Load Weights (Bypassing from_pretrained checks)
    # Look for pytorch_model.bin or model.safetensors next to the config
    model_root = config_path.parent
    bin_path = model_root / "pytorch_model.bin"
    print(f"Loading weights from: {bin_path}")
    state_dict = torch.load(bin_path, map_location="cpu", weights_only=True)

    dtype = torch.float16

    model = EagleModel(cfg)
    model.load_weights(state_dict)
    model.to(device, dtype=dtype)
    model.eval()

    # Mock parameters
    batch_size = 1
    seq_len = 8
    hidden_dim = 4096
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Mock Hidden States (3 layers concatenated)
    # In reality, these come from Llama 3.1: [layer_2, layer_16, layer_30]
    mock_hidden_states = torch.randn(
        (batch_size, seq_len, hidden_dim * 3), device=device, dtype=dtype
    )

    # 2. Mock Input IDs (Llama 3.1 vocab size 128256)
    input_ids = torch.randint(0, 128256, (batch_size, seq_len), device=device, dtype=torch.long)

    # 3. Position IDs
    position_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)

    with torch.inference_mode():
        output_logits = model(
            input_ids=input_ids, position_ids=position_ids, hidden_states=mock_hidden_states
        )

    print(f"Output shape: {output_logits.shape}")
    # Expected: [batch, seq, 32000] (Draft vocab size)

    # ... (previous code in main) ...

    print("\n--- Starting torch.export ---")

    # 1. Prepare Inputs for Export
    # torch.export requires a tuple of arguments matching the forward signature strictly.
    # Signature: forward(hidden_states, position_ids, attention_mask, input_ids)

    # We create a dummy mask. For Llama, a 4D mask is standard [Batch, 1, Seq, Seq],
    # but passing None is allowed if your model logic handles it.
    # We will pass a concrete tensor to be safe and explicit for the graph.
    mock_attention_mask = torch.ones((batch_size, 1, seq_len, seq_len), device=device, dtype=dtype)
    # Alternatively, use None if you want to trace the 'no mask' path:
    # mock_attention_mask = None

    print(input_ids.shape, position_ids.shape, mock_hidden_states.shape, mock_attention_mask.shape)

    example_args = (
        input_ids,  # input_ids
        position_ids,  # position_ids
        mock_hidden_states,  # hidden_states
        mock_attention_mask,  # attention_mask (Optional)
    )

    # 2. Run torch.export
    try:
        # We use strict=False tentatively because Transformers models sometimes
        # have dictionary outputs or specific python control flow that strict mode dislikes.
        # However, for pure graph capture, standard export is usually fine.
        exported_program = torch.export.export(model, args=example_args)

        print("✅ Export successful!")

        # 3. Print the Graph
        # .graph prints the raw IR (Intermediate Representation) nodes
        print("\n--- Exported Graph IR ---")
        print(exported_program.graph)

        # .graph_module.code prints the Python-like code representation (easier to read)
        print("\n--- Generated Graph Code ---")
        print(exported_program.graph_module.code)

    except Exception as e:
        print(f"❌ Export failed: {e}")
        # Identify if it is a side-effect issue (like the print statement in your layer)


if __name__ == "__main__":
    main()
