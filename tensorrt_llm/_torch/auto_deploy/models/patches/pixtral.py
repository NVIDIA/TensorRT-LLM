"""Patches for the PixtralVisionModel to make it compatible with `torch.export`.

On top of the patching, `custom_op`s are registered to replace specific parts of the Pixtral model's
forward pass that are not compatible with `torch.export`. Note that the `register_fake` portion of
the ops needs to return the shape (and dtype) of the output tensor(s) without accessing the values in
the input tensors, which is where things get tricky, and why so many custom ops / patches are needed.

NOTE: most patches are not used at the moment since only text submodule is exported. Keeping it here
for future reference in case we decide to also export the image model.
"""

import torch
from transformers.models.mistral3.modeling_mistral3 import Mistral3PatchMerger
from transformers.models.pixtral.modeling_pixtral import (
    PixtralRMSNorm,
    PixtralVisionModel,
    position_ids_in_meshgrid,
)

from ...export.interface import DisabledBaseExportPatch, ExportPatchRegistry

# NOTES:
# 1. Everything decorated by a `custom_op` must be type annotated.
# 2. The annotations must be one of the internally supported param types. As such, `self: PixtralVisionModel`
#    is a no-go.
# 3. This means that pretty much only free-standing functions with tensor inputs are supported - instance
#    methods cannot be decorated.


@torch.library.custom_op("auto_deploy::pixtral_process_patch_embeds", mutates_args={})
def _process_patch_embeds(
    patch_embeds: torch.Tensor,
    image_sizes: torch.Tensor,
    patch_size: int,
    hidden_size: int,
    max_width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    patch_embeds_list = []
    for embed, size in zip(patch_embeds, image_sizes):
        # size is a 1-D tensor [H, W]; convert to Python ints for indexing.
        h = int((size[0] // patch_size).item())
        w = int((size[1] // patch_size).item())
        patch_embeds_list.append(embed[..., :h, :w])

    # flatten to a single sequence
    patch_embeds = torch.cat([p.flatten(1).T for p in patch_embeds_list], dim=0).unsqueeze(0)

    position_ids = position_ids_in_meshgrid(patch_embeds_list, max_width=max_width)

    return patch_embeds, position_ids


@_process_patch_embeds.register_fake
def _process_patch_embeds_meta(
    patch_embeds: torch.Tensor,
    image_sizes: torch.Tensor,
    patch_size: int,
    hidden_size: int,
    max_width: int,
):
    B = (image_sizes // patch_size).prod(dim=1).sum()
    device = patch_embeds.device
    return (
        # Leading 1 = `unsqueeze(0)` after concatenating the `patch_embeds_list`.
        torch.empty(1, B, hidden_size, device=device),
        torch.empty(B, device=device, dtype=torch.int64),
    )


def _pixtral_forward(
    self: PixtralVisionModel,
    pixel_values: torch.Tensor,
    image_sizes: torch.Tensor | None,
    output_hidden_states: bool | None = None,
    output_attentions: bool | None = None,
    return_dict: bool | None = None,
    *args,
    **kwargs,
):
    if image_sizes is None:
        batch_size, _, height, width = pixel_values.shape
        image_sizes = torch.tensor([(height, width)] * batch_size, device=pixel_values.device)

    # pass images through initial convolution independently
    patch_embeds = self.patch_conv(pixel_values)
    patch_embeds, position_ids = torch.ops.auto_deploy.pixtral_process_patch_embeds(
        patch_embeds=patch_embeds,
        image_sizes=image_sizes,
        patch_size=self.patch_size,
        hidden_size=self.config.hidden_size,
        max_width=self.config.image_size // self.config.patch_size,
    )

    patch_embeds = self.ln_pre(patch_embeds)

    # Constrain sequence length to be size-like and > 1 for export guards.
    _seq_len = patch_embeds.shape[1]
    torch._check_is_size(_seq_len)
    torch._check(_seq_len > 1)

    position_embeddings = self.patch_positional_embedding(patch_embeds, position_ids)

    if self.config._attn_implementation == "flash_attention_2":
        # We only rely on position_ids when using flash_attention_2
        attention_mask = None
    else:
        attention_mask = generate_block_attention_mask(
            (image_sizes // self.config.patch_size).prod(dim=1),
            patch_embeds,
        )

    out = self.transformer(
        patch_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        position_embeddings=position_embeddings,
        output_hidden_states=output_hidden_states,
        output_attentions=output_attentions,
        return_dict=True,
        **kwargs,
    )
    return out


def generate_block_attention_mask(num_ids_per_image, tensor):
    dtype = tensor.dtype
    device = tensor.device

    if not isinstance(num_ids_per_image, torch.Tensor):
        num_ids_per_image = torch.as_tensor(num_ids_per_image, device=device, dtype=torch.long)
    else:
        num_ids_per_image = num_ids_per_image.to(device=device, dtype=torch.long)

    # Build per-token block ids: [0 repeated n0, 1 repeated n1, ...].
    block_ids = torch.repeat_interleave(
        torch.arange(num_ids_per_image.numel(), device=device), num_ids_per_image
    )
    # same_block[i, j] is True if tokens i and j belong to the same image block.
    same_block = block_ids[:, None] == block_ids[None, :]

    # Mask: 0 inside blocks, 1 outside blocks (match previous function's output), tensor-only.
    mask = (~same_block).to(dtype)
    d_min = torch.finfo(dtype).min
    mask *= d_min

    return mask


@torch.library.custom_op("auto_deploy::pixtral_unfold_to_2d_grid", mutates_args={})
def _unfold_to_2d_grid(
    image_features: torch.Tensor,
    image_sizes: torch.Tensor,
    patch_size: int,
    spatial_merge_size: int,
) -> torch.Tensor:
    image_sizes = [
        (image_size[0] // patch_size, image_size[1] // patch_size) for image_size in image_sizes
    ]

    tokens_per_image = [h * w for h, w in image_sizes]
    d = image_features.shape[-1]

    permuted_tensor = []
    for image_index, image_tokens in enumerate(image_features.split(tokens_per_image)):
        # Reshape image_tokens into a 2D grid
        h, w = image_sizes[image_index]
        image_grid = image_tokens.view(h, w, d).permute(2, 0, 1).unsqueeze(0)
        grid = torch.nn.functional.unfold(
            image_grid, kernel_size=spatial_merge_size, stride=spatial_merge_size
        )
        grid = grid.view(d * spatial_merge_size**2, -1).t()
        permuted_tensor.append(grid)

    image_features = torch.cat(permuted_tensor, dim=0)

    return image_features


@_unfold_to_2d_grid.register_fake
def _unfold_to_2d_grid_meta(
    image_features: torch.Tensor,
    image_sizes: torch.Tensor,
    patch_size: int,
    spatial_merge_size: int,
):
    embedding_sizes = (image_sizes // patch_size).prod(dim=1)
    spatial_factor = spatial_merge_size * spatial_merge_size
    grid_sizes = embedding_sizes // spatial_factor
    total_size = grid_sizes.sum()

    return image_features.new_empty(total_size, image_features.shape[-1] * spatial_factor)


def _patch_merger_forward(
    self, image_features: torch.Tensor, image_sizes: torch.Tensor
) -> torch.Tensor:
    unfolded_features = torch.ops.auto_deploy.pixtral_unfold_to_2d_grid(
        image_features=image_features,
        image_sizes=image_sizes,
        patch_size=self.patch_size,
        spatial_merge_size=self.spatial_merge_size,
    )
    image_features = self.merging_layer(unfolded_features)
    return image_features


# Somehow there are dtype mismatches at runtime between bfloat16 and float32 without this.
def _pixtral_rms_norm_forward(self, hidden_states):
    input_dtype = torch.bfloat16
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    return self.weight * hidden_states.to(input_dtype)


# NOTE: registered as patch that is disabled by default since it is not used at the moment
@ExportPatchRegistry.register("hf_pixtral_vit")
class PixtralVisionModelPatch(DisabledBaseExportPatch):
    """Patch for `PixtralVisionModel`."""

    def _apply_patch(self):
        """Apply the PixtralVisionModel patch."""
        self.original_values["PixtralVisionModel.forward"] = PixtralVisionModel.forward
        self.original_values["Mistral3PatchMerger.forward"] = Mistral3PatchMerger.forward
        self.original_values["PixtralRMSNorm.forward"] = PixtralRMSNorm.forward

        PixtralVisionModel.forward = _pixtral_forward
        Mistral3PatchMerger.forward = _patch_merger_forward
        PixtralRMSNorm.forward = _pixtral_rms_norm_forward

    def _revert_patch(self):
        """Revert the PixtralVisionModel patch."""
        PixtralVisionModel.forward = self.original_values["PixtralVisionModel.forward"]
        Mistral3PatchMerger.forward = self.original_values["Mistral3PatchMerger.forward"]
        PixtralRMSNorm.forward = self.original_values["PixtralRMSNorm.forward"]


# NOTE: registered as patch that is disabled by default since it is applied globally...
@ExportPatchRegistry.register("hf_pixtral_dtype")
class PixtralDtypePatch(DisabledBaseExportPatch):
    """Patch for `PixtralVisionModel`."""

    def _apply_patch(self):
        """Fix the dtype of pixel_values to align with pixtral weights dtype."""

        def _forward(mod: PixtralVisionModel, pixel_values: torch.Tensor, *args, **kwargs):
            pixel_values = pixel_values.to(mod.patch_conv.weight.dtype)
            return self.original_values["forward"](mod, pixel_values, *args, **kwargs)

        self.original_values["forward"] = PixtralVisionModel.forward
        PixtralVisionModel.forward = _forward

    def _revert_patch(self):
        """Revert the PixtralVisionModel patch."""
        PixtralVisionModel.forward = self.original_values["forward"]


# TODO: figure out how to properly register and apply patches like this that are global
ExportPatchRegistry.create_patch("hf_pixtral_dtype", {"enabled": True}).__enter__()
