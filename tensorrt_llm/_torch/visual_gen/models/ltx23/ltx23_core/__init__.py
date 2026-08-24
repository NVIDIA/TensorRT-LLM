# SPDX-FileCopyrightText: Copyright (c) 2025-2026 Lightricks Ltd.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-LTX-2
"""LTX-2.3-only core components.

- connector.py: split video/audio feature extractor and gated 8-layer connectors.
- modality.py: adds the global sigma alongside the LTX-2 fields.
- video_vae_ltx23.py: LTX2VideoDecoder subclass with the LTX-2.3 channel recipe.
- vocoder_ltx23.py: BigVGAN-v2 AMP1 vocoder plus fp32 bandwidth extension.
"""
