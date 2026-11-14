# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Install the TP8 locality MoE state factory (LOCAL_MOE) and its INFO workspace receipt.

Called at the end of the TP8 target's constructor, after every refusal check and before any engine state is built.
It adds no traffic admission and no TP2 locality lifecycle. The constructor skips the call under the kill switch
(TRTLLM_M31_DISABLE_LOCAL_MOE), which ``m31_local_moe_op.make_state`` also reads (it then builds the stock state and
logs the off line).
"""

from . import m31_local_moe_op as local_module
from .serving_hook import install


def install_tp8(model, mapping):
    if mapping.tp_size != 8 or mapping.tp_rank not in range(8):
        raise ValueError("TP8 locality bootstrap requires TP8 rank")
    previous = getattr(model, "_tp8_locality_bootstrap", None)
    identity = (mapping.tp_size, mapping.tp_rank)
    if previous is not None:
        if previous != identity:
            raise RuntimeError("TP8 locality bootstrap rank changed within process")
        return
    stock = model.M3MoeState

    # Only the 8-row decode call with the shared expert takes the variant; make_state owns the kill switch.
    def make_state(device, num_tokens, num_ctas=None, shared=False, **kwargs):
        if shared and num_tokens == 8:
            return local_module.make_state(
                device, num_tokens, num_ctas=num_ctas, shared=True, **kwargs
            )
        return stock(device, num_tokens=num_tokens, num_ctas=num_ctas, shared=shared, **kwargs)

    install(model, rank=mapping.tp_rank, make_state=make_state)
    model._tp8_locality_bootstrap = identity
