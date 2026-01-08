import torch
from torch.export import Dim

from tensorrt_llm._torch.auto_deploy.custom_ops.flashinfer_fused_add_rms_norm import *  # noqa
from tensorrt_llm._torch.auto_deploy.custom_ops.rms_norm import *  # noqa
from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.transform.optimizer import InferenceOptimizer
from tensorrt_llm._torch.auto_deploy.utils.node_utils import is_op


class TestModel(torch.nn.Module):
    def __init__(self, hidden_size=128, eps=1e-5):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.ones(hidden_size, device="cuda", dtype=torch.bfloat16)
        )
        self.eps = eps

    def forward(self, x, residual):
        added = x + residual
        cast = added.to(torch.bfloat16)
        norm = torch.ops.auto_deploy.flashinfer_rms_norm(cast, self.weight, self.eps)
        return norm, added


def _run_test(model):
    # The replacement uses flashinfer_fused_add_rms_norm python wrapper which calls the inplace op
    # auto_deploy::flashinfer_fused_add_rms_norm_inplace
    op = torch.ops.auto_deploy.flashinfer_fused_add_rms_norm_inplace

    def checker(gm):
        return any(is_op(n, op) for n in gm.graph.nodes)

    bsz, seq_len, hidden = 2, 8, 128
    # Inputs should be bfloat16
    x = torch.randn(bsz, seq_len, hidden, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn(bsz, seq_len, hidden, device="cuda", dtype=torch.bfloat16)

    # Dynamic shapes
    dyn_batch_size = Dim.DYNAMIC
    ds_x = {0: dyn_batch_size}
    ds_res = {0: dyn_batch_size}

    gm = torch_export_to_gm(model, args=(x, residual), dynamic_shapes=(ds_x, ds_res), clone=True)

    gm_transformed = InferenceOptimizer(
        None,
        {
            "fuse_add_rms_norm": {
                "stage": "post_load_fusion",
            },
        },
    )(None, gm)

    # Check if transform happened
    if not checker(gm_transformed):
        raise AssertionError(
            "flashinfer_fused_add_rms_norm_inplace op not found in transformed graph"
        )

    # Validation
    # Clone inputs because the fused op is inplace
    x_in = x.clone()
    res_in = residual.clone()

    # The fused op is inplace, so inputs x_in and res_in will be modified.
    # gm_transformed returns (x_in, res_in) which are the modified tensors.
    y_transformed = gm_transformed(x_in, res_in)

    y_model = model(x.clone(), residual.clone())
    torch.testing.assert_close(y_transformed[0], y_model[0], atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(y_transformed[1], y_model[1], atol=1e-2, rtol=1e-2)


def test_fuse_add_rms_norm():
    model = TestModel()
    _run_test(model)
