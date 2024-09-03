from typing import Optional

from tensorrt_llm.functional import Tensor, silu
from tensorrt_llm.layers import ColumnLinear
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.module import Module, ModuleList

from ..._utils import str_dtype_to_trt


class ResBlock(Module):

    def __init__(self,
                 exit_dim: int,
                 dtype: Optional[str],
                 mapping: Mapping = Mapping()):
        super().__init__()
        self.linear = ColumnLinear(
            exit_dim,
            exit_dim,
            bias=True,
            dtype=dtype,
            tp_group=mapping.tp_group,
            tp_size=mapping.tp_size,
            gather_output=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + silu(self.linear(x))


class Drafter(Module):

    def __init__(
            self,
            num_layers: int,
            hidden_size: int,
            exit_dim: int,
            vocab_size: int,
            dtype: Optional[str] = None,
            is_rnn: bool = False,
            mapping: Mapping = Mapping(),
    ):
        super().__init__()
        self.num_layers = num_layers
        self.is_rnn = is_rnn
        self.dtype = str_dtype_to_trt(dtype)

        input_dim = 2 * hidden_size
        self.input_proj = (None if input_dim == exit_dim else ColumnLinear(
            input_dim,
            exit_dim,
            bias=True,
            dtype=dtype,
            tp_group=mapping.tp_group,
            tp_size=mapping.tp_size,
            gather_output=True,
        ))

        self.layers = ModuleList([
            ResBlock(exit_dim, dtype, mapping) for _ in range(self.num_layers)
        ])
        self.lm_head = ColumnLinear(
            exit_dim,
            vocab_size,
            bias=False,
            dtype=dtype,
            tp_group=mapping.tp_group,
            tp_size=mapping.tp_size,
            gather_output=True,
        )

        if is_rnn:
            self.rnn_u = ColumnLinear(
                hidden_size,
                hidden_size,
                bias=True,
                dtype=dtype,
                tp_group=mapping.tp_group,
                tp_size=mapping.tp_size,
                gather_output=True,
            )
            self.rnn_w = ColumnLinear(
                hidden_size,
                hidden_size,
                bias=False,
                dtype=dtype,
                tp_group=mapping.tp_group,
                tp_size=mapping.tp_size,
                gather_output=True,
            )
        return

    @classmethod
    def from_config(cls, config, vocab_size_padded):
        kwargs = {
            "num_layers": config.redrafter_num_layers,
            "hidden_size": config.redrafter_hidden_size,
            "exit_dim": config.redrafter_exit_dim,
            "vocab_size": vocab_size_padded,
            "dtype": config.dtype,
            "is_rnn": config.redrafter_is_rnn,
            "mapping": config.mapping,
        }
        return cls(**kwargs)

    def forward(self, x: Tensor) -> Tensor:
        hidden_states = self.input_proj(x) if self.input_proj is not None else x
        for layer in self.layers:
            hidden_states = layer(hidden_states)

        return self.lm_head(hidden_states)

    def rnn_embed(self, x: Tensor, prev: Tensor = None) -> Tensor:
        assert self.is_rnn, "This function should not be called when redrafter_is_rnn is false."
        w_embd = self.rnn_w(x)
        return w_embd if prev is None else w_embd + self.rnn_u(prev)
