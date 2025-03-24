from typing import Optional, Union

import torch

from .distributed import PPComm


class PipelineInterface:
    """
    A container class for passing intermediate tensors between pipeline parallel ranks.

    It contains two intermediate tensors: [hidden_states, residual], supporting:
    - Dict access: pp['hidden_states'], pp['residual']
    - Unpacking: hidden, residual = pp
    - PP communication: pp.send(), pp.recv()
    - Slicing: pp[start:end]

    Note: When using this interface in pp, the packing/unpacking and send/recv
    operations must be used symmetrically within stage and between succsive ranks.
    """
    _pp_comm = None

    def __init__(self,
                 hidden_states: Optional[torch.Tensor] = None,
                 residual: Optional[torch.Tensor] = None):
        self.hidden_states = hidden_states
        self.residual = residual
        self.tag = 1234

    @classmethod
    def init_pp_comm(cls, mapping):
        """Initialize PPComm once at startup"""
        cls._pp_comm = PPComm(mapping)

    def __getitem__(self, key: Union[str, slice]):
        if isinstance(key, str):
            if key == 'hidden_states':
                return self.hidden_states
            elif key == 'residual':
                return self.residual
            raise KeyError(f"Unknown key: {key}")
        elif isinstance(key, slice):
            return PipelineInterface(hidden_states=self.hidden_states[key] if
                                     self.hidden_states is not None else None,
                                     residual=self.residual[key]
                                     if self.residual is not None else None)

    def __setitem__(self, key: Union[str, slice], value: torch.Tensor):
        if isinstance(key, str):
            if key == 'hidden_states':
                self.hidden_states = value
            elif key == 'residual':
                self.residual = value
            else:
                raise KeyError(f"Unknown key: {key}")
        elif isinstance(key, slice):
            if self.hidden_states is not None:
                self.hidden_states[key] = value
            if self.residual is not None:
                self.residual[key] = value

    def __iter__(self):
        return iter((self.hidden_states, self.residual))

    def recv(self):
        """Receive tensors from previous rank."""
        if self.hidden_states is not None:
            self._pp_comm.recv(self.hidden_states, tag=self.tag)
        if self.residual is not None:
            self._pp_comm.recv(self.residual, tag=self.tag)

    def send(self):
        """Send tensors to next rank."""
        if self.hidden_states is not None:
            self._pp_comm.send(self.hidden_states, tag=self.tag)
        if self.residual is not None:
            self._pp_comm.send(self.residual, tag=self.tag)
