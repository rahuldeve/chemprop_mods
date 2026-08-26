import torch
import torch.nn.functional as F
from chemprop.conf import DEFAULT_ATOM_FDIM, DEFAULT_BOND_FDIM, DEFAULT_HIDDEN_DIM
from chemprop.data import BatchMolGraph
from chemprop.models import MPNN
from chemprop.nn import Activation, BondMessagePassing, GraphTransform, ScaleTransform
from torch import Tensor
from torch.nn.modules import Module


class ResidualFFN(torch.nn.Module):
    def __init__(self, dims: int, dropout: float = 0.0, n_layers: int = 1) -> None:
        super().__init__()

        self.norm = torch.nn.LayerNorm(dims)
        # 4 * dims = a gate and a value branch of 2 * dims each
        self.gate_up_proj = torch.nn.Linear(dims, 4 * dims, bias=False)
        self.down_proj = torch.nn.Linear(2 * dims, dims, bias=False)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, inp: Tensor) -> Tensor:
        gate, up = self.gate_up_proj(self.norm(inp)).chunk(2, dim=-1)
        return self.dropout(self.down_proj(F.silu(gate) * up)) + inp


class ModdedBondMessagePassing(BondMessagePassing):
    def __init__(
        self,
        d_v: int = DEFAULT_ATOM_FDIM,
        d_e: int = DEFAULT_BOND_FDIM,
        d_h: int = DEFAULT_HIDDEN_DIM,
        bias: bool = False,
        depth: int = 3,
        dropout: float = 0,
        activation: str | Module | Activation = Activation.RELU,
        undirected: bool = False,
        d_vd: int | None = None,
        V_d_transform: ScaleTransform | None = None,
        graph_transform: GraphTransform | None = None,
    ):
        super().__init__(
            d_v,
            d_e,
            d_h,
            bias,
            depth,
            dropout,
            activation,
            undirected,
            d_vd,
            V_d_transform,
            graph_transform,
        )

        self.layer_ffn = torch.nn.ModuleList(
            [ResidualFFN(d_h, n_layers=depth) for _ in range(depth)]
        )

    def update(self, M_t, H_0, H_prev, t):  # type: ignore
        """Calcualte the updated hidden for each edge"""
        H_t = self.W_h(M_t)
        H_t = self.tau(H_0 + H_t)
        H_t = self.dropout(H_t)

        H_t = self.layer_ffn[t](H_t)
        return H_t

    def forward(self, bmg: BatchMolGraph, V_d: Tensor | None = None) -> Tensor:
        bmg = self.graph_transform(bmg)
        H_0 = self.initialize(bmg)

        H = self.tau(H_0)
        for t in range(1, self.depth):
            if self.undirected:
                H = (H + H[bmg.rev_edge_index]) / 2

            M = self.message(H, bmg)
            H = self.update(M, H_0, H, t - 1)

        index_torch = bmg.edge_index[1].unsqueeze(1).repeat(1, H.shape[1])
        M = torch.zeros(
            len(bmg.V), H.shape[1], dtype=H.dtype, device=H.device
        ).scatter_reduce_(0, index_torch, H, reduce="sum", include_self=False)
        return self.finalize(M, bmg.V, V_d)
