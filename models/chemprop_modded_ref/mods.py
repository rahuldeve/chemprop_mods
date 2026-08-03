import torch
from chemprop.conf import DEFAULT_ATOM_FDIM, DEFAULT_BOND_FDIM, DEFAULT_HIDDEN_DIM
from chemprop.data import BatchMolGraph
from chemprop.nn import Activation, BondMessagePassing, GraphTransform, ScaleTransform
from torch import Tensor
from torch.nn.modules import Module


class ResidualFFN(torch.nn.Module):
    def __init__(self, dims: int) -> None:
        super().__init__()

        self.norm = torch.nn.LayerNorm(dims)
        self.gate = torch.nn.Linear(dims, 2 * dims)
        self.up_proj = torch.nn.Sequential(
            torch.nn.Linear(dims, 2 * dims), torch.nn.GELU()
        )
        self.down_proj = torch.nn.Linear(2 * dims, dims)

    def forward(self, inp, res):
        inp_normed = self.norm(inp)
        return self.down_proj(self.gate(inp_normed) * self.up_proj(inp_normed)) + res


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

        self.layer_ffn = torch.nn.ModuleList([ResidualFFN(d_h) for _ in range(depth)])
        self.norms = torch.nn.ModuleList(
            [torch.nn.LayerNorm(d_h) for _ in range(depth)]
        )

    def update(self, M_t, H_0, H_prev, t):  # type: ignore
        """Calcualte the updated hidden for each edge"""
        H_t = self.W_h(M_t)
        H_t = self.tau(self.norms[t](H_0 + H_t))
        H_t = self.dropout(H_t)

        H_t = self.layer_ffn[t](H_t, H_0)
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
