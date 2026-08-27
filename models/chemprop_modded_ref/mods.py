import torch
import torch.nn.functional as F
from chemprop.conf import DEFAULT_ATOM_FDIM, DEFAULT_BOND_FDIM, DEFAULT_HIDDEN_DIM
from chemprop.data import BatchMolGraph
from chemprop.models import MPNN
from chemprop.nn import Activation, BondMessagePassing, GraphTransform, ScaleTransform
from chemprop.schedulers import build_NoamLike_LRSched
from torch import Tensor, optim
from torch.nn.modules import Module


class ResidualFFN(torch.nn.Module):
    def __init__(
        self, dims: int, dropout: float = 0.0, zero_init: bool = True
    ) -> None:
        super().__init__()

        self.norm = torch.nn.LayerNorm(dims)
        self.gate_up_proj = torch.nn.Linear(dims, 2 * dims, bias=False)
        self.down_proj = torch.nn.Linear(dims, dims, bias=False)
        self.dropout = torch.nn.Dropout(dropout)

        # Zero the output projection so the block starts as an exact identity:
        # `forward` returns `0 + inp` on the first step, which makes the network
        # numerically equal to stock chemprop at init. The block only moves away
        # from identity if the gradient says it earns its place, instead of
        # perturbing a message-passing recurrence that already works.
        #
        # `zero_init=False` restores the ordinary random init, which is the
        # control for whether the identity start is what the block needed.
        if zero_init:
            torch.nn.init.zeros_(self.down_proj.weight)

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
        zero_init: bool = True,
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

        self.layer_ffn = ResidualFFN(d_h, dropout=dropout, zero_init=zero_init)

    def update(self, M_t, H_0, H_prev, t):  # type: ignore
        """Calcualte the updated hidden for each edge"""
        H_t = self.W_h(M_t)
        H_t = self.tau(H_0 + H_t)
        H_t = self.dropout(H_t)

        H_t = self.layer_ffn(H_t)
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


class ModdedMPNN(MPNN):
    """MPNN trained with AdamW instead of chemprop's plain Adam.

    Upstream hardcodes `optim.Adam(self.parameters(), self.init_lr)`, i.e. no
    weight decay at all. Adam's L2 term is also coupled to the gradient, so on a
    Noam schedule the effective decay would ride the learning rate up through
    warmup and back down through cooldown; AdamW decouples it, and the shrink
    stays a fixed fraction per step.

    Only matrices are decayed. Biases and the LayerNorm/BatchNorm gains are 1-D
    and are left alone -- pulling a norm's gain toward zero only rescales what
    the next layer has to undo, and there are too few of those parameters for
    the regularisation to buy anything.
    """

    def __init__(self, *args, weight_decay: float = 0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_decay = weight_decay
        self.hparams["weight_decay"] = weight_decay

    def configure_optimizers(self):
        decay, no_decay = [], []
        for p in self.parameters():
            if p.requires_grad:
                (decay if p.ndim >= 2 else no_decay).append(p)

        opt = optim.AdamW(
            [
                {"params": decay, "weight_decay": self.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            self.init_lr,
        )

        if self.trainer.train_dataloader is None:
            # Touch `estimated_stepping_batches` so `num_training_batches` is populated;
            # see the same workaround in `MPNN.configure_optimizers`.
            self.trainer.estimated_stepping_batches
        steps_per_epoch = self.trainer.num_training_batches
        warmup_steps = self.warmup_epochs * steps_per_epoch
        cooldown_steps = (self.trainer.max_epochs - self.warmup_epochs) * steps_per_epoch

        lr_sched = build_NoamLike_LRSched(
            opt, warmup_steps, cooldown_steps, self.init_lr, self.max_lr, self.final_lr
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": lr_sched, "interval": "step"},
        }
