from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F


vmap_bmm_n0 = torch.vmap(torch.bmm, in_dims=(None, 0))
vmap_bmm_00 = torch.vmap(torch.bmm, in_dims=(0, 0))


class GLNonlin(nn.Module):
    def forward(self, u: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        to_norm = vmap_bmm_00(u, v.swapaxes(-1, -2))
        left_mul = F.relu(torch.sign(to_norm.sum(dim=3)).unsqueeze(-1))
        right_mul = F.relu(torch.sign(to_norm.sum(dim=2)).unsqueeze(-1))
        return left_mul * u, right_mul * v


class EquivariantLinear(nn.Module):
    def __init__(self, n: int, m: int, out_dim: int):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(out_dim // 2, n) / (n**0.5))
        self.w2 = nn.Parameter(torch.randn(out_dim // 2, m) / (m**0.5))

    def forward(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.w1 @ u, self.w2 @ v], dim=1)


class EquivariantMLP(nn.Module):
    def __init__(self, ns: list[int], ms: list[int], hidden_dim: int, n_layers: int, n_input_layers: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_input_layers = n_input_layers
        self.first_layer = nn.ModuleList([EquivariantLinear(n, m, 2 * hidden_dim) for n, m in zip(ns, ms)])

        self.equivariant_layers_u = nn.ParameterList()
        self.equivariant_layers_v = nn.ParameterList()
        for _ in range(1, n_layers):
            layer_u = nn.Parameter(torch.randn(n_input_layers, hidden_dim, hidden_dim) / (hidden_dim**0.5))
            layer_v = nn.Parameter(torch.randn(n_input_layers, hidden_dim, hidden_dim) / (hidden_dim**0.5))
            self.equivariant_layers_u.append(layer_u)
            self.equivariant_layers_v.append(layer_v)
        self.nonlins = nn.ModuleList([GLNonlin() for _ in range(max(0, n_layers - 1))])

    def forward(self, uvs: list[tuple[torch.Tensor, torch.Tensor]]) -> list[tuple[torch.Tensor, torch.Tensor]]:
        new_list = []
        for i, (u, v) in enumerate(uvs):
            out = self.first_layer[i](u, v)
            u_h, v_h = torch.split(out, self.hidden_dim, dim=1)
            new_list.append((u_h, v_h))

        us = torch.stack([uv[0] for uv in new_list], dim=1)
        vs = torch.stack([uv[1] for uv in new_list], dim=1)

        for i in range(max(0, self.n_layers - 1)):
            us_nl, vs_nl = self.nonlins[i](us, vs)
            us = vmap_bmm_n0(self.equivariant_layers_u[i], us_nl)
            vs = vmap_bmm_n0(self.equivariant_layers_v[i], vs_nl)

        out_uvs = []
        for i in range(len(us[0])):
            out_uvs.append((us[:, i, ...], vs[:, i, ...]))
        return out_uvs


class InvariantOutput(nn.Module):
    def __init__(self, n: int, m: int, clip: bool = False):
        super().__init__()
        self.clip = clip
        self.num_elem = 1 if clip else n * m

    def forward(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        uv = torch.bmm(u, v.swapaxes(-1, -2))
        if self.clip:
            return uv.mean(dim=(-1, -2)).unsqueeze(-1)
        return uv


class InvariantHead(nn.Module):
    def __init__(self, ns: list[int], ms: list[int], hidden_dim: int, out_dim: int, clip: bool = False):
        super().__init__()
        self.invariant_outputs = nn.ModuleList([InvariantOutput(n, m, clip=clip) for n, m in zip(ns, ms)])
        total_length = sum([x.num_elem for x in self.invariant_outputs])
        self.linear1 = nn.Linear(total_length, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, uvs: list[tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        out = torch.cat(
            [f(u, v).flatten(start_dim=1) for (u, v), f in zip(uvs, self.invariant_outputs)],
            dim=1,
        )
        out = self.linear1(out)
        out = self.ln1(out)
        out = F.relu(out)
        out = self.linear2(out)
        return out


class GLInvariantMLP(nn.Module):
    def __init__(
        self,
        ns: list[int],
        ms: list[int],
        n_input_layers: int,
        out_dim: int,
        hidden_dim_equiv: int = 128,
        n_layers: int = 1,
        hidden_dim_inv: int = 128,
        clip: bool = False,
    ):
        super().__init__()
        self.equiv_mlp = EquivariantMLP(ns, ms, hidden_dim_equiv, n_layers, n_input_layers)
        ns_h = [hidden_dim_equiv for _ in range(n_input_layers)]
        ms_h = [hidden_dim_equiv for _ in range(n_input_layers)]
        self.invariant_head = InvariantHead(ns_h, ms_h, hidden_dim_inv, out_dim, clip=clip)

    def forward(self, uvs: list[tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        uvs_h = self.equiv_mlp(uvs)
        return self.invariant_head(uvs_h)


class AttentionPool(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1, bias=False),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        logits = self.score(x).squeeze(-1)
        if mask is not None:
            logits = logits.masked_fill(mask, -1e9)
        w = torch.softmax(logits, dim=1).unsqueeze(-1)
        return torch.sum(x * w, dim=1)


def make_mlp_head(in_dim: int, out_dim: int, mlp_dim: int, dropout: float) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_dim, mlp_dim),
        nn.LayerNorm(mlp_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(mlp_dim, out_dim),
    )


class FlattenMLP_Layerwise(nn.Module):
    def __init__(self, layer_dims: list[int], out_dim: int, hidden_dim: int = 256, mlp_dim: int = 128, dropout: float = 0.0):
        super().__init__()
        self.encoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                )
                for d in layer_dims
            ]
        )
        self.pool = AttentionPool(hidden_dim, dropout)
        self.head = make_mlp_head(hidden_dim, out_dim, mlp_dim, dropout)

    def forward(self, x_layers: list[torch.Tensor]) -> torch.Tensor:
        hs = [enc(x_layers[i]) for i, enc in enumerate(self.encoders)]
        h = torch.stack(hs, dim=1)
        z = self.pool(h)
        return self.head(z)


class _CNN1DEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 256, dropout: float = 0.0):
        super().__init__()
        c1, c2, c3, c4, c5, c6, c7, c8, c9, c10 = 16, 32, 64, 128, 256, 384, 512, 768, 1024, 64
        self.net = nn.Sequential(
            nn.Conv1d(1, c1, 7, stride=2, padding=3),
            nn.BatchNorm1d(c1),
            nn.ReLU(inplace=True),
            nn.Conv1d(c1, c2, 7, stride=2, padding=3),
            nn.BatchNorm1d(c2),
            nn.ReLU(inplace=True),
            nn.Conv1d(c2, c3, 7, stride=2, padding=3),
            nn.BatchNorm1d(c3),
            nn.ReLU(inplace=True),
            nn.Conv1d(c3, c4, 7, stride=2, padding=3),
            nn.BatchNorm1d(c4),
            nn.ReLU(inplace=True),
            nn.Conv1d(c4, c5, 7, stride=2, padding=3),
            nn.BatchNorm1d(c5),
            nn.ReLU(inplace=True),
            nn.Conv1d(c5, c6, 7, stride=2, padding=3),
            nn.BatchNorm1d(c6),
            nn.ReLU(inplace=True),
            nn.Conv1d(c6, c7, 7, stride=2, padding=3),
            nn.BatchNorm1d(c7),
            nn.ReLU(inplace=True),
            nn.Conv1d(c7, c8, 7, stride=2, padding=3),
            nn.BatchNorm1d(c8),
            nn.ReLU(inplace=True),
            nn.Conv1d(c8, c9, 7, stride=2, padding=3),
            nn.BatchNorm1d(c9),
            nn.ReLU(inplace=True),
            nn.Conv1d(c9, c10, 7, stride=2, padding=3),
            nn.BatchNorm1d(c10),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Sequential(
            nn.Linear(c10, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.net(x)
        x = self.pool(x).squeeze(-1)
        return self.proj(x)


class CNN1D_Layerwise(nn.Module):
    def __init__(self, layer_dims: list[int], out_dim: int, hidden_dim: int = 256, mlp_dim: int = 128, dropout: float = 0.0):
        super().__init__()
        self.encoders = nn.ModuleList([_CNN1DEncoder(hidden_dim, dropout) for _ in layer_dims])
        self.pool = AttentionPool(hidden_dim, dropout)
        self.head = make_mlp_head(hidden_dim, out_dim, mlp_dim, dropout)

    def forward(self, x_layers: list[torch.Tensor]) -> torch.Tensor:
        hs = [self.encoders[i](x_layers[i]) for i in range(len(self.encoders))]
        h = torch.stack(hs, dim=1)
        z = self.pool(h)
        return self.head(z)


class TokenViT_Encoder(nn.Module):
    def __init__(self, token_size: int, embed_dim: int = 192, depth: int = 4, nhead: int = 8, dropout: float = 0.0):
        super().__init__()
        self.proj = nn.Linear(token_size, embed_dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos = None

        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=4 * embed_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        nn.init.normal_(self.cls, std=0.02)

    def _build_pos(self, t: int, d: int, device: torch.device) -> torch.Tensor:
        pos = torch.zeros(1, 1 + t, d, device=device)
        nn.init.normal_(pos, std=0.02)
        return pos

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        x = self.proj(x)
        cls = self.cls.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1)
        if (self.pos is None) or (self.pos.shape[1] != 1 + t) or (self.pos.device != x.device):
            self.pos = self._build_pos(t, x.shape[-1], x.device)
        x = x + self.pos
        x = self.encoder(x)
        x = self.norm(x[:, 0])
        return x


class TokenViT_Layerwise(nn.Module):
    def __init__(
        self,
        num_layers: int,
        out_dim: int,
        token_size: int = 2048,
        embed_dim: int = 192,
        depth: int = 4,
        nhead: int = 8,
        mlp_dim: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.proj = nn.Linear(token_size, embed_dim)
        self.layer_embed = nn.Embedding(num_layers, embed_dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, embed_dim))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=4 * embed_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = make_mlp_head(embed_dim, out_dim, mlp_dim, dropout)
        nn.init.normal_(self.cls, std=0.02)

    def _build_sinusoidal_pos(self, seq_len: int, dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        position = torch.arange(seq_len, device=device, dtype=dtype).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, device=device, dtype=dtype) * (-math.log(10000.0) / dim))
        pe = torch.zeros(1, seq_len, dim, device=device, dtype=dtype)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        if dim > 1:
            cos_width = pe[0, :, 1::2].shape[1]
            pe[0, :, 1::2] = torch.cos(position * div_term[:cos_width])
        return pe

    def _from_layerwise_batch(self, x_layers: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz = x_layers[0].shape[0]
        tokens = []
        masks = []
        layer_ids = []
        for lid, x in enumerate(x_layers):
            if x.dim() != 3:
                raise RuntimeError("TokenViT expects each layer tensor with shape [B, T, token_size].")
            if x.shape[0] != bsz:
                raise RuntimeError("Inconsistent batch size across layers.")
            tokens.append(x)
            masks.append(torch.sum(torch.abs(x), dim=-1) == 0)
            layer_ids.append(torch.full((bsz, x.shape[1]), lid, dtype=torch.long, device=x.device))
        tok = torch.cat(tokens, dim=1)
        pad = torch.cat(masks, dim=1)
        lid = torch.cat(layer_ids, dim=1)
        return tok, pad, lid

    def forward(self, x) -> torch.Tensor:
        if isinstance(x, dict):
            tokens = x["tokens"]
            padding_mask = x.get("padding_mask", None)
            layer_ids = x.get("layer_ids", None)
        else:
            tokens, padding_mask, layer_ids = self._from_layerwise_batch(x)

        bsz, t, _ = tokens.shape
        h = self.proj(tokens)

        if layer_ids is not None:
            layer_ids = layer_ids.clamp(min=0, max=self.num_layers - 1)
            h = h + self.layer_embed(layer_ids)

        cls = self.cls.expand(bsz, -1, -1)
        h = torch.cat([cls, h], dim=1)
        h = h + self._build_sinusoidal_pos(h.shape[1], self.embed_dim, h.device, h.dtype)

        src_key_padding_mask = None
        if padding_mask is not None:
            cls_mask = torch.zeros((bsz, 1), dtype=torch.bool, device=padding_mask.device)
            src_key_padding_mask = torch.cat([cls_mask, padding_mask.bool()], dim=1)

        h = self.encoder(h, src_key_padding_mask=src_key_padding_mask)
        z = self.norm(h[:, 0])
        return self.head(z)
