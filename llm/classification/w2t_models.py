import torch
import torch.nn as nn
import torch.nn.functional as F

import math

class MultiQueryRankPool(nn.Module):
    """
    PMA-style multi-query pooling for rank tokens.
    Inputs:
      x:     [B, r, H]
      sigma: [B, r] or [B, r, 1]  (optional, used as soft prior)
      mask:  [B, r] (True=pad)
    Output:
      pooled: [B, H]
    """
    def __init__(self, hidden_dim: int, num_queries: int = 4, dropout: float = 0.1, sigma_prior_alpha: float = 0.5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.sigma_prior_alpha = float(sigma_prior_alpha)

        # learnable queries: [M, H]
        # self.queries = nn.Parameter(torch.randn(num_queries, hidden_dim) / math.sqrt(hidden_dim))
        self.queries = nn.Parameter(torch.zeros(num_queries, hidden_dim))
        nn.init.normal_(self.queries, std=0.02)

        self.attn_drop = nn.Dropout(dropout)

        # fuse M pooled vectors back to H
        self.out = nn.Sequential(
            nn.Linear(num_queries * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    @staticmethod
    def _sigma_feat(sigma: torch.Tensor) -> torch.Tensor:
        if sigma.dim() == 3:
            sigma = sigma.squeeze(-1)
        sigma = torch.log1p(torch.clamp(sigma, min=0.0))
        return sigma  # [B, r]

    def forward(self, x: torch.Tensor, sigma: torch.Tensor | None = None, mask: torch.Tensor | None = None) -> torch.Tensor:
        B, r, H = x.shape
        q = self.queries.unsqueeze(0).expand(B, -1, -1)  # [B, M, H]

        # logits: [B, M, r]
        logits = torch.einsum("bmh,brh->bmr", q, x) / math.sqrt(H)

        # optional sigma prior (soft, not hard)
        if sigma is not None and self.sigma_prior_alpha != 0.0:
            s = self._sigma_feat(sigma)                  # [B, r]
            logits = logits + self.sigma_prior_alpha * s.unsqueeze(1)

        if mask is not None:
            logits = logits.masked_fill(mask.unsqueeze(1), -1e9)

        attn = torch.softmax(logits, dim=-1)            # [B, M, r]
        attn = self.attn_drop(attn)

        pooled = torch.einsum("bmr,brh->bmh", attn, x)  # [B, M, H]
        pooled = pooled.reshape(B, self.num_queries * H)
        return self.out(pooled)

class SVDBilinearProjector(nn.Module):
    def __init__(self, d_out: int, d_in: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.proj_u = nn.Linear(d_out, hidden_dim)
        self.proj_v = nn.Linear(d_in, hidden_dim)

        self.u_norm = nn.LayerNorm(hidden_dim)
        self.v_norm = nn.LayerNorm(hidden_dim)

        self.fuse = nn.Linear(hidden_dim * 2, hidden_dim)

        self.sigma_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
        )
        self.film_eps = nn.Parameter(torch.tensor(0.0))
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _stabilize_sigma(sigma: torch.Tensor) -> torch.Tensor:
        # sigma: [B,r,1] or [B,r]
        if sigma.dim() == 2:
            sigma = sigma.unsqueeze(-1)
        sigma = torch.log1p(torch.clamp(sigma, min=0.0))
        sigma = (sigma - sigma.mean(dim=1, keepdim=True)) / (sigma.std(dim=1, keepdim=True) + 1e-6)
        return sigma

    def forward(self, u: torch.Tensor, v: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        sigma_n = self._stabilize_sigma(sigma)

        h_u = F.relu(self.u_norm(self.proj_u(u)))
        h_v = F.relu(self.v_norm(self.proj_v(v)))

        x = self.fuse(torch.cat([h_u, h_v], dim=-1))  # [B,r,H]

        film = self.sigma_mlp(sigma_n)                # [B,r,2H]
        scale, shift = film.chunk(2, dim=-1)
        eps = self.film_eps
        x = x * (1.0 + eps * torch.tanh(scale)) + eps * shift

        x = self.out_norm(x)
        x = self.dropout(F.gelu(x))
        return x


class AttentionPool(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1, bias=False),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        logits = self.score(x).squeeze(-1)  # [B, P]
        if mask is not None:
            logits = logits.masked_fill(mask, -1e9)
        w = torch.softmax(logits, dim=1).unsqueeze(-1)  # [B, P, 1]
        return torch.sum(x * w, dim=1)


class FullTransformer(nn.Module):
    def __init__(
        self,
        input_dims,                # [(d_out_p, d_in_p), ...] for each position p
        layer_ids,                 # list[int] length P
        module_ids,                # list[int] length P
        num_layers: int,
        num_modules: int,
        hidden_dim: int = 128,
        out_dim: int = 40,
        num_rank_layers: int = 1,
        num_layer_layers: int = 2,
        nhead: int = 8,
        dropout: float = 0.1,
        mlp_dim: int = 128,
        sign_aug_prob: float = 0.0,
        rank_perm_prob: float = 0.0,

    ):
        super().__init__()
        assert len(input_dims) == len(layer_ids) == len(module_ids), "input_dims/layer_ids/module_ids must align."
        self.num_pos = len(input_dims)
        self.hidden_dim = hidden_dim
        self.sign_aug_prob = float(sign_aug_prob)
        self.rank_perm_prob = float(rank_perm_prob)

        # Per-position projector (dims may vary across modules)
        self.projectors = nn.ModuleList([
            SVDBilinearProjector(d_out, d_in, hidden_dim, dropout)
            for (d_out, d_in) in input_dims
        ])
        self.pos_token_norm = nn.LayerNorm(hidden_dim)

        # Store position metadata as buffers (moved with the model)
        self.register_buffer("pos_layer_ids", torch.tensor(layer_ids, dtype=torch.long), persistent=False)
        self.register_buffer("pos_module_ids", torch.tensor(module_ids, dtype=torch.long), persistent=False)

        # Position embeddings
        self.layer_embedding = nn.Embedding(num_layers, hidden_dim)
        self.module_embedding = nn.Embedding(num_modules, hidden_dim)

        # Optional intra-position (rank-wise) Transformer, shared across positions
        self.rank_encoder = None
        if num_rank_layers > 0:
            rank_enc_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=nhead,
                dim_feedforward=4 * hidden_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.rank_encoder = nn.TransformerEncoder(rank_enc_layer, num_layers=num_rank_layers)

        # Inter-position Transformer
        pos_enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.layer_encoder = nn.TransformerEncoder(pos_enc_layer, num_layers=num_layer_layers)

        self.rank_pool = MultiQueryRankPool(
            hidden_dim=hidden_dim,
            num_queries=4,
            dropout=dropout,
            sigma_prior_alpha=0.5,
        )
        self.pos_pool = AttentionPool(hidden_dim, dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.LayerNorm(mlp_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, out_dim),
        )

    def forward(self, data, src_key_padding_masks=None):
        if isinstance(data, dict):
            x_list = data["features"]
        else:
            x_list = data

        device = x_list[0][0].device
        do_perm = self.training and self.rank_perm_prob > 0 and torch.rand((), device=device) < self.rank_perm_prob
        do_sign = self.training and self.sign_aug_prob > 0 and torch.rand((), device=device) < self.sign_aug_prob

        pos_tokens = []

        for p, (u, v, s) in enumerate(x_list):
            rank_mask = None
            if src_key_padding_masks is not None:
                rank_mask = src_key_padding_masks[p].bool()

            if do_perm:
                B, r, _ = u.shape
                perm = torch.randperm(r, device=u.device)
                u = u[:, perm, :]
                v = v[:, perm, :]
                # s can be [B,r] or [B,r,1]
                if s.dim() == 2:
                    s = s[:, perm]
                else:
                    s = s[:, perm, :]
                if rank_mask is not None:
                    rank_mask = rank_mask[:, perm]

            if do_sign:
                B, r, _ = u.shape
                sign = (torch.randint(0, 2, (B, r, 1), device=u.device).float() * 2 - 1.0)
                u = u * sign
                v = v * sign

            # Rank tokens for this position
            h = self.projectors[p](u, v, s)  # [B, r, H]

            # Optional intra-position rank-wise Transformer (shared)
            if self.rank_encoder is not None:
                h = self.rank_encoder(h, src_key_padding_mask=rank_mask)  # [B, r, H]

            T_p = self.rank_pool(h, sigma=s, mask=rank_mask)                    # [B, H]

            # Add structural position embeddings at the position-level token
            # layer_id = int(self.pos_layer_ids[p].item())
            # module_id = int(self.pos_module_ids[p].item())
            # T_p = T_p + self.layer_embedding.weight[layer_id].view(1, -1) + self.module_embedding.weight[module_id].view(1, -1)

            pos_tokens.append(T_p)

        # Inter-position sequence: [B, P, H]
        H_pos = torch.stack(pos_tokens, dim=1)
        E = self.layer_embedding(self.pos_layer_ids) + self.module_embedding(self.pos_module_ids)
        H_pos = self.pos_token_norm(H_pos + E.unsqueeze(0))  # [B,P,H]
        H_pos = self.layer_encoder(H_pos)  # [B, P, H]

        # Attention pool over positions (supports masked/variable P)
        pos_mask = None
        if isinstance(data, dict) and "pos_mask" in data:
            pos_mask = data["pos_mask"].bool()
        pooled = self.pos_pool(H_pos, mask=pos_mask)  # [B, H]
        return self.classifier(pooled)