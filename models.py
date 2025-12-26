import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from timm import create_model

#--------------------------------------------------
# Pytorch implemenattion of Mamba
#---------------------------------------------------
class MambaPyTorch(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        hidden = d_model * expand

        self.in_proj = nn.Linear(d_model, hidden * 2)

        self.conv = nn.Conv1d(
            hidden, hidden,
            kernel_size=d_conv,
            padding=d_conv // 2,
            groups=hidden
        )

        self.A = nn.Parameter(torch.randn(hidden, d_state))
        self.B = nn.Parameter(torch.randn(hidden, d_state))

        self.out_proj = nn.Linear(hidden, d_model)
        self.activation = nn.SiLU()

    def forward(self, x, mask=None):
        B, T, D = x.shape

        inp = self.in_proj(x)
        H = inp.shape[-1] // 2

        gate = torch.sigmoid(inp[:, :, :H])
        update = self.activation(inp[:, :, H:]) #SilU

        u = update.permute(0, 2, 1)
        u = self.conv(u)
        u = u.permute(0, 2, 1)

        h = torch.zeros(B, self.d_state, device=x.device, dtype=x.dtype)
        outputs = []

        A = self.A
        Bmat = self.B

        for t in range(T):
            ut = u[:, t]          # (B, H)
            gt = gate[:, t]       # (B, H)

            Bu = ut.matmul(Bmat)  # (B, d_state)
            h_new = h + Bu

            out_t = h_new.matmul(A.t())

            out_t = gt * out_t

            if mask is not None:
                m = mask[:, t].float().unsqueeze(1)
                h_new = h_new * m + h * (1 - m)
                out_t = out_t * m

            outputs.append(out_t.unsqueeze(1))
            h = h_new

        out = torch.cat(outputs, dim=1)
        return self.out_proj(out)

# --------------------------------------------------
# Utility: masks from lengths
# --------------------------------------------------
def make_mask_from_lengths(lengths, max_len):
    B = lengths.size(0)
    device = lengths.device
    idx = torch.arange(max_len, device=device).unsqueeze(0).expand(B, -1)
    mask = idx < lengths.unsqueeze(1)
    return mask  # bool

# --------------------------------------------------
# PointNet++ Frame Encoder (simplified PointNet variant)
# --------------------------------------------------
class PointNetFrameEncoder(nn.Module):
    """
    Input:  (B*T, N, 3)
    Output: (B*T, feat_dim)
    """
    def __init__(self, feat_dim=256, hidden=128):
        super().__init__()
        self.mlp1 = nn.Sequential(
            nn.Conv1d(3, hidden, 1, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
        )
        self.mlp2 = nn.Sequential(
            nn.Conv1d(hidden, hidden, 1, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
        )
        self.proj = nn.Linear(hidden, feat_dim)

    def forward(self, pts):
        # pts: (B*T, N, 3)
        x = pts.transpose(1, 2).contiguous()  # (B*T, 3, N)
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = torch.max(x, dim=2)[0]           # (B*T, hidden)
        x = self.proj(x)                     # (B*T, feat_dim)
        return x


# --------------------------------------------------
# Projection head
# --------------------------------------------------
class ProjectionHead(nn.Module):
    def __init__(self, in_dim, out_dim=128, hidden=512):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, x):
        return self.fc(x)


# --------------------------------------------------
# RD_Encoder (updated for input shape B, T, H, W)
# --------------------------------------------------
class RD_Encoder(nn.Module):
    """
    Range–Doppler encoder.
    Input:  x -> (B, T, H, W)
            lengths -> (B,)
    Output: embeddings -> (B, emb_dim)
    """

    def __init__(
        self,
        in_channels=1,
        backbone="convnext_tiny",
        emb_dim=128,
        mamba_layers=2,
        mamba_d_state=8,
    ):
        super().__init__()

        # ----------------------------------------------------------
        # 1. ConvNeXt backbone (feature extractor only)
        # ----------------------------------------------------------
        self.backbone = create_model(
            backbone,
            pretrained=False,
            in_chans=in_channels,
            features_only=True
        )

        # ConvNeXt-Tiny → last feature map has 768 channels
        self.frame_feat_dim = self.backbone.feature_info[-1]["num_chs"]  # 768

        # Global average pooling → (B*T, 768)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # ----------------------------------------------------------
        # 2. Temporal model (pure PyTorch Mamba)
        # ----------------------------------------------------------
        self.mamba_layers = nn.ModuleList([
            MambaPyTorch(
                d_model=self.frame_feat_dim,
                d_state=mamba_d_state,
                d_conv=4,
                expand=2
            )
            for _ in range(mamba_layers)
        ])

        # ----------------------------------------------------------
        # 3. Projection head
        # ----------------------------------------------------------
        self.norm = nn.LayerNorm(self.frame_feat_dim)
        self.proj = nn.Sequential(
            nn.Linear(self.frame_feat_dim, self.frame_feat_dim),
            nn.GELU(),
            nn.Linear(self.frame_feat_dim, emb_dim)
        )

    def forward(self, x, lengths):
        """
        x: (B, T, H, W)  --> channel=1 implicit
        lengths: (B,)
        """
        B, T, H, W = x.shape
        device = x.device

        # Add channel dim → (B, T, 1, H, W)
        x = x.unsqueeze(2)

        # Flatten batch and time → (B*T, 1, H, W)
        x_flat = x.reshape(B * T, 1, H, W)

        # Forward through ConvNeXt
        feats = self.backbone(x_flat)       # list of feature maps
        last = feats[-1]                    # (B*T, 768, h, w)

        pooled = self.global_pool(last)     # (B*T, 768, 1, 1)
        frame_feats = pooled.reshape(B * T, self.frame_feat_dim)  # (B*T, 768)

        # Reshape to sequence format
        seq = frame_feats.view(B, T, self.frame_feat_dim)  # (B, T, 768)

        # Build mask
        mask = make_mask_from_lengths(lengths, T)  # (B, T)

        # Temporal Mamba layers
        for layer in self.mamba_layers:
            seq = layer(seq, mask=mask)
        seq = self.norm(seq)
        seq = self.proj(seq)
        # Length-aware mean pooling
        # mask_f = mask.unsqueeze(-1).float()     # (B, T, 1)
        # summed = (seq * mask_f).sum(dim=1)
        # denom = lengths.clamp(min=1).unsqueeze(1).float()
        # pooled = summed / denom                 # (B, 768)
        #basically mean pooling over valid frames
        # pooled = self.norm(pooled)
        # emb = self.proj(pooled)
        # emb = F.normalize(emb, dim=1)

        return seq

# -----------------------
# Patch embedding (2D)
# -----------------------
class PatchEmbed2D(nn.Module):
    """
    Extract non-overlapping patches from single-channel frames and project to embed_dim.
    Input: (B, 1, H, W)
    Output: (B, n_patches, embed_dim)
    """
    def __init__(self, img_size=(64,64), patch_size=8, embed_dim=384):
        super().__init__()
        H, W = img_size
        assert H % patch_size == 0 and W % patch_size == 0, "H,W must be divisible by patch_size"
        self.patch_size = patch_size
        self.n_h = H // patch_size
        self.n_w = W // patch_size
        self.num_patches = self.n_h * self.n_w
        self.embed_dim = embed_dim
        # conv with stride=patch_size to extract patches and linearize
        self.proj = nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size)
        # learnable 2D positional embedding for patches (num_patches)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))

    def forward(self, x):
        # x: (B, 1, H, W)
        x = self.proj(x)                         # (B, embed_dim, n_h, n_w)
        B, C, Hc, Wc = x.shape
        x = x.flatten(2).transpose(1, 2)         # (B, num_patches, embed_dim)
        x = x + self.pos_embed
        return x  # (B, num_patches, embed_dim)


# -----------------------
# Transformer block (MLP + MultiheadAttention)
# -----------------------
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x, key_padding_mask: Optional[torch.Tensor] = None):
        # x: (B, L, D), key_padding_mask: (B, L_bool) where True indicates padding (to be ignored)
        q = self.norm1(x)
        attn_out, _ = self.attn(q, q, q, key_padding_mask=key_padding_mask)  # attn_out (B,L,D)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


# -----------------------
# RD_ViViT_Encoder
# -----------------------
class RD_ViViT_Encoder(nn.Module):
    """
    ViViT-style Range-Doppler encoder (factorized: spatial transformer per-frame -> temporal transformer).
    Input:
        x: (B, T, H, W)  (single channel implicit)
        lengths: (B,) true lengths
    Output:
        embeddings: (B, emb_dim)  (L2-normalized)
    """
    def __init__(
        self,
        img_size=(64,64),
        patch_size=8,
        spatial_embed_dim=384,
        spatial_depth=2,
        spatial_heads=6,
        temporal_depth=4,
        temporal_heads=8,
        emb_dim=512,               # final projection dimension
        projection_hidden=1024,
        dropout=0.0
    ):
        super().__init__()

        # patch embedding per frame
        self.patch_embed = PatchEmbed2D(img_size=img_size, patch_size=patch_size, embed_dim=spatial_embed_dim)
        self.num_patches = self.patch_embed.num_patches
        self.spatial_dim = spatial_embed_dim

        # spatial transformer blocks operate per-frame. We'll run them on (B*T, num_patches, D)
        self.spatial_blocks = nn.ModuleList([
            TransformerBlock(dim=spatial_embed_dim, num_heads=spatial_heads, mlp_ratio=4.0, dropout=dropout)
            for _ in range(spatial_depth)
        ])
        # after spatial blocks, pool patches to single frame token (mean)
        self.spatial_pool_ln = nn.LayerNorm(spatial_embed_dim)

        # temporal transformer operates on (B, T, spatial_embed_dim)
        self.temporal_blocks = nn.ModuleList([
            TransformerBlock(dim=spatial_embed_dim, num_heads=temporal_heads, mlp_ratio=4.0, dropout=dropout)
            for _ in range(temporal_depth)
        ])
        self.temporal_ln = nn.LayerNorm(spatial_embed_dim)

        # final projection head
        self.proj = nn.Sequential(
            nn.Linear(spatial_embed_dim, projection_hidden),
            nn.GELU(),
            nn.Linear(projection_hidden, emb_dim)
        )

        self.emb_dim = emb_dim

    def forward(self, x, lengths):
        """
        x: (B, T, H, W)
        lengths: (B,)
        returns: (B, emb_dim)
        """
        B, T, H, W = x.shape
        device = x.device

        # ensure channel dim
        x = x.unsqueeze(2)               # (B, T, 1, H, W)
        x_flat = x.view(B * T, 1, H, W)  # (B*T, 1, H, W)

        # ---- spatial patch embedding ----
        patches = self.patch_embed(x_flat)   # (B*T, P, D)
        # run spatial transformer (per-frame)
        bt, P, D = patches.shape
        p = patches
        for blk in self.spatial_blocks:
            p = blk(p)                       # (B*T, P, D)
        # mean pool patches -> per-frame token
        frame_tokens = p.mean(dim=1)         # (B*T, D)
        frame_tokens = self.spatial_pool_ln(frame_tokens)
        frame_tokens = frame_tokens.view(B, T, D)  # (B, T, D)

        # ---- temporal transformer ----
        # create key_padding_mask for temporal attention: True = padding to be ignored
        frame_mask = make_mask_from_lengths(lengths, T)   # (B, T) bool (True = valid)
        # but MultiheadAttention expects key_padding_mask with True indicating **to be ignored**
        attn_key_padding_mask = ~frame_mask                 # (B, T) True where padded

        seq = frame_tokens
        for blk in self.temporal_blocks:
            seq = blk(seq, key_padding_mask=attn_key_padding_mask)

        seq = self.temporal_ln(seq)   # (B, T, D)

        # ---- length-aware mean pooling over valid frames ----
        mask = frame_mask.unsqueeze(-1).float()   # (B, T, 1)
        summed = (seq * mask).sum(dim=1)          # (B, D)
        lengths_clamped = lengths.clamp(min=1).unsqueeze(1).to(device=device).float()
        pooled = summed / lengths_clamped         # (B, D)

        # ---- projection to embedding and normalize ----
        emb = self.proj(pooled)       # (B, emb_dim)
        emb = F.normalize(emb, dim=1)
        return emb


# --------------------------------------------------
# PC_Encoder: PointNet++ frame encoder + TRUE Mamba
# --------------------------------------------------
class PC_Encoder(nn.Module):
    """
    Input:
        x: (B, T, N, 3)
        lengths: (B,)
    Output:
        embeddings: (B, emb_dim)
    """

    def __init__(
        self,
        num_points=64,
        frame_feat_dim=256,
        temporal_hidden=256,
        mamba_d_state=16,
        mamba_layers=2,
        emb_dim=128
    ):
        super().__init__()

        # 1) Per-frame encoder (PointNet++)
        self.frame_enc = PointNetFrameEncoder(feat_dim=frame_feat_dim)

        # 2) Temporal Mamba stack
        self.mamba_layers = nn.ModuleList([
            MambaPyTorch(
                d_model=frame_feat_dim,
                d_state=mamba_d_state,
                d_conv=4,
                expand=2
            )
            for _ in range(mamba_layers)
        ])

        # 3) Projection
        self.norm = nn.LayerNorm(frame_feat_dim)
        self.proj = ProjectionHead(frame_feat_dim, emb_dim, hidden=frame_feat_dim)

    def forward(self, x, lengths):
        """
        x: (B, T, N, 3)
        lengths: (B,)
        """
        B, T, N, C = x.shape
        device = x.device

        # Flatten batch+time for PointNet
        x_reshaped = x.view(B * T, N, C)  # (B*T, N, 3)
        frame_feats = self.frame_enc(x_reshaped)  # (B*T, frame_feat_dim)

        dim = frame_feats.shape[-1]
        seq = frame_feats.view(B, T, dim)        # (B, T, dim)

        # Mask
        mask = make_mask_from_lengths(lengths, T)
        mask_bool = mask.bool()

        # Mamba temporal layers
        for layer in self.mamba_layers:
            seq = layer(seq, mask=mask_bool)   # (B, T, dim)

        # Length-aware pooling
        seq = self.norm(seq)
        seq = self.proj(seq)
        # mask_f = mask.unsqueeze(-1)
        # summed = (seq * mask_f).sum(dim=1)
        # denom = lengths.clamp(min=1).unsqueeze(1).to(seq.dtype)
        # pooled = summed / denom

        # pooled = self.norm(pooled)
        # emb = self.proj(pooled)
        # emb = F.normalize(emb, dim=1)
        return seq

class new_PC_Encoder(nn.Module):
    def __init__(
        self,
        num_points=64,
        frame_feat_dim=256,
        mamba_d_state=16,
        mamba_layers=4,            # slightly deeper
        emb_dim=512                # bigger emb dimension
    ):
        super().__init__()

        self.frame_enc = PointNetFrameEncoder(feat_dim=frame_feat_dim)

        # Pre-Mamba LayerNorm
        self.pre_ln = nn.LayerNorm(frame_feat_dim)

        # Mamba layers
        self.mamba_layers = nn.ModuleList([
            MambaPyTorch(
                d_model=frame_feat_dim,
                d_state=mamba_d_state,
                d_conv=4,
                expand=2
            )
            for _ in range(mamba_layers)
        ])

        # Post-Mamba LayerNorm
        self.post_ln = nn.LayerNorm(frame_feat_dim)

        # Stronger projection head
        self.proj = nn.Sequential(
            nn.Linear(frame_feat_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.Linear(1024, emb_dim),
        )

    def forward(self, x, lengths):
        B, T, N, C = x.shape

        # Flatten frames
        x_flat = x.view(B * T, N, C)
        means = x_flat.mean(dim=1, keepdim=True)         # (B*T, 1, 3)
        stds  = x_flat.std(dim=1, keepdim=True) + 1e-10   # (B*T, 1, 3)

        normalized_x_flat = (x_flat - means) / stds                 # normalized points
        frame_feats = self.frame_enc(normalized_x_flat)

        seq = frame_feats.view(B, T, -1)
        seq = self.pre_ln(seq)

        mask = make_mask_from_lengths(lengths, T)

        for layer in self.mamba_layers:
            seq = layer(seq, mask=mask)

        seq = self.post_ln(seq)

        # Masked mean pooling
        mask_f = mask.unsqueeze(-1).float()
        summed = (seq * mask_f).sum(dim=1)
        denom = lengths.clamp(min=1).unsqueeze(1).float()
        pooled = summed / denom

        emb = self.proj(pooled)
        emb = F.normalize(emb, dim=1)
        return emb


# -------------------------------------------------------------
# 4D Spatio-Temporal Patch Sampling
# -------------------------------------------------------------
class STPatchEmbed(nn.Module):
    """
    Extracts small spatio-temporal patches around each point.
    Produces patch embeddings of dimension C.

    Input: (B, T, N, 3)
    Output: (B, T*N, C)
    """
    def __init__(self, in_dim=3, patch_size=8, embed_dim=128):
        super().__init__()
        self.patch_size = patch_size
        self.mlp = nn.Sequential(
            nn.Linear(in_dim * patch_size, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, x):
        """
        x: (B, T, N, 3)
        returns: (B, T*N, embed_dim)
        """
        B, T, N, C = x.shape

        # simple local patching: take nearest patch_size points
        # for sparse mmWave, simplest is random local grouping
        x = x.view(B, T * N, C)  # (B, TN, 3)

        # create K-nearest grouping by simple chunking
        # for 64 points, patch_size=8 → 8 patches per frame
        P = self.patch_size
        total = T * N
        num_patches = total // P

        x = x[:, :num_patches * P, :]  # trim
        patches = x.view(B, num_patches, P * C)  # (B, num_patches, P*C)

        feats = self.mlp(patches)  # (B, num_patches, embed_dim)
        return feats  # treat (num_patches) as sequence length

# -------------------------------------------------------------
# Self-Attention Block (Transformer Encoder Layer)
# -------------------------------------------------------------
class STTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=4, mlp_ratio=4.0):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim)
        )

    def forward(self, x, mask=None):
        # x: (B, L, D)
        if mask is not None:
            # convert frame mask to token mask
            # mask is (B, L), MultiheadAttention expects True=ignore
            attn_mask = ~mask
        else:
            attn_mask = None

        x2, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x),
                          key_padding_mask=attn_mask)
        x = x + x2

        x2 = self.mlp(self.norm2(x))
        x = x + x2
        return x

# -------------------------------------------------------------
# P4Transformer (simplified ST-Pyramid Transformer)
# -------------------------------------------------------------
class P4Transformer(nn.Module):
    def __init__(
        self,
        output_dim=512,
        hidden_dim=256,
        patch_size=8,
        embed_dim=128,
        depth=4,
        num_heads=4,
        mlp_ratio=4.0
    ):
        super().__init__()

        self.patch_embed = STPatchEmbed(
            in_dim=3,
            patch_size=patch_size,
            embed_dim=embed_dim
        )

        self.blocks = nn.ModuleList([
            STTransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # classification head
        self.head = nn.Sequential(
        nn.Linear(embed_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, output_dim)
        )


    def forward(self, x, lengths):
        """
        x: (B, T, N, 3)
        lengths: (B,)
        returns: logits (B, num_classes)
        """

        B, T, N, C = x.shape

        # ---------------------------------------------------------
        # (1) Normalize each frame's points
        # ---------------------------------------------------------
        x_flat = x.view(B * T, N, C)
        means = x_flat.mean(dim=1, keepdim=True)
        stds = x_flat.std(dim=1, keepdim=True) + 1e-6
        x_flat = (x_flat - means) / stds
        x = x_flat.view(B, T, N, C)

        # ---------------------------------------------------------
        # (2) Patch embedding (B, T*N/P, embed_dim)
        # ---------------------------------------------------------
        tokens = self.patch_embed(x)
        B, L, D = tokens.shape

        # ---------------------------------------------------------
        # (3) Mask: turn frame-mask into token-mask
        # ---------------------------------------------------------
        frame_mask = make_mask_from_lengths(lengths, T)  # (B, T)

        # a patch comes from points → map frame_mask to token_mask:
        # we approximate by repeating each frame-mask value
        tokens_per_frame = (N // self.patch_embed.patch_size)
        token_mask = frame_mask.repeat_interleave(tokens_per_frame, dim=1)[:, :L]

        # ---------------------------------------------------------
        # (4) Transformer blocks
        # ---------------------------------------------------------
        for blk in self.blocks:
            tokens = blk(tokens, mask=token_mask)

        # ---------------------------------------------------------
        # (5) Global average over valid tokens
        # ---------------------------------------------------------
        m = token_mask.unsqueeze(-1).float()        # (B, L, 1)
        summed = (tokens * m).sum(dim=1)            # (B, D)
        denom = m.sum(dim=1).clamp(min=1e-6)        # (B, 1)
        pooled = summed / denom                     # (B, D)

        # ---------------------------------------------------------
        # (6) Classification
        # ---------------------------------------------------------
        pooled = self.norm(pooled)
        embeddings = self.head(pooled)
        embeddings = F.normalize(embeddings, dim=1)
        return embeddings

# --------------------------------------------------
# Smoke test
# --------------------------------------------------
if __name__ == "__main__":
    B = 16
    T = 12
    N = 64

    # RD test
    rd = RD_Encoder().eval()
    x_rd = torch.randn(B, T, 64, 64)
    lengths = torch.tensor([12, 10, 6, 9,12, 10, 6, 9,12, 10, 6, 9,12, 10, 6, 9])
    out_rd = rd(x_rd, lengths)
    print("RD:", out_rd.shape)

    # PC test
    pc = PC_Encoder(num_points=N).eval()
    x_pc = torch.randn(B, T, N, 3)
    out_pc = pc(x_pc, lengths)
    print("PC:", out_pc.shape)
