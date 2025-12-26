import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random

# ================================================================
# Utility Modules
# ================================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * 
                             (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)     
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)                  

    def forward(self, x):
        """ x: (B, L, D) """
        L = x.size(1)
        return x + self.pe[:L]


class TransformerBlock(nn.Module):
    """Standard Transformer Encoder Block with: 
       - Multi-head self-attention
       - FFN
       - LayerNorm
    """
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x,padding_mask = None):
        attn_out, _ = self.attn(x, x, x, key_padding_mask = padding_mask)
        x = self.ln1(x + attn_out)
        ff_out = self.ff(x)
        x = self.ln2(x + ff_out)
        return x


class CrossAttention(nn.Module):
    """Cross-attention: Q attends to K/V."""
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, Q, K, V, kv_padding_mask=None):
        out, _ = self.attn(Q, K, V,key_padding_mask = kv_padding_mask)
        return self.ln(Q + out)


# ================================================================
# Main Fusion Model
# ================================================================

class Combined_Encoder(nn.Module):
    def __init__(
        self,
        PC_Encoder,
        RD_Encoder,
        d_model=256,
        n_heads=8,
        d_ff=512,
        num_fusion_layers=4,
        max_pc_len=128,
        max_rd_len=128,
        mask_ratio=0.5,
        temperature=0.1
    ):
        super().__init__()

        # ----------------------------------------------------------
        # Freeze PC + RD Encoders
        # ----------------------------------------------------------
        self.PC_Encoder = PC_Encoder.eval()
        self.RD_Encoder = RD_Encoder.eval()

        for p in self.PC_Encoder.parameters():
            p.requires_grad = False
        for p in self.RD_Encoder.parameters():
            p.requires_grad = False

        # Save parameters
        self.d_model = d_model
        self.mask_ratio = mask_ratio
        self.temperature = temperature
        self.max_pc_len = max_pc_len
        self.max_rd_len = max_rd_len

        # ----------------------------------------------------------
        # Positional + Modality embeddings
        # ----------------------------------------------------------
        self.pos_pc = PositionalEncoding(d_model, max_pc_len)
        self.pos_rd = PositionalEncoding(d_model, max_rd_len)

        # Modality embeddings (added AFTER fusion)
        self.mod_pc = nn.Parameter(torch.randn(d_model))
        self.mod_rd = nn.Parameter(torch.randn(d_model))

        # Global CLS token
        self.cls_token = nn.Parameter(torch.randn(d_model))

        # Mask token
        self.mask_token = nn.Parameter(torch.randn(d_model))

        # ----------------------------------------------------------
        # Fusion layers: each consists of:
        #  - SelfAttn_PC
        #  - SelfAttn_RD
        #  - CrossAttn PC->RD
        #  - CrossAttn RD->PC
        # ----------------------------------------------------------
        self.self_pc = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff)
            for _ in range(num_fusion_layers)
        ])

        self.self_rd = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff)
            for _ in range(num_fusion_layers)
        ])

        self.cross_pc_rd = nn.ModuleList([
            CrossAttention(d_model, n_heads)
            for _ in range(num_fusion_layers)
        ])

        self.cross_rd_pc = nn.ModuleList([
            CrossAttention(d_model, n_heads)
            for _ in range(num_fusion_layers)
        ])

        # ----------------------------------------------------------
        # Recovery Transformer (MASTER-style)
        # Use ModuleList so each block can receive the padding mask
        # ----------------------------------------------------------
        self.recovery = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff),
            TransformerBlock(d_model, n_heads, d_ff)
        ])


    # ================================================================
    # Masking
    # ================================================================
    #padded tokens are also getting masked
    def apply_masking(self, seq):
        """
        seq: (B, L, D)
        Returns:
            masked_seq
            mask_indices (list of lists)
            original_seq (detached)
        """
        B, L, D = seq.size()
        num_to_mask = int(L * self.mask_ratio)

        seq_masked = seq.clone()
        all_mask_indices = []
        original = seq.clone().detach()

        for b in range(B):
            indices = list(range(1, L))  # do NOT mask CLS at idx 0
            random.shuffle(indices)
            mask_idx = indices[:num_to_mask]
            all_mask_indices.append(mask_idx)

            for i in mask_idx:
                r = random.random()
                if r < 0.8:
                    #replace with mask token
                    seq_masked[b, i] = self.mask_token
                elif r < 0.9:
                    # replace with random token in batch
                    rand_b = random.randint(0, B - 1)
                    rand_l = random.randint(1, L - 1)
                    seq_masked[b, i] = seq[rand_b, rand_l]
                else:
                    # leave token unchanged
                    pass

        return seq_masked, all_mask_indices, original


    # ================================================================
    # Contrastive Recovery Loss (MASTER)
    # ================================================================
    def recovery_loss(self, pred, true, mask_indices, padding_mask):
        """
        pred: (B, L, D)
        true: (B, L, D)
        mask_indices: list of lists
        """
        B, L, D = pred.size()
        valid_mask = (~padding_mask).reshape(B * L)
        # Flatten batch for negatives
        true_flat = true.reshape(B * L, D)[valid_mask]

        loss = 0
        count = 0
        tau = self.temperature

        for b in range(B):
            for idx in mask_indices[b]:
                if padding_mask[b,idx]: ##REMOVE TO AVOID CASUAL MASKING
                    continue

                t = true[b, idx]      # (D,)
                p = pred[b, idx]      # (D,)

                # Numerator
                pos = torch.exp(F.cosine_similarity(p, t, dim=0) / tau)

                # Denominator (negatives = ALL true except this one)
                # pred and true matrices are normalized before being passed so it gives cosine similarity
                neg = torch.exp(torch.matmul(true_flat, p) / tau).sum() - pos

                loss += -torch.log(pos / neg)
                count += 1
        return loss / max(count, 1)


    # ================================================================
    # Forward Pass
    # ================================================================
    def forward(self, pc_batch, pc_len, rd_batch, rd_len):
        """
        pc_batch: (B, L_pc, input_dim_pc) or None
        pc_len: (B,) lengths
        rd_batch: (B, L_rd, input_dim_rd) or None 
        rd_len: (B,)
        """

        B = pc_batch.size(0) if pc_batch is not None else rd_batch.size(0)

        # ------------------------------------------------------
        # 1. Encode PC
        # ------------------------------------------------------
        if pc_batch is not None:
            with torch.no_grad():
                E_pc = self.PC_Encoder(pc_batch, pc_len)  # (B, L_pc, D)
                pc_mask = torch.zeros(B, E_pc.size(1), dtype=torch.bool, device=E_pc.device)
                for i in range(B) :
                    for j in range(pc_len[i],E_pc.size(1)):
                        pc_mask[i][j] = 1
        else:
            E_pc = torch.zeros(B, self.max_pc_len, self.d_model,
                               device=self.cls_token.device)
            pc_mask = torch.zeros(B, self.max_pc_len, dtype=torch.bool, device=E_pc.device)

        # ------------------------------------------------------
        # 2. Encode RD
        # ------------------------------------------------------
        if rd_batch is not None:
            with torch.no_grad():
                E_rd = self.RD_Encoder(rd_batch, rd_len)  # (B, L_rd, D)
                rd_mask = torch.zeros(B, E_rd.size(1), dtype=torch.bool, device=E_rd.device)
                for i in range(B) :
                    for j in range(rd_len[i],E_rd.size(1)):
                        rd_mask[i][j] = 1
        else:
            E_rd = torch.zeros(B, self.max_rd_len, self.d_model,
                               device=self.cls_token.device)
            rd_mask = torch.zeros(B, self.max_rd_len, dtype=torch.bool, device=E_rd.device)

        # ------------------------------------------------------
        # 3. Add positional encodings
        # ------------------------------------------------------
        E_pc = self.pos_pc(E_pc)
        E_rd = self.pos_rd(E_rd)

        # ------------------------------------------------------
        # 4. Fusion Layers
        # ------------------------------------------------------

        for SA_PC, SA_RD, CA_pc_rd, CA_rd_pc in zip(
            self.self_pc, self.self_rd,
            self.cross_pc_rd, self.cross_rd_pc
        ):
            E_pc = SA_PC(E_pc, pc_mask)
            E_rd = SA_RD(E_rd, rd_mask)

            # Bidirectional cross-attention
            E_pc_cross = CA_pc_rd(E_pc, E_rd, E_rd, rd_mask)
            E_rd_cross = CA_rd_pc(E_rd, E_pc, E_pc, pc_mask)
            E_pc = E_pc_cross
            E_rd = E_rd_cross
        # ------------------------------------------------------
        # 5. Concatenate
        # ------------------------------------------------------
        fused = torch.cat([E_pc, E_rd], dim=1)  # (B, L_pc + L_rd, D)
        fused_mask = torch.cat([pc_mask,rd_mask],dim=1)
        B, L_concat, D = fused.size()

        # ------------------------------------------------------
        # 6. Add modality embeddings
        # ------------------------------------------------------
        fused[:, :E_pc.size(1)] += self.mod_pc
        fused[:, E_pc.size(1):] += self.mod_rd

        # ------------------------------------------------------
        # 7. Add CLS token
        # ------------------------------------------------------
        CLS = self.cls_token.unsqueeze(0).unsqueeze(1).repeat(B, 1, 1)
        full_seq = torch.cat([CLS, fused], dim=1)   # (B, 1 + L_concat, D)
        full_mask = torch.cat([torch.zeros(B,1, dtype=torch.bool, device=E_rd.device),fused_mask],dim=1)
        # ------------------------------------------------------
        # 8. Apply masking
        # ------------------------------------------------------
        masked_seq, mask_idx, original = self.apply_masking(full_seq)

        # ------------------------------------------------------
        # 9. Recovery Transformer (apply each block with padding mask)
        # ------------------------------------------------------
        x = masked_seq
        for blk in self.recovery:
            x = blk(x, padding_mask=full_mask)
        recovered = x

        # ------------------------------------------------------
        # 10. Recovery loss
        # ------------------------------------------------------
        loss = self.recovery_loss(F.normalize(recovered,dim=2), F.normalize(original,dim=2), mask_idx, full_mask)

        return loss, recovered, original
