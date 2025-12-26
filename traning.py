import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import BatchSampler, ConcatDataset, DataLoader
from tqdm import tqdm
from dataset_h5 import SoliHD5Dataset, TinyRadarDataset, collate_sequences_padded
from pantomime import PantomimeDataset, collate_point_clouds
from models import P4Transformer, RD_ViViT_Encoder
from moco import MoCoWrapper
# import copy
# import math

# -------------------------
# MoCo v2 helper classes
# -------------------------
# class ProjectionMLP(nn.Module):
#     """Simple 2-layer projection MLP used in MoCo v2: 128 -> 2048 -> 128"""
#     def __init__(self, in_dim=128, hidden_dim=2048, out_dim=128):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(in_dim, hidden_dim, bias=True),
#             nn.BatchNorm1d(hidden_dim),
#             nn.ReLU(inplace=True),
#             nn.Linear(hidden_dim, out_dim, bias=True),
#         )

#     def forward(self, x):
#         # x: (B, in_dim)
#         return self.net(x)

# class MoCoWrapper(nn.Module):
#     """
#     Holds:
#       - query_encoder: online encoder + projection head
#       - key_encoder: momentum encoder + projection head (no grads)
#       - queue: tensor of shape (feat_dim, K)
#     """
#     def __init__(self, base_encoder, feat_dim=128, K=8192, m=0.999, T=0.2, device='cuda'):
#         """
#         base_encoder: callable returning (B, feat_dim) already (your PC_Encoder/RD_Encoder)
#         We assume base_encoder is an nn.Module (so we can .to(device) and copy)
#         """
#         super().__init__()
#         self.m = m
#         self.K = K
#         self.T = T
#         self.device = device
#         self.feat_dim = feat_dim

#         # Query encoder and its projection head
#         self.encoder_q = base_encoder
#         self.proj_q = ProjectionMLP(in_dim=feat_dim).to(device)

#         # Key encoder (momentum copy) and its projection head
#         # We'll deep-copy the modules so params start equal
#         self.encoder_k = copy.deepcopy(base_encoder)
#         self.proj_k = copy.deepcopy(self.proj_q)

#         # Put key encoder in eval mode and disable grads
#         for p in self.encoder_k.parameters():
#             p.requires_grad = False
#         for p in self.proj_k.parameters():
#             p.requires_grad = False

#         # Create the queue (feat_dim x K)
#         self.register_buffer("queue", torch.randn(feat_dim, K).to(device))
#         self.queue = nn.functional.normalize(self.queue, dim=0)
#         self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

#     @torch.no_grad()
#     def _momentum_update_key_encoder(self):
#         """Momentum update of key encoder: param_k = m * param_k + (1 - m) * param_q"""
#         for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
#             param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)
#         for param_q, param_k in zip(self.proj_q.parameters(), self.proj_k.parameters()):
#             param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

#     @torch.no_grad()
#     def _dequeue_and_enqueue(self, keys):
#         """
#         keys: (B, feat_dim) - already normalized
#         We enqueue along axis K (columns). The queue has shape (feat_dim, K)
#         """
#         keys = keys.detach()  # (B, feat_dim)
#         B = keys.shape[0]

#         ptr = int(self.queue_ptr.item())
#         # if K not multiple of B, wrap around
#         if ptr + B <= self.K:
#             self.queue[:, ptr:ptr+B] = keys.T
#             ptr = ptr + B
#         else:
#             # split
#             end = self.K - ptr
#             self.queue[:, ptr:self.K] = keys[:end].T
#             remain = B - end
#             self.queue[:, 0:remain] = keys[end:].T
#             ptr = remain

#         self.queue_ptr[0] = ptr % self.K

#     def forward_backbone(self, online_view, online_lengths, target_view, target_lengths, device):
#         """
#         Compute query features (online_view through encoder_q+proj_q)
#                 key features   (target_view through encoder_k+proj_k)
#         Returns:
#             q: (B, feat_dim) normalized
#             k: (B, feat_dim) normalized, detached (no grads)
#         """
#         # online (query)
#         q_feat = self.encoder_q(online_view, online_lengths)  # (B, feat_dim)
#         q = self.proj_q(q_feat)  # (B, feat_dim)
#         q = nn.functional.normalize(q, dim=1)

#         # key (momentum) - no gradient
#         with torch.no_grad():
#             # update momentum encoders BEFORE using them? MoCo does momentum update each step after query encoder grads.
#             k_feat = self.encoder_k(target_view, target_lengths)
#             k = self.proj_k(k_feat)
#             k = nn.functional.normalize(k, dim=1)

#         return q, k

#     def forward(self, x, lengths):
#         """
#         For compatibility: just run the query encoder + proj head
#         """
#         feat = self.encoder_q(x, lengths)  # (B, feat_dim)
#         q = self.proj_q(feat)  # (B, feat_dim)
#         q = nn.functional.normalize(q, dim=1)
#         return q

#     def compute_moco_loss(self, q, k):
#         """
#         q: (B, D), k: (B, D)
#         queue: (D, K)
#         returns cross-entropy loss
#         """
#         # positive logits: Nx1
#         # negative logits: NxK -> from queue
#         D = self.feat_dim
#         K = self.K

#         # compute positive logits: (B,)
#         l_pos = torch.sum(q * k, dim=1, keepdim=True)  # (B,1)
#         # compute negative logits: q @ queue  -> (B, K)
#         l_neg = torch.matmul(q, self.queue.clone().detach())  # (B, K)

#         # logits: concat
#         logits = torch.cat([l_pos, l_neg], dim=1)  # (B, 1+K)
#         logits /= self.T

#         # labels: positives are index 0
#         labels = torch.zeros(logits.size(0), dtype=torch.long, device=q.device)

#         loss = F.cross_entropy(logits, labels)
#         return loss

# Create a custom batch sampler that alternates between datasets
class MixedBatchSampler(BatchSampler):
    def __init__(self, dataset1_size, dataset2_size, batch_size, drop_last=False):
        self.dataset1_size = dataset1_size
        self.dataset2_size = dataset2_size
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        indices1 = list(range(self.dataset1_size))
        indices2 = list(
            range(self.dataset1_size, self.dataset1_size + self.dataset2_size)
        )

        random.shuffle(indices1)
        random.shuffle(indices2)

        for i in range(0, min(len(indices1), len(indices2)), self.batch_size // 2):
            batch = (
                indices1[i : i + self.batch_size // 2]
                + indices2[i : i + self.batch_size // 2]
            )
            if len(batch) == self.batch_size or not self.drop_last:
                yield batch

#These are all functions for different RD augmentations
def temporal_crop(x, min_frac=0.8):
    T = x.shape[0]
    t_len = int(T * (min_frac + random.random() * (1 - min_frac)))
    if t_len < 1:
        t_len = 1
    start = random.randint(0, T - t_len)
    return x[start : start + t_len]

def frame_dropout(x, drop_prob=0.1):
    keep_mask = torch.rand(x.shape[0]) > drop_prob
    if keep_mask.sum() == 0:
        keep_mask[random.randint(0, x.shape[0] - 1)] = True
    return x[keep_mask]

def temporal_jitter(x, repeat_prob=0.1, skip_prob=0.1):
    frames = []
    for i in range(x.shape[0]):
        r = random.random()
        if r < skip_prob:
            continue  # drop frame
        frames.append(x[i])
        if r > 1 - repeat_prob:
            frames.append(x[i])  # repeat frame
    if len(frames) == 0:
        return x
    return torch.stack(frames, dim=0)

#This is the class for adifferent Point Cloud augmentation
class PointCloudAugment(nn.Module):
    """
    Augmentation module for point-cloud sequences.
    Implements:
        A1: Gaussian Jitter
        A2: Random Scaling
        A3: Temporal Jittering

    Input:
        x: (B, T, N, 3)
        lengths: (B,)
    Output:
        augmented_x: (B, T, N, 3)
        augmented_lengths: (B,)  # identical because A1/A4/A7 don't change T
    """

    def __init__(
        self,
        jitter_sigma=0.01,  # A1
        jitter_clip=0.05,
        scale_low=0.9,
        scale_high=1.1,  # A4
        temporal_jitter_sigma=0.01,  # A7
    ):
        super().__init__()
        self.jitter_sigma = jitter_sigma
        self.jitter_clip = jitter_clip
        self.scale_low = scale_low
        self.scale_high = scale_high
        self.temporal_jitter_sigma = temporal_jitter_sigma

    def forward(self, x, lengths):
        """
        x: (B, T, N, 3)
        lengths: (B,)
        """
        B, T, N, _ = x.shape
        device = x.device

        # Mask for padded frames: (B, T, 1, 1)
        frame_mask = torch.arange(T, device=device).unsqueeze(0) < lengths.unsqueeze(1)
        frame_mask = frame_mask.float().unsqueeze(-1).unsqueeze(-1)

        # --------------------------------------------------
        # A1: Gaussian Jitter
        # --------------------------------------------------
        if self.jitter_sigma > 0:
            noise = torch.clamp(
                self.jitter_sigma * torch.randn_like(x),
                -self.jitter_clip,
                self.jitter_clip,
            )
            x = x + noise * frame_mask

        # --------------------------------------------------
        # A4: Random Scaling (global per sequence)
        # --------------------------------------------------
        scales = torch.empty(B, device=device).uniform_(self.scale_low, self.scale_high)
        scales = scales.view(B, 1, 1, 1)
        x = x * scales

        # --------------------------------------------------
        # A7: Temporal Jittering (per-frame motion noise)
        # --------------------------------------------------
        if self.temporal_jitter_sigma > 0:
            t_offsets = self.temporal_jitter_sigma * torch.randn(
                B, T, 1, 3, device=device
            )
            x = x + t_offsets * frame_mask

        # A1, A4, A7 do NOT change sequence length
        new_lengths = lengths.clone()

        return x, new_lengths

#This is the augmentation function for RD heatmaps
def augmentation(sequences, lengths):
    """
    sequences: (B, T_max, H, W)
    lengths:   (B,) true lengths before padding

    Returns:
    new_sequences: (B, T_max, H, W) padded augmented sequences
    new_lengths:   (B,) updated lengths
    """

    B, T_max, H, W = sequences.shape

    new_sequences = torch.zeros_like(sequences)
    new_lengths = torch.zeros_like(lengths)

    AUGS = [temporal_crop, frame_dropout, temporal_jitter]

    for b in range(B):
        L = lengths[b].item()
        x = sequences[b, :L]  # extract valid sequence (T, H, W)

        # ---- choose 1, 2, or 3 augmentations randomly ----
        num_augs = random.randint(1, 3)
        chosen = random.sample(AUGS, num_augs)

        # ---- apply augmentations sequentially ----
        for aug in chosen:
            x = aug(x)

        # ---- store augmented (variable-length) ----
        new_L = x.shape[0]
        new_lengths[b] = min(new_L, T_max)

        # ---- pad back to (T_max, H, W) ----
        padded = torch.zeros(
            T_max, H, W, device=sequences.device, dtype=sequences.dtype
        )
        padded[:new_L] = x[:T_max]
        new_sequences[b] = padded

    return new_sequences, new_lengths

#Loss function
def info_nce_loss(query, key, temperature=0.3):
    query = F.normalize(query, dim=-1)
    key = F.normalize(key, dim=-1)

    logits = torch.matmul(query, key.T) / temperature
    labels = torch.arange(len(query)).long().to(query.device)

    loss = F.cross_entropy(logits, labels)
    return loss

# ======== Training Epoch ======== #
def train_epoch(moco_model:MoCoWrapper, train_loader, optimizer, scheduler, device, aug,accumulation_steps=1):
    moco_model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    pbar = tqdm(train_loader, desc="Training")
    batch_idx = 0 
    for sequences, lengths, labels in pbar:
        # sequences, lengths, labels = (
        #     sequences.to(device),
        #     lengths.to(device),
        #     labels.to(device),
        # )
        augmented_sequences, augmented_lengths = aug(sequences, lengths)
        sequences = sequences.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        aug_sequences = augmented_sequences.to(device, non_blocking=True)
        aug_lengths = augmented_lengths.to(device, non_blocking=True)
        # embeddings = model(sequences, lengths)
        # augmented_embeddings = model(aug_sequences, aug_lengths)
        q,k = moco_model.forward_backbone(sequences, lengths, aug_sequences, aug_lengths, device)

        #loss = info_nce_loss(augmented_embeddings, embeddings, temperature=0.3)
        loss = moco_model.compute_moco_loss(q, k)
        total_loss += loss.item()

        # -----------------------------
        # Gradient Accumulation
        # -----------------------------
        loss = loss / accumulation_steps
        loss.backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            with torch.no_grad():
                moco_model._momentum_update_key_encoder()

        with torch.no_grad():
            k_norm = nn.functional.normalize(k.float(), dim=1)
            moco_model._dequeue_and_enqueue(k_norm)

        if scheduler:
            current_lr = optimizer.param_groups[0]["lr"]
            scheduler.step()
        batch_idx += 1
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "LR": f"{current_lr:.8f}"})

    return total_loss / len(train_loader)

def evaluate(model, eval_loader, device, aug):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        pbar = tqdm(eval_loader, desc="Evaluating")
        for sequences, lengths, labels in pbar:
            # sequences = sequences.to(device)
            # lengths = lengths.to(device)
            # labels = labels.to(device)
            augmented_sequences, augmented_lengths = aug(sequences, lengths)
            sequences = sequences.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)
            aug_sequences = augmented_sequences.to(device, non_blocking=True)
            aug_lengths = augmented_lengths.to(device, non_blocking=True)

            embeddings = model(sequences, lengths)
            augmented_embeddings = model(aug_sequences, aug_lengths)

            loss = info_nce_loss(augmented_embeddings, embeddings, temperature=0.3)
            total_loss += loss.item()

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / len(eval_loader)

# ======== Main ======== #
def main_PC():
    config = {
        "batch_size": 128,
        "epochs": 100,
        "lr": 1e-4,
        "momentum": 0.9,
        "patience": 25,
        "lr_schedule": True,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    print("=" * 70)
    print("Fine-tuning Gesture Encoder (InfoNCE loss)")
    print("=" * 70)

    for k, v in config.items():
        print(f"   {k}: {v}")

    checkpoint_dir = "Encoder_Checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    print("\n📚 Loading datasets...")

    #Note: PantomimeDataset for point clouds needs separate implementation
    data_pantomime_office = PantomimeDataset('./pantomime_foundational', environment='office')
    data_pantomime_open = PantomimeDataset('./pantomime_foundational', environment='open')

    # Combine both environments
    combined_pc_dataset = ConcatDataset([data_pantomime_office, data_pantomime_open])
    train_pc_dataset, val_pc_dataset = torch.utils.data.random_split(
        combined_pc_dataset, [int(0.8 * len(combined_pc_dataset)), len(combined_pc_dataset) - int(0.8 * len(combined_pc_dataset))])

    train_pc_loader = DataLoader(
        train_pc_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=collate_point_clouds,
        num_workers=4,
        pin_memory=True
    )
    val_pc_loader = DataLoader(
        val_pc_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=collate_point_clouds,
        num_workers=4,
        pin_memory=True
    )


    print("\n🧠 Creating Point Cloud Encoder...")
    print(f"   Device: {config['device']}")


    # PC_model = PC_Encoder()
    # PC_model.to(config["device"])
    base_PC = P4Transformer()
    base_PC.to(config["device"])
    queue_size = 8192  # try 8192 or 4096; tune for memory/speed
    moco_m = 0.999
    temperature = 0.1
    moco_PC = MoCoWrapper(base_encoder=base_PC, feat_dim=512, K=queue_size, m=moco_m, T=temperature, device=config["device"])
    moco_PC = moco_PC.to(config["device"])
    #ckpt = torch.load('Encoder_Checkpoints/best_PC_encoder.pth', map_location=config["device"])
    #PC_model.load_state_dict(ckpt['model_state_dict'], strict=True)
    print(f"   Model: {moco_PC.__class__.__name__}")
    print(f"   Parameters: {sum(p.numel() for p in moco_PC.parameters()):,}")
    ckpt = torch.load('Encoder_Checkpoints/best_P4_PC_encoder_with_moco.pth', map_location=config["device"])
    moco_PC.load_state_dict(ckpt['model_state_dict'], strict=True)
    moco_PC.proj_k.load_state_dict(moco_PC.proj_q.state_dict())
    # PC_optimizer = optim.Adam(moco_PC.parameters(), lr=config["lr"])
    # PC_scheduler = (
    #     optim.lr_scheduler.StepLR(PC_optimizer, step_size=500, gamma=0.9)
    #     if config["lr_schedule"]
    #     else None
    # )
    PC_optimizer = optim.Adam(list(moco_PC.encoder_q.parameters()) + list(moco_PC.proj_q.parameters()), lr=config["lr"])
    PC_scheduler = optim.lr_scheduler.StepLR(PC_optimizer, step_size=500, gamma=0.9) if config["lr_schedule"] else None
    print("\n🚀 Starting Fine-tuning Point Cloud Encoder...")
    best_loss_PC = float("inf")
    PC_patience_counter = 0
    PC_aug = PointCloudAugment(
        jitter_sigma=0.01,
        jitter_clip=0.05,
        scale_low=0.9,
        scale_high=1.1,
        temporal_jitter_sigma=0.01,
    )

    for epoch in range(1, config["epochs"] + 1):
        torch.cuda.empty_cache()
        print(f"\nEpoch {epoch}/{config['epochs']}")

        train_loss_PC = train_epoch(
            moco_PC, train_pc_loader, PC_optimizer, PC_scheduler, config["device"], PC_aug,accumulation_steps=1)
        eval_loss_PC = evaluate(
            moco_PC, val_pc_loader, config["device"], PC_aug
        )
        print(f"Train Loss: PC = {train_loss_PC:.5f}")
        print(f"Eval Loss: PC = {eval_loss_PC:.5f}")

        if PC_scheduler:
            current_lr = PC_optimizer.param_groups[0]["lr"]
            print(f"  PC LR: {current_lr:.6f}")
            PC_scheduler.step()

        if eval_loss_PC < best_loss_PC:
            best_loss_PC = eval_loss_PC
            PC_patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": moco_PC.state_dict(),
                    "optimizer_state_dict": PC_optimizer.state_dict(),
                    "eval_loss": eval_loss_PC,
                },
                os.path.join(checkpoint_dir, "best_P4_PC_encoder_with_moco.pth"),
            )
            print(f"✅ Saved best model (loss: {eval_loss_PC:.5f})")
        else:
            PC_patience_counter += 1

        if PC_patience_counter >= config["patience"]:
            print(f"\n⚠️ Early stopping triggered after {epoch} epochs")
            break

    print("\n" + "=" * 70)
    print(f"Training Complete! Best Eval Loss: PC = {best_loss_PC:.5f}")
    print("=" * 70)
    return  best_loss_PC

def main_RD():
    config = {
        "batch_size": 8,
        "epochs": 100,
        "lr": 1e-7,
        "momentum": 0.9,
        "patience": 25,
        "lr_schedule": True,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    print("=" * 70)
    print("Fine-tuning Gesture Encoder (InfoNCE loss)")
    print("=" * 70)

    for k, v in config.items():
        print(f"   {k}: {v}")

    checkpoint_dir = "Encoder_Checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    print("\n📚 Loading datasets...")
    # Load preprocessed TinyRadar datasets (5G and 11G)
    # These return ((T, H, W), label) where T=40, H=W=64
    data_tiny_5g = TinyRadarDataset("./processed/5G")  # (40, 64, 64)
    data_tiny_11g = TinyRadarDataset("./processed/11G")  # (40, 64, 64)

    # Load Soli dataset from original HDF5 files
    data_soli = SoliHD5Dataset("./SOLI")  # (T, 64, 64)

    # # Combine all RD datasets
    combined_rd_dataset = ConcatDataset([data_soli, data_tiny_5g, data_tiny_11g])
    train_rd_dataset, val_rd_dataset = torch.utils.data.random_split(
        combined_rd_dataset, [int(0.8 * len(combined_rd_dataset)), len(combined_rd_dataset) - int(0.8 * len(combined_rd_dataset))])
    train_rd_loader = DataLoader(
        train_rd_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=collate_sequences_padded,
        pin_memory=True
    )
    val_rd_loader = DataLoader(
        val_rd_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=collate_sequences_padded,
        pin_memory=True
    )

    print("\n🧠 Creating Range Doppler Encoder...")
    print(f"   Device: {config['device']}")

    # RD_model = RD_Encoder()
    # RD_model.to(config["device"])
    # print(f"   Model: {RD_model.__class__.__name__}")
    # print(f"   Parameters: {sum(p.numel() for p in RD_model.parameters()):,}")
    # RD_optimizer = optim.Adam(RD_model.parameters(), lr=config["lr"])
    # RD_scheduler = (
    #     optim.lr_scheduler.StepLR(RD_optimizer, step_size=100, gamma=0.75,)
    #     if config["lr_schedule"]
    #     else None
    # )

    base_RD = RD_ViViT_Encoder()
    base_RD.to(config["device"])
    queue_size = 8192  # try 8192 or 4096; tune for memory/speed
    moco_m = 0.99
    temperature = 0.05
    moco_RD = MoCoWrapper(base_encoder=base_RD, feat_dim=512, K=queue_size, m=moco_m, T=temperature, device=config["device"])
    moco_RD = moco_RD.to(config["device"])
    #ckpt = torch.load('Encoder_Checkpoints/best_RD_encoder.pth', map_location=config["device"])
    #RD_model.load_state_dict(ckpt['model_state_dict'], strict=True)
    print(f"   Model: {moco_RD.__class__.__name__}")
    print(f"   Parameters: {sum(p.numel() for p in moco_RD.parameters()):,}")
    # ckpt = torch.load('Encoder_Checkpoints/best_ViViT_RD_encoder.pth', map_location=config["device"])
    # moco_RD.encoder_q.load_state_dict(ckpt['model_state_dict'], strict=True)
    moco_RD.proj_k.load_state_dict(moco_RD.proj_q.state_dict())
    RD_optimizer = optim.Adam(list(moco_RD.encoder_q.parameters()) + list(moco_RD.proj_q.parameters()), lr=config["lr"])
    RD_scheduler = optim.lr_scheduler.StepLR(RD_optimizer, step_size=100, gamma=0.9) if config["lr_schedule"] else None
    print("\n🚀 Starting Fine-tuning Range Doppler Encoder...")

    best_loss_RD = float("inf")
    RD_patience_counter = 0
    for epoch in range(1, config["epochs"] + 1):
        torch.cuda.empty_cache()
        print(f"\nEpoch {epoch}/{config['epochs']}")

        train_loss_RD = train_epoch(
            moco_RD,
            train_rd_loader,
            RD_optimizer,
            RD_scheduler,
            config["device"],
            augmentation,
            accumulation_steps=4
        )
        eval_loss_RD = evaluate(
            moco_RD, val_rd_loader, config["device"], augmentation
        )
        print(f"Train Loss: RD = {train_loss_RD:.4f}")
        print(f"Eval Loss: RD = {eval_loss_RD:.4f}")

        if eval_loss_RD < best_loss_RD:
            best_loss_RD = eval_loss_RD
            RD_patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": moco_RD.state_dict(),
                    "optimizer_state_dict": RD_optimizer.state_dict(),
                    "eval_loss": eval_loss_RD,
                },
                os.path.join(checkpoint_dir, "best_ViViT_RD_encoder_with_moco.pth"),
            )
            print(f"✅ Saved best model (loss: {eval_loss_RD:.2f})")
        else:
            RD_patience_counter += 1

        if RD_patience_counter >= config["patience"]:
            print(f"\n⚠️ Early stopping triggered after {epoch} epochs")
            break

    print("\n" + "=" * 70)
    print(f"Training Complete! Best Eval Loss: RD = {best_loss_RD:.5f}")
    print("=" * 70)
    return  best_loss_RD

if __name__ == "__main__":
    try:
        # best_loss_RD = main_RD()
        best_loss_PC = main_RD()
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback

        traceback.print_exc()
        raise
