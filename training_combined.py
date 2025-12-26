# import os
# import torch
# import torch.optim as optim
# from torch.utils.data import DataLoader, ConcatDataset
# from tqdm import tqdm
# from dataset_h5 import SoliHD5Dataset, TinyRadarDataset, collate_sequences_padded
# from pantomime import PantomimeDataset, collate_point_clouds
# from model_combined import combined_Encoder
# from models import PC_Encoder, RD_Encoder

# # ============================================================
# # Utility Functions (Unimodal)
# # ============================================================
# def train_epoch_unimodal(model, loader, optimizer, device, modality="PC"):
#     """
#     Train on a SINGLE modality only.
#     """
#     model.train()
#     total_loss = 0

#     pbar = tqdm(loader, desc=f"Train ({modality})")

#     for batch in pbar:
#         loss = train_one_unimodal_step(model, batch, optimizer, device, modality)
#         total_loss += loss
#         pbar.set_postfix({"loss": f"{loss:.5f}"})

#     return total_loss / len(loader)

# def train_one_unimodal_step(model, batch, optimizer, device, modality):
#     if modality == "PC":
#         pc_batch, pc_len, _ = batch
#         pc_batch = pc_batch.to(device)
#         pc_len = pc_len.to(device)
#         rd_batch, rd_len = None, None

#     else:  # RD
#         rd_batch, rd_len, _ = batch
#         rd_batch = rd_batch.to(device)
#         rd_len = rd_len.to(device)
#         pc_batch, pc_len = None, None

#     loss, _, _, _ = model(pc_batch, pc_len, rd_batch, rd_len)

#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

#     return loss.item()

# # ============================================================
# # Stage 1A — Train model using ONLY PC batches
# # ============================================================
# def stage1_pc_only(model, train_pc_loader, optimizer, scheduler,device, epochs=10):
#     print("\n========== Stage 1A: PC-Only Warmup ==========\n")

#     for epoch in range(1, epochs + 1):
#         print(f"Stage 1A - Epoch {epoch}/{epochs}")

#         loss = train_epoch_unimodal(model, train_pc_loader, optimizer, scheduler, device, modality="PC")
#         print(f"Stage 1A Train Loss (PC): {loss:.5f}")

# # ============================================================
# # Stage 1B — Train model using ONLY RD batches
# # ============================================================
# def stage1_rd_only(model, train_rd_loader, optimizer, scheduler, device, epochs=10):
#     print("\n========== Stage 1B: RD-Only Warmup ==========\n")

#     for epoch in range(1, epochs + 1):
#         print(f"Stage 1B - Epoch {epoch}/{epochs}")

#         loss = train_epoch_unimodal(model, train_rd_loader, optimizer, scheduler, device, modality="RD")
#         print(f"Stage 1B Train Loss (RD): {loss:.5f}")

# # ============================================================
# # Stage 2 — Alternating PC & RD batches per iteration
# # ============================================================
# def stage2_alternating(model, train_pc_loader, train_rd_loader, optimizer, device, epochs=40):
#     print("\n========== Stage 2: Alternating PC/RD ==========\n")

#     pc_iter = None
#     rd_iter = None

#     for epoch in range(1, epochs + 1):
#         print(f"\nStage 2 - Epoch {epoch}/{epochs}")

#         # fresh iterators every epoch
#         pc_iter = iter(train_pc_loader)
#         rd_iter = iter(train_rd_loader)

#         total_loss = 0
#         steps = min(len(train_pc_loader), len(train_rd_loader)) * 2  # PC+RD pairs

#         pbar = tqdm(range(steps), desc="Stage 2 Training")

#         for step in pbar:
#             if step % 2 == 0:  # even step → PC
#                 try:
#                     batch = next(pc_iter)
#                     loss = train_one_unimodal_step(model, batch, optimizer, device, modality="PC")
#                 except StopIteration:
#                     continue
#             else:  # odd step → RD
#                 try:
#                     batch = next(rd_iter)
#                     loss = train_one_unimodal_step(model, batch, optimizer, device, modality="RD")
#                 except StopIteration:
#                     continue

#             total_loss += loss
#             pbar.set_postfix({"loss": f"{loss:.5f}"})

#         avg_loss = total_loss / steps
#         print(f"Stage 2 Avg Loss: {avg_loss:.5f}")

# # ============================================================
# # Stage 3 — Mixed PC and RD batches inside each training step
# # ============================================================
# # ============================================================
# # Stage 3 — Mixed PC and RD batches inside ONE forward pass
# # ============================================================

# def stage3_mixed(model, train_pc_loader, train_rd_loader, optimizer, device, epochs=20):
#     print("\n========== Stage 3: Mixed PC+RD Batches (Unified Batch) ==========\n")

#     for epoch in range(1, epochs + 1):
#         print(f"\nStage 3 - Epoch {epoch}/{epochs}")

#         pc_iter = iter(train_pc_loader)
#         rd_iter = iter(train_rd_loader)

#         pbar = tqdm(range(min(len(train_pc_loader), len(train_rd_loader))),
#                     desc="Stage 3 Training")

#         total_loss = 0

#         for _ in pbar:
#             try:
#                 pc_batch, pc_len, _ = next(pc_iter)
#                 rd_batch, rd_len, _ = next(rd_iter)
#             except StopIteration:
#                 break

#             # Move original batches to device ------------------
#             pc_batch = pc_batch.to(device)
#             pc_len   = pc_len.to(device)
#             rd_batch = rd_batch.to(device)
#             rd_len   = rd_len.to(device)

#             B_pc = pc_batch.size(0)
#             B_rd = rd_batch.size(0)
#             B_combined = B_pc + B_rd

#             # ---------------------------------------------------
#             # Construct combined batch
#             # ---------------------------------------------------

#             # For PC samples: PC=real, RD=zeros
#             rd_placeholder = torch.zeros(
#                 B_pc, rd_batch.size(1), rd_batch.size(2), rd_batch.size(3),
#                 device=device
#             )
#             rd_len_placeholder = torch.zeros(B_pc, device=device, dtype=rd_len.dtype)

#             # For RD samples: RD=real, PC=zeros
#             pc_placeholder = torch.zeros(
#                 B_rd, pc_batch.size(1), pc_batch.size(2), pc_batch.size(3),
#                 device=device
#             )
#             pc_len_placeholder = torch.zeros(B_rd, device=device, dtype=pc_len.dtype)

#             # Combine PC and RD parts -----------------------------
#             combined_pc   = torch.cat([pc_batch, pc_placeholder], dim=0)   # (B_pc + B_rd, ...)
#             combined_pc_l = torch.cat([pc_len,   pc_len_placeholder], dim=0)

#             combined_rd   = torch.cat([rd_placeholder, rd_batch], dim=0)
#             combined_rd_l = torch.cat([rd_len_placeholder, rd_len], dim=0)

#             # ---------------------------------------------------
#             # Forward pass with TRUE multimodal batch
#             # ---------------------------------------------------
#             loss, _, _, _ = model(
#                 combined_pc, combined_pc_l,
#                 combined_rd, combined_rd_l
#             )

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             total_loss += loss.item()
#             pbar.set_postfix({"loss": f"{loss.item():.5f}"})

#         avg_loss = total_loss / max(1, len(pbar))
#         print(f"Stage 3 Avg Loss: {avg_loss:.5f}")

# def train_all_stages(model, train_pc_loader, train_rd_loader, device):
#     # -------- Stage 1 (Warm-Up) -------- #
#     optimizer = optim.Adam(model.parameters(), lr=1e-4)
#     scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.8)
#     stage1_pc_only(model, train_pc_loader, optimizer,scheduler, device, epochs=10)
#     optimizer = optim.Adam(model.parameters(), lr=1e-4)
#     scheduler = optim.lr_scheduler.StepLR(optimizer, scheduler, step_size=200, gamma=0.8)
#     stage1_rd_only(model, train_rd_loader, optimizer, scheduler, device, epochs=10)

#     # -------- Stage 2 (Alternating) -------- #
#     optimizer = optim.Adam(model.parameters(), lr=1e-4)
#     scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.8)
#     stage2_alternating(model, train_pc_loader, train_rd_loader, optimizer, scheduler ,device, epochs=40)

#     # -------- Stage 3 (Mixed Exposure) -------- #
#     optimizer = optim.Adam(model.parameters(), lr=1e-4)
#     scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.8)
#     stage3_mixed(model, train_pc_loader, train_rd_loader, optimizer, scheduler, device, epochs=20)

#     print("\nTraining finished across all 3 stages.")
# # ============================================================
# # Combined Dataset Loader for PC + RD
# # ============================================================
# def load_pc_dataset(batch_size=16):
#     data_pantomime_office = PantomimeDataset('./pantomime_foundational', environment='office')
#     data_pantomime_open   = PantomimeDataset('./pantomime_foundational', environment='open')

#     combined_pc = ConcatDataset([data_pantomime_office, data_pantomime_open])

#     train_pc, val_pc = torch.utils.data.random_split(
#         combined_pc,
#         [int(0.8 * len(combined_pc)), len(combined_pc) - int(0.8 * len(combined_pc))]
#     )

#     train_pc_loader = DataLoader(
#         train_pc,
#         batch_size=batch_size,
#         shuffle=True,
#         collate_fn=collate_point_clouds,
#         num_workers=4,
#         pin_memory=True
#     )
#     val_pc_loader = DataLoader(
#         val_pc,
#         batch_size=batch_size,
#         shuffle=False,
#         collate_fn=collate_point_clouds,
#         num_workers=4,
#         pin_memory=True
#     )
#     return train_pc_loader, val_pc_loader


# def load_rd_dataset(batch_size=8):
#     data_tiny5  = TinyRadarDataset("./processed/5G")
#     data_tiny11 = TinyRadarDataset("./processed/11G")
#     data_soli   = SoliHD5Dataset("./SOLI")

#     combined_rd = ConcatDataset([data_soli, data_tiny5, data_tiny11])

#     train_rd, val_rd = torch.utils.data.random_split(
#         combined_rd,
#         [int(0.8 * len(combined_rd)), len(combined_rd) - int(0.8 * len(combined_rd))]
#     )

#     train_rd_loader = DataLoader(
#         train_rd,
#         batch_size=batch_size,
#         shuffle=True,
#         collate_fn=collate_sequences_padded,
#         num_workers=4,
#         pin_memory=True
#     )
#     val_rd_loader = DataLoader(
#         val_rd,
#         batch_size=batch_size,
#         shuffle=False,
#         collate_fn=collate_sequences_padded,
#         num_workers=4,
#         pin_memory=True
#     )
#     return train_rd_loader, val_rd_loader

# # ============================================================
# # Training Function for Combined Encoder (MASTER Pretraining)
# # ============================================================
# # def train_epoch(model, pc_loader, rd_loader, optimizer, scheduler, device, accumulation_steps=1):
# #     """
# #     We iterate RD loader and PC loader in parallel.
# #     If one loader runs out first → fallback begins automatically inside the model.
# #     """

# #     model.train()
# #     total_loss = 0.0

# #     rd_iter = iter(rd_loader)
# #     pc_iter = iter(pc_loader)

# #     pbar = tqdm(range(min(len(pc_loader), len(rd_loader))), desc="Train Combined")

# #     for _ in pbar:
# #         try:
# #             pc_batch, pc_len, _ = next(pc_iter)
# #         except StopIteration:
# #             pc_batch = None
# #             pc_len = None

# #         try:
# #             rd_batch, rd_len, _ = next(rd_iter)
# #         except StopIteration:
# #             rd_batch = None
# #             rd_len = None

# #         # Move to device
# #         if pc_batch is not None:
# #             pc_batch = pc_batch.to(device)
# #             pc_len   = pc_len.to(device)
# #         if rd_batch is not None:
# #             rd_batch = rd_batch.to(device)
# #             rd_len   = rd_len.to(device)

# #         # Forward pass (MASTER loss inside)
# #         loss, _, _, _ = model(pc_batch, pc_len, rd_batch, rd_len)

# #         optimizer.zero_grad()
# #         loss.backward()
# #         optimizer.step()

# #         total_loss += loss.item()
# #         pbar.set_postfix({"loss": f"{loss.item():.5f}"})

# #     return total_loss / max(1, len(pbar))


# # ============================================================
# # Evaluation (Validation)
# # ============================================================
# # def evaluate(model, pc_loader, rd_loader, device):
# #     model.eval()
# #     total_loss = 0.0

# #     rd_iter = iter(rd_loader)
# #     pc_iter = iter(pc_loader)

# #     with torch.no_grad():
# #         pbar = tqdm(range(min(len(pc_loader), len(rd_loader))), desc="Eval Combined")

# #         for _ in pbar:
# #             try:
# #                 pc_batch, pc_len, _ = next(pc_iter)
# #             except StopIteration:
# #                 pc_batch = None
# #                 pc_len = None

# #             try:
# #                 rd_batch, rd_len, _ = next(rd_iter)
# #             except StopIteration:
# #                 rd_batch = None
# #                 rd_len = None

# #             if pc_batch is not None:
# #                 pc_batch = pc_batch.to(device)
# #                 pc_len   = pc_len.to(device)
# #             if rd_batch is not None:
# #                 rd_batch = rd_batch.to(device)
# #                 rd_len   = rd_len.to(device)

# #             loss, _, _, _ = model(pc_batch, pc_len, rd_batch, rd_len)
# #             total_loss += loss.item()

# #             pbar.set_postfix({"loss": f"{loss.item():.5f}"})

# #     return total_loss / max(1, len(pbar))


# # ============================================================
# # Main Training Loop
# # ============================================================
# def main_combined():
#     config = {
#         "epochs": 50,
#         "batch_size_pc": 16,
#         "batch_size_rd": 8,
#         "accumulation_steps": 1,
#         "lr": 1e-4,
#         "device": "cuda" if torch.cuda.is_available() else "cpu",
#     }

#     print("\n=== Combined Encoder Pretraining (MASTER-style Masked Modeling) ===\n")
#     for k,v in config.items():
#         print(f"{k}: {v}")
#     print()

#     # ---------------------------------------------
#     # Load datasets
#     # ---------------------------------------------
#     train_pc, val_pc = load_pc_dataset(batch_size=config["batch_size_pc"])
#     train_rd, val_rd = load_rd_dataset(batch_size=config["batch_size_rd"])

#     # ---------------------------------------------
#     # Initialize Encoders
#     # ---------------------------------------------
#     device = config["device"]

#     print("\nLoading PC & RD Encoders (Frozen)")
#     pc_encoder = PC_Encoder().to(device)
#     rd_encoder = RD_Encoder().to(device)

#     # Load checkpoints if needed:
#     pc_state = torch.load("Encoder_Checkpoints/best_PC_encoder_with_moco.pth", map_location=device)["model_state_dict"]
#     rd_state = torch.load("Encoder_Checkpoints/best_RD_encoder_with_moco.pth", map_location=device)["model_state_dict"]
    
#     pc_clean = {}
#     for k, v in pc_state.items():
#         if k.startswith("encoder_q."):
#             pc_clean[k.replace("encoder_q.", "")] = v

#     rd_clean = {}
#     for k, v in rd_state.items():
#         if k.startswith("encoder_q."):
#             rd_clean[k.replace("encoder_q.", "")] = v
    
#     pc_encoder.load_state_dict(pc_clean,strict=True)
#     rd_encoder.load_state_dict(rd_clean,strict=True)

#     # ---------------------------------------------
#     # Combined Model
#     # ---------------------------------------------
#     model = combined_Encoder(
#         PC_Encoder=pc_encoder,
#         RD_Encoder=rd_encoder,
#         d_model=256,
#         n_heads=8,
#         d_ff=512,
#         num_fusion_layers=4,
#         max_pc_len=128,
#         max_rd_len=128,
#         mask_ratio=0.5,
#         temperature=0.1
#     ).to(device)

#     print(f"Total Parameters: {sum(p.numel() for p in model.parameters()):,}\n")
#     print(f"\nTotal Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
#     optimizer = optim.Adam(model.parameters(), lr=config["lr"])
#     scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.8)

#     best_loss = float("inf")
#     save_dir = "Encoder_Checkpoints"
#     os.makedirs(save_dir, exist_ok=True)

#     # ---------------------------------------------
#     # Training Loop
#     # ---------------------------------------------
#     train_all_stages(model, train_pc, train_rd, device)
#     # for epoch in range(1, config["epochs"] + 1):
#     #     print(f"\nEpoch {epoch}/{config['epochs']}")

#     #     train_loss = train_epoch(model, train_pc, train_rd, optimizer, scheduler ,device, accumulation_steps=config["accumulation_steps"])
#     #     val_loss   = evaluate(model, val_pc, val_rd, device)

#     #     print(f"  Train Loss = {train_loss:.5f}")
#     #     print(f"  Eval Loss  = {val_loss:.5f}")

#     #     #scheduler.step()

#     #     # Save best
#     #     if val_loss < best_loss:
#     #         best_loss = val_loss
#     #         ckpt_path = os.path.join(save_dir, "best_combined_encoder.pth")
#     #         torch.save(
#     #             {
#     #                 "epoch": epoch,
#     #                 "model_state_dict": model.state_dict(),
#     #                 "optimizer_state_dict": optimizer.state_dict(),
#     #                 "val_loss": best_loss,
#     #             },
#     #             ckpt_path
#     #         )
#     #         print(f"  ✅ Saved Best Model @ {ckpt_path}")

#     print("\n=== Training Complete ===")
#     print(f"Best Validation Loss: {best_loss:.6f}\n")
#     return best_loss


# # ============================================================
# # Entry point
# # ============================================================
# if __name__ == "__main__":
#     best_loss = main_combined()


import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
from dataset_h5 import SoliHD5Dataset, TinyRadarDataset, collate_sequences_padded
from pantomime import PantomimeDataset, collate_point_clouds
from model_combined import Combined_Encoder
from models import PC_Encoder, RD_Encoder


# ============================================================
# Utility (Unimodal Step)
# ============================================================

def train_one_unimodal_step(model, batch, optimizer, scheduler, device, modality):
    if modality == "PC":
        pc_batch, pc_len, _ = batch
        rd_batch, rd_len = None, None

        pc_batch = pc_batch.to(device)
        pc_len = pc_len.to(device)

    else:  # RD
        rd_batch, rd_len, _ = batch
        pc_batch, pc_len = None, None

        rd_batch = rd_batch.to(device)
        rd_len = rd_len.to(device)

    loss, _, _ = model(pc_batch, pc_len, rd_batch, rd_len)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    return loss.item()


def train_epoch_unimodal(model, loader, optimizer, scheduler, device, modality):
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc=f"Train ({modality})")
    for batch in pbar:
        loss = train_one_unimodal_step(model, batch, optimizer, scheduler, device, modality)
        total_loss += loss
        pbar.set_postfix({"loss": f"{loss:.5f}","LR": f"{scheduler.get_last_lr()[0]:.6f}"})
    return total_loss / len(loader)


# ============================================================
# STAGE 1A — PC-only warmup
# ============================================================

def stage1_pc_only(model, train_pc_loader, optimizer, scheduler, device,
                   save_dir, epochs=10):

    print("\n========== Stage 1A: PC-Only Warmup ==========\n")
    best_loss = float("inf")

    for epoch in range(1, epochs+1):
        print(f"\nStage 1A — Epoch {epoch}/{epochs}")

        train_loss = train_epoch_unimodal(model, train_pc_loader,
                                          optimizer, scheduler, device, modality="PC")
        print(f"Stage 1A Train Loss: {train_loss:.5f}")

        # checkpoint
        if train_loss < best_loss:
            best_loss = train_loss
            save_path = os.path.join(save_dir, "stage1A_best_PC.pth")
            torch.save({"model": model.state_dict(),
                        "loss": best_loss}, save_path)
            print(f"  Saved checkpoint: {save_path}")

# ============================================================
# STAGE 1B — RD-only warmup
# ============================================================

def stage1_rd_only(model, train_rd_loader, optimizer, scheduler, device,
                   save_dir, epochs=10):

    print("\n========== Stage 1B: RD-Only Warmup ==========\n")
    best_loss = float("inf")

    for epoch in range(1, epochs+1):
        print(f"\nStage 1B — Epoch {epoch}/{epochs}")

        train_loss = train_epoch_unimodal(model, train_rd_loader,
                                          optimizer, scheduler, device, modality="RD")
        print(f"Stage 1B Train Loss: {train_loss:.5f}")

        # checkpoint
        if train_loss < best_loss:
            best_loss = train_loss
            save_path = os.path.join(save_dir, "stage1B_best_RD.pth")
            torch.save({"model": model.state_dict(),
                        "loss": best_loss}, save_path)
            print(f"  Saved checkpoint: {save_path}")

# ============================================================
# STAGE 2 — alternating PC/RD
# ============================================================

def stage2_alternating(model, train_pc_loader, train_rd_loader,
                       optimizer, scheduler, device, save_dir, epochs=40):

    print("\n========== Stage 2: Alternating PC/RD ==========\n")
    best_loss = float("inf")

    for epoch in range(1, epochs+1):
        print(f"\nStage 2 — Epoch {epoch}/{epochs}")

        pc_iter = iter(train_pc_loader)
        rd_iter = iter(train_rd_loader)

        steps = min(len(train_pc_loader), len(train_rd_loader)) * 2
        total_loss = 0

        pbar = tqdm(range(steps), desc="Stage 2 Training")

        for step in pbar:
            if step % 2 == 0:   # PC step
                try:
                    batch = next(pc_iter)
                    loss = train_one_unimodal_step(model, batch,
                                                   optimizer, scheduler, device, "PC")
                except StopIteration:
                    continue
            else:               # RD step
                try:
                    batch = next(rd_iter)
                    loss = train_one_unimodal_step(model, batch,
                                                   optimizer, scheduler, device, "RD")
                except StopIteration:
                    continue

            total_loss += loss
            pbar.set_postfix({"loss": f"{loss:.5f}","LR": f"{scheduler.get_last_lr()[0]:.6f}"})

        avg_loss = total_loss / steps

        print(f"Stage 2 Avg Loss: {avg_loss:.5f}")

        # checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = os.path.join(save_dir, "stage2_best.pth")
            torch.save({"model": model.state_dict(),
                        "loss": best_loss}, save_path)
            print(f"  Saved checkpoint: {save_path}")


# ============================================================
# STAGE 3 — Mixed PC+RD unified batch exposure
# ============================================================

def stage3_mixed(model, train_pc_loader, train_rd_loader,
                 optimizer, scheduler, device, save_dir, epochs=20):

    print("\n========== Stage 3: Mixed PC+RD Unified Batch ==========\n")
    best_loss = float("inf")

    for epoch in range(1, epochs+1):
        print(f"\nStage 3 — Epoch {epoch}/{epochs}")

        pc_iter = iter(train_pc_loader)
        rd_iter = iter(train_rd_loader)

        total_loss = 0
        pbar = tqdm(range(min(len(train_pc_loader), len(train_rd_loader))),
                    desc="Stage 3 Training")

        for _ in pbar:
            try:
                pc_batch, pc_len, _ = next(pc_iter)
                rd_batch, rd_len, _ = next(rd_iter)
            except StopIteration:
                break

            pc_batch = pc_batch.to(device)
            pc_len   = pc_len.to(device)

            rd_batch = rd_batch.to(device)
            rd_len   = rd_len.to(device)

            # ---------------- unified batch ----------------
            B_pc = pc_batch.size(0)
            B_rd = rd_batch.size(0)

            # RD placeholder for PC samples
            rd_placeholder = torch.zeros(
                B_pc, rd_batch.size(1), rd_batch.size(2), rd_batch.size(3),
                device=device
            )
            rd_len_placeholder = torch.zeros(B_pc, dtype=rd_len.dtype, device=device)

            # PC placeholder for RD samples
            pc_placeholder = torch.zeros(
                B_rd, pc_batch.size(1), pc_batch.size(2), pc_batch.size(3),
                device=device
            )
            pc_len_placeholder = torch.zeros(B_rd, dtype=pc_len.dtype, device=device)

            combined_pc   = torch.cat([pc_batch, pc_placeholder], dim=0)
            combined_pc_l = torch.cat([pc_len,   pc_len_placeholder], dim=0)

            combined_rd   = torch.cat([rd_placeholder, rd_batch], dim=0)
            combined_rd_l = torch.cat([rd_len_placeholder, rd_len], dim=0)

            loss, _, _ = model(
                combined_pc, combined_pc_l,
                combined_rd, combined_rd_l
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.5f}","LR": f"{scheduler.get_last_lr()[0]:.6f}"})

        avg_loss = total_loss / max(1, len(pbar))
        print(f"Stage 3 Avg Loss: {avg_loss:.5f}")

        # checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = os.path.join(save_dir, "stage3_best.pth")
            torch.save({"model": model.state_dict(),
                        "loss": best_loss}, save_path)
            print(f"  Saved checkpoint: {save_path}")


# ============================================================
# DATA LOADING
# ============================================================

def load_pc_dataset(batch_size=16):
    data1 = PantomimeDataset('./pantomime_foundational', environment='office')
    data2 = PantomimeDataset('./pantomime_foundational', environment='open')
    combined = ConcatDataset([data1, data2])

    train, val = torch.utils.data.random_split(
        combined,
        [int(0.8*len(combined)), len(combined) - int(0.8*len(combined))]
    )

    train_loader = DataLoader(train, batch_size=batch_size,
                              shuffle=True, collate_fn=collate_point_clouds,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size,
                            shuffle=False, collate_fn=collate_point_clouds,
                            num_workers=4, pin_memory=True)
    return train_loader, val_loader


def load_rd_dataset(batch_size=8):
    data = ConcatDataset([
        SoliHD5Dataset("./SOLI"),
        TinyRadarDataset("./processed/5G"),
        TinyRadarDataset("./processed/11G"),
    ])

    train, val = torch.utils.data.random_split(
        data,
        [int(0.8*len(data)), len(data) - int(0.8*len(data))]
    )

    train_loader = DataLoader(train, batch_size=batch_size,
                              shuffle=True, collate_fn=collate_sequences_padded,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size,
                            shuffle=False, collate_fn=collate_sequences_padded,
                            num_workers=4, pin_memory=True)
    return train_loader, val_loader


# ============================================================
# MAIN
# ============================================================

def main_combined():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # datasets
    train_pc, val_pc = load_pc_dataset(batch_size=16)
    train_rd, val_rd = load_rd_dataset(batch_size=8)

    # encoders
    pc_encoder = PC_Encoder().to(device)
    rd_encoder = RD_Encoder().to(device)

    pc_ckpt = torch.load("Encoder_Checkpoints/best_PC_encoder_with_moco.pth", map_location=device)["model_state_dict"]
    rd_ckpt = torch.load("Encoder_Checkpoints/best_RD_encoder_with_moco.pth", map_location=device)["model_state_dict"]

    pc_clean = {k.replace("encoder_q.", ""): v
                for k,v in pc_ckpt.items()
                if k.startswith("encoder_q.")}

    rd_clean = {k.replace("encoder_q.", ""): v
                for k,v in rd_ckpt.items()
                if k.startswith("encoder_q.")}

    pc_encoder.load_state_dict(pc_clean, strict=True)
    rd_encoder.load_state_dict(rd_clean, strict=True)

    # combined model
    model = Combined_Encoder(
        PC_Encoder=pc_encoder,
        RD_Encoder=rd_encoder,
        d_model=128,
        n_heads=4,
        d_ff=512,
        num_fusion_layers=4,
        max_pc_len=128,
        max_rd_len=256,
        mask_ratio=0.5,
        temperature=0.1
    ).to(device)
    #model.load_state_dict(torch.load('Combined_Checkpoints/stage3_best.pth')['model'], strict=True)
    print(f"Total Parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    print(f"\nTotal Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    save_dir = "Combined_Checkpoints_improved"
    os.makedirs(save_dir, exist_ok=True)

    # ------------ Stage 1A PC ------------
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=400, gamma=0.8)
    stage1_pc_only(model, train_pc, optimizer, scheduler, device, save_dir, epochs=5)

    # ------------ Stage 1B RD ------------
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=400, gamma=0.8)
    stage1_rd_only(model, train_rd, optimizer, scheduler, device, save_dir, epochs=5)

    # ------------ Stage 2 Alternating ------------
    optimizer = optim.Adam(model.parameters(), lr=5e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=400, gamma=0.8)
    stage2_alternating(model, train_pc, train_rd, optimizer, scheduler, device, save_dir, epochs=15)

    # ------------ Stage 3 Mixed Unified ------------
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.8)
    stage3_mixed(model, train_pc, train_rd, optimizer, scheduler, device, save_dir, epochs=20)

    print("\nTraining Completed Successfully!\n")


if __name__ == "__main__":
    main_combined()
