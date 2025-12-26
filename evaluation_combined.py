#!/usr/bin/env python3
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
import random
from sklearn.neighbors import KNeighborsClassifier
from dataset_h5 import SoliHD5Dataset, TinyRadarDataset, collate_sequences_padded
from pantomime import PantomimeDataset, collate_point_clouds
from models import PC_Encoder, RD_Encoder
from model_combined import Combined_Encoder
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from moco import MoCoWrapper
import matplotlib.pyplot as plt
import json
import pacmap

# -------------------------
# Helper functions
# -------------------------
@torch.no_grad()
def get_embeddings_and_labels(model, loader, device, moco=False, modality="RD"):
    """
    Returns:
      all_embeddings: torch.Tensor (N, D)
      all_labels: torch.Tensor (N,)
    Supports:
      - MoCoWrapper (moco=True or model has encoder_q)
      - combined_Encoder (returns loss, recovered, original, full_seq) -> we use CLS token full_seq[:,0,:]
    modality: "RD" or "PC" determines how to call combined model
    """
    all_embeddings = []
    all_labels = []
    model.eval()

    with torch.no_grad():
        for batch in tqdm(loader, desc="Embedding Extraction"):
            # loader collate returns (sequences, lengths, labels)
            sequences, lengths, labels = batch
            labels = labels.long().cpu()
            # Combined encoder expected signature: model(pc_batch, pc_len, rd_batch, rd_len)
            # Call it with missing modality depending on `modality`
            if modality == "PC":
                pc_batch = sequences.to(device)
                pc_len = lengths.to(device)
                rd_batch = None
                rd_len = None
                # _, recovered_seq, _ = model(pc_batch, pc_len, rd_batch, rd_len)
                # # full_seq : (B, 1+L_concat, D) where index 0 is CLS
                # emb = recovered_seq[:, 0, :].detach().cpu()
                emb = model(pc_batch, pc_len, rd_batch, rd_len).detach().cpu()
                emb = nn.functional.normalize(emb.float(), dim=1)
            elif modality == "RD":
                rd_batch = sequences.to(device)
                rd_len = lengths.to(device)
                pc_batch = None
                pc_len = None
                # _, recovered_seq, _ = model(pc_batch, pc_len, rd_batch, rd_len)
                # #emb = recovered_seq[:, 0, :].detach().cpu()
                # emb = recovered_seq.mean(dim=1).detach().cpu()
                emb = model(pc_batch, pc_len, rd_batch, rd_len).detach().cpu()
                emb = nn.functional.normalize(emb.float(), dim=1)
            else:
                # if modality == "BOTH" then sequences must be a tuple - not used here
                raise ValueError("modality must be 'PC' or 'RD' when using combined model")

            all_embeddings.append(emb)
            all_labels.append(labels)

    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    return all_embeddings, all_labels


def sample_N_per_class(dataset, N):
    print(f"Sampling {N} samples per class from the dataset...")
    class_to_indices = {}
    for idx in range(len(dataset)):
        item = dataset[idx]
        # dataset item shapes differ: try several common patterns
        if isinstance(item, tuple) and len(item) >= 2:
            # (sequence, label) or (seq, len, label)
            if isinstance(item[-1], int):
                label = item[-1]
            else:
                # assume last element is label tensor
                label = int(item[-1])
        else:
            raise RuntimeError("Dataset sample format not recognized in sample_N_per_class")

        class_to_indices.setdefault(label, []).append(idx)

    sampled_indices = []
    for label, indices in class_to_indices.items():
        sampled = random.sample(indices, min(N, len(indices)))
        sampled_indices.extend(sampled)
    return sampled_indices


def visualize_embeddings(train_embeddings, train_labels, eval_embeddings, eval_labels, output_dir):
    train_embeddings = np.array(train_embeddings)
    eval_embeddings = np.array(eval_embeddings)
    train_labels = np.array(train_labels)
    eval_labels = np.array(eval_labels)

    all_embeddings = np.vstack((train_embeddings, eval_embeddings))
    split_index = len(train_embeddings)

    reducer = pacmap.PaCMAP(n_components=2, random_state=42)
    reduced = reducer.fit_transform(all_embeddings)

    reduced_train = reduced[:split_index]
    reduced_eval = reduced[split_index:]

    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    unique_labels = np.unique(np.concatenate((train_labels, eval_labels)))
    colors = plt.cm.get_cmap('tab10', len(unique_labels))

    for i, label in enumerate(unique_labels):
        mask = (train_labels == label)
        axes[0].scatter(reduced_train[mask, 0], reduced_train[mask, 1], color=colors(i), label=str(label), alpha=0.7, s=20)
    axes[0].set_title("Training Embeddings (PaCMAP)")
    axes[0].legend(loc='best', fontsize='small')

    for i, label in enumerate(unique_labels):
        mask = (eval_labels == label)
        axes[1].scatter(reduced_eval[mask, 0], reduced_eval[mask, 1], color=colors(i), label=str(label), alpha=0.7, s=20)
    axes[1].set_title("Evaluation Embeddings (PaCMAP)")
    axes[1].legend(loc='best', fontsize='small')

    plt.tight_layout()
    output_path = os.path.join(output_dir, "embeddings.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Embedding visualization saved to {output_path}")


def knn_eval(model, train_dataset, train_loader, eval_loader, device, collate_fn, output_dir='zero_shot_checkpoints', N=5, k=5, modality="RD", moco=False):
    """
    model: either MoCoWrapper or combined_Encoder
    train_dataset: full dataset object to sample N per class
    train_loader: DataLoader for train_dataset (unused for sampling but used if you want)
    eval_loader: DataLoader for evaluation set
    modality: 'RD' or 'PC' for how to call the combined model
    moco: True if model is MoCoWrapper (will use encoder_q+proj_q)
    """
    os.makedirs(output_dir, exist_ok=True)

    sampled_indices = sample_N_per_class(train_dataset, N)
    sampled_dataset = torch.utils.data.Subset(train_dataset, sampled_indices)
    sampled_loader = DataLoader(sampled_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn, pin_memory=True)

    train_embeddings, train_labels = get_embeddings_and_labels(model, sampled_loader, device, moco=moco, modality=modality)
    eval_embeddings, eval_labels = get_embeddings_and_labels(model, eval_loader, device, moco=moco, modality=modality)

    # visualize
    visualize_embeddings(train_embeddings.numpy(), train_labels.numpy(), eval_embeddings.numpy(), eval_labels.numpy(), output_dir)

    knn = KNeighborsClassifier(n_neighbors=k, metric='cosine')
    knn.fit(train_embeddings.numpy(), train_labels.numpy())
    pred_labels = knn.predict(eval_embeddings.numpy())

    acc = (pred_labels == eval_labels.numpy()).mean() * 100
    print(f"KNN Clustering Accuracy: {acc:.2f}%")
    print('Saving confusion matrix..')
    os.makedirs(output_dir, exist_ok=True)
    cm = confusion_matrix(eval_labels.numpy(), pred_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig(f"{output_dir}/confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    print('Saving Classification Report')
    report_dict = classification_report(eval_labels.numpy(), pred_labels, output_dict=True)
    with open(f"{output_dir}/classification_report.json", "w") as f:
        json.dump(report_dict, f, indent=4)

    print(f"Saved results to {output_dir}")
    return acc

# -------------------------
# main_combined: evaluate your combined encoder on PC-only and RD-only datasets using CLS embeddings
# -------------------------
def main_combined():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size_pc', type=int, default=16)
    parser.add_argument('--batch_size_rd', type=int, default=8)
    parser.add_argument('--N', type=int, default=10, help='Number of samples per class for KNN')
    parser.add_argument('--K', type=int, default=5, help='Number of neighbors for KNN')
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # pc dataset/loaders
    data_pantomime_office = PantomimeDataset('./pantomime_foundational', environment='office')
    data_pantomime_open = PantomimeDataset('./pantomime_foundational', environment='open')
    combined_pc_dataset = ConcatDataset([data_pantomime_office, data_pantomime_open])
    pc_loader = DataLoader(combined_pc_dataset, batch_size=args.batch_size_pc, shuffle=True, collate_fn=collate_point_clouds)

    # rd dataset/loaders
    data_tiny_5g = TinyRadarDataset("./processed/5G")
    data_tiny_11g = TinyRadarDataset("./processed/11G")
    data_soli = SoliHD5Dataset("./SOLI")
    combined_rd_dataset = ConcatDataset([data_soli, data_tiny_5g, data_tiny_11g])
    rd_loader = DataLoader(data_soli, batch_size=args.batch_size_rd, shuffle=True, collate_fn=collate_sequences_padded)

    pc_encoder = PC_Encoder().to(device)
    rd_encoder = RD_Encoder().to(device)
    backbone = Combined_Encoder(
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

    backbone.eval()
    print("Loaded combined encoder checkpoint")

    class ClassifierHead(nn.Module):
        def __init__(self, backbone, input_dim=128, hidden_dim=256, output_dim=128):
            super().__init__()
            self.backbone = backbone
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            )
        
        def forward(self, pc_batch, pc_len, rd_batch, rd_len):
            _, recovered_seq, _ = self.backbone(pc_batch, pc_len, rd_batch, rd_len)
            cls_token = recovered_seq[:, 0, :]
            return self.mlp(cls_token)
    
    model = ClassifierHead(backbone).to(device)
    model.load_state_dict(torch.load('Combined_Checkpoints_supervised/Soli_with_MLP.pth')['model_state_dict'], strict=True)
    # Evaluate on PC-only (use combined model but pass RD=None)
    # acc_pc = knn_eval(model, combined_pc_dataset, None, pc_loader, device, collate_point_clouds,
    #                   output_dir=f'Combined_Results_improved/PC_{args.N}_{args.K}', N=args.N, k=args.K, modality="PC", moco=False)
    # print(f"PC-only KNN Accuracy: {acc_pc:.2f}%")

    # Evaluate on RD-only (use combined model but pass PC=None)
    acc_rd = knn_eval(model, data_soli, None, rd_loader, device, collate_sequences_padded,
                      output_dir=f'Combined_Results_supervised/RD_{args.N}_{args.K}', N=args.N, k=args.K, modality="RD", moco=False)
    print(f"RD-only KNN Accuracy: {acc_rd:.2f}%")

    # Optionally: evaluate combined (if you have both modalities together at eval time)
    # You can build a combined loader that returns paired items and call knn_eval with modality="BOTH" (not implemented here).


if __name__ == '__main__':
    # change these to main_PC(), main_RD() or main_combined() depending on which you want to run
    main_combined()
