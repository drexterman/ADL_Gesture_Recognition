import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, BatchSampler, ConcatDataset
from tqdm import tqdm
import random
from sklearn.neighbors import KNeighborsClassifier
from dataset_h5 import SoliHD5Dataset, TinyRadarDataset, collate_sequences_padded
from pantomime import PantomimeDataset, collate_point_clouds
from models import P4Transformer, RD_ViViT_Encoder
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix ,ConfusionMatrixDisplay
from moco import MoCoWrapper
import matplotlib.pyplot as plt
import json
import pacmap

@torch.no_grad()
def evaluate(model, loader, device, text_embeddings, classes_sorted):
    model.eval()
    correct = 0
    total = 0
    for sequences, lengths, labels in tqdm(loader, desc='Evaluating'):
        sequences = sequences.to(device)
        lengths = lengths.to(device)
        labels = labels.to(device, dtype=torch.long)
        _ = model(sequences, lengths)
        gesture_emb = model.embeddings
        gesture_emb = nn.functional.normalize(gesture_emb.float(), dim=1)
        logits = gesture_emb @ text_embeddings.T
        preds = logits.argmax(dim=1)
        # Map back to original class ids
        preds = [classes_sorted[p] for p in preds.cpu().tolist()]
        labels_orig = labels.cpu().tolist()
        for p, l in zip(preds, labels_orig):
            if p == l:
                correct += 1
            total += 1
    acc = 100.0 * correct / total if total > 0 else 0.0
    return acc

def get_embeddings_and_labels(model, loader, device, moco=False):
    all_embeddings, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for sequences, lengths, labels in tqdm(loader, desc="Embedding Extraction"):
            sequences = sequences.to(device)
            lengths = lengths.to(device)
            if moco:
                feat = model.encoder_q(sequences, lengths)
                emb = model.proj_q(feat)
            else:
                emb = model(sequences, lengths)
            emb = nn.functional.normalize(emb.float(), dim=1)
            all_embeddings.append(emb.cpu())
            all_labels.append(labels.cpu())
    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    return all_embeddings, all_labels

def sample_N_per_class(dataset, N):
    print(f"Sampling {N} samples per class from the dataset...")
    class_to_indices = {}
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        class_to_indices.setdefault(label, []).append(idx)
    sampled_indices = []
    for label, indices in class_to_indices.items():
        sampled = random.sample(indices, min(N, len(indices)))
        sampled_indices.extend(sampled)
    return sampled_indices

def visualize_embeddings(train_embeddings, train_labels, eval_embeddings, eval_labels, output_dir):
    # Ensure numpy arrays
    train_embeddings = np.array(train_embeddings)
    eval_embeddings = np.array(eval_embeddings)
    train_labels = np.array(train_labels)
    eval_labels = np.array(eval_labels)

    # Combine embeddings for joint reduction
    all_embeddings = np.vstack((train_embeddings, eval_embeddings))
    split_index = len(train_embeddings)

    # Reduce to 2D using PaCMAP
    reducer = pacmap.PaCMAP(n_components=2, random_state=42)
    reduced = reducer.fit_transform(all_embeddings)

    # Split reduced embeddings back
    reduced_train = reduced[:split_index]
    reduced_eval = reduced[split_index:]
    #print(f'all_embeddings shape: {all_embeddings.shape}\nreduced_train shape: {reduced_train.shape}\nreduced_eval shape: {reduced_eval.shape}')
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Prepare figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Get consistent color mapping for all labels
    unique_labels = np.unique(np.concatenate((train_labels, eval_labels)))
    colors = plt.cm.get_cmap('tab10', len(unique_labels))

    # --- Plot Training Embeddings ---
    for i, label in enumerate(unique_labels):
        mask = (train_labels == label)
        axes[0].scatter(
            reduced_train[mask, 0],
            reduced_train[mask, 1],
            color=colors(i),
            label=str(label),
            alpha=0.7,
            s=20
        )
    axes[0].set_title("Training Embeddings (PaCMAP)")
    axes[0].set_xlabel("PaCMAP Dimension 1")
    axes[0].set_ylabel("PaCMAP Dimension 2")
    axes[0].legend(loc='best', fontsize='small')

    # --- Plot Evaluation Embeddings ---
    for i, label in enumerate(unique_labels):
        mask = (eval_labels == label)
        axes[1].scatter(
            reduced_eval[mask, 0],
            reduced_eval[mask, 1],
            color=colors(i),
            label=str(label),
            alpha=0.7,
            s=20
        )
    axes[1].set_title("Evaluation Embeddings (PaCMAP)")
    axes[1].set_xlabel("PaCMAP Dimension 1")
    axes[1].set_ylabel("PaCMAP Dimension 2")
    axes[1].legend(loc='best', fontsize='small')

    plt.tight_layout()

    # Save the figure
    output_path = os.path.join(output_dir, "embeddings.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Embedding visualization saved to {output_path}")

def knn_eval(model, train_dataset, eval_loader, device, collate_fn, output_dir='zero_shot_checkpoints',N=5, k=5):
    # Sample N gestures per class from train set
    sampled_indices = sample_N_per_class(train_dataset, N)
    sampled_dataset = torch.utils.data.Subset(train_dataset, sampled_indices)
    sampled_loader = DataLoader(sampled_dataset, batch_size=16, shuffle=False,
                               collate_fn=collate_fn, pin_memory=True)
    train_embeddings, train_labels = get_embeddings_and_labels(model, sampled_loader, device)
    eval_embeddings, eval_labels = get_embeddings_and_labels(model, eval_loader, device)
    visualize_embeddings(train_embeddings, train_labels, eval_embeddings, eval_labels, output_dir)
    # KNN classifier
    knn = KNeighborsClassifier(n_neighbors=k, metric='cosine')
    knn.fit(train_embeddings.numpy(), train_labels.numpy())
    pred_labels = knn.predict(eval_embeddings.numpy())

    # Accuracy
    acc = (pred_labels == eval_labels.numpy()).mean() * 100
    print(f"KNN Clustering Accuracy: {acc:.2f}%")
    print('Saving confusion matrix..')
    os.makedirs(output_dir, exist_ok=True)
    cm = confusion_matrix(eval_labels.numpy(), pred_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)  # You can change the colormap if you like
    plt.title("Confusion Matrix")
    plt.savefig(f"{output_dir}/confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    print('Saving classification report..')
    report_dict = classification_report(eval_labels.numpy(), pred_labels, output_dict=True)
    output_filename = f"{output_dir}/classification_report.json"
    with open(output_filename, 'w') as json_file:
        json.dump(report_dict, json_file, indent=4)
    return acc

def main_RD():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--N', type=int, default=10, help='Number of samples per class for KNN')
    parser.add_argument('--K', type=int, default=5, help='Number of neighbors for KNN')
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    data_tiny_5g = TinyRadarDataset("./processed/5G") 
    data_tiny_11g = TinyRadarDataset("./processed/11G")
    data_soli = SoliHD5Dataset("./SOLI")

    combined_rd_dataset = ConcatDataset([data_soli, data_tiny_5g, data_tiny_11g])
    rd_loader = DataLoader(
        combined_rd_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_sequences_padded,
    )

    # Load model
    # RD_model = RD_Encoder()
    # RD_model.to(device)

    # ckpt = torch.load('Encoder_Checkpoints/best_RD_encoder.pth', map_location=device)
    # RD_model.load_state_dict(ckpt['model_state_dict'], strict=False)
    # RD_model.eval()
    # print("Loaded RD model from Encoder_Checkpoints/best_RD_encoder.pth")
    # acc = knn_eval(RD_model, combined_rd_dataset, rd_loader, device, collate_sequences_padded, output_dir =f'Encoder_Results/RD_{args.N}_{args.K}' ,N=args.N, k=args.K)

    # print(f"\nEvaluation Accuracy: {acc:.2f}%")

        # Load model
    RD_model = RD_ViViT_Encoder()
    RD_model.to(device)
    queue_size = 8192  # try 8192 or 4096; tune for memory/speed
    moco_m = 0.999
    temperature = 0.1
    moco_RD = MoCoWrapper(base_encoder=RD_model, feat_dim=512, K=queue_size, m=moco_m, T=temperature, device=device)
    moco_RD = moco_RD.to(device)
    ckpt = torch.load('Encoder_Checkpoints/best_ViViT_RD_encoder_with_moco.pth', map_location=device)
    # RD_model.load_state_dict(ckpt['model_state_dict'], strict=False)
    # RD_model.eval()
    moco_RD.load_state_dict(ckpt['model_state_dict'], strict=False)
    moco_RD.eval()
    print("Loaded RD model from Encoder_Checkpoints/best_ViViT_RD_encoder_with_moco.pth")
    acc = knn_eval(moco_RD, combined_rd_dataset, rd_loader, device, collate_sequences_padded ,output_dir =f'Encoder_with_moco_Results/ViViT_RD_{args.N}_{args.K}' ,N=args.N, k=args.K)
    
    print(f"\nEvaluation Accuracy: {acc:.2f}%")

def main_PC():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--N', type=int, default=10, help='Number of samples per class for KNN')
    parser.add_argument('--K', type=int, default=5, help='Number of neighbors for KNN')
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    data_pantomime_office = PantomimeDataset('./pantomime_foundational', environment='office')
    data_pantomime_open = PantomimeDataset('./pantomime_foundational', environment='open')

    combined_pc_dataset = ConcatDataset([data_pantomime_office, data_pantomime_open])
    pc_loader = DataLoader(
        combined_pc_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_point_clouds,
    )

    # Load model
    PC_model = P4Transformer()
    PC_model.to(device)
    queue_size = 8192  # try 8192 or 4096; tune for memory/speed
    moco_m = 0.999
    temperature = 0.1
    moco_PC = MoCoWrapper(base_encoder=PC_model, feat_dim=512, K=queue_size, m=moco_m, T=temperature, device=device)
    moco_PC = moco_PC.to(device)
    ckpt = torch.load('Encoder_Checkpoints/best_P4_PC_encoder_with_moco.pth', map_location=device)
    # PC_model.load_state_dict(ckpt['model_state_dict'], strict=False)
    # PC_model.eval()
    moco_PC.load_state_dict(ckpt['model_state_dict'], strict=False)
    moco_PC.eval()
    print("Loaded PC model from Encoder_Checkpoints/best_P4_PC_encoder_with_moco.pth")
    acc = knn_eval(moco_PC, combined_pc_dataset, pc_loader, device, collate_point_clouds ,output_dir =f'Encoder_with_moco_Results/P4_PC_{args.N}_{args.K}' ,N=args.N, k=args.K)

    print(f"\nEvaluation Accuracy: {acc:.2f}%")

if __name__ == '__main__':
    main_PC()