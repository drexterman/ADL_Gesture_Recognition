import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import argparse
import random
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from model_combined import Combined_Encoder
from models import PC_Encoder, RD_Encoder
from dataset_h5 import SoliHD5Dataset, TinyRadarDataset, collate_sequences_padded
from pantomime import PantomimeDataset, collate_point_clouds
from torch.utils.data import DataLoader, ConcatDataset
from pytorch_metric_learning import losses, miners
import matplotlib.pyplot as plt
import json
import pacmap


# ======== Combined Loss Function ======== #
class CombinedLoss(nn.Module):
    def __init__(self, arcface_loss, triplet_loss, triplet_miner, lambda_arc=1.0, lambda_triplet=1.0):
        super().__init__()
        self.arcface_loss = arcface_loss
        self.triplet_loss = triplet_loss
        self.triplet_miner = triplet_miner
        self.lambda_arc = lambda_arc
        self.lambda_triplet = lambda_triplet

    def forward(self, embeddings, labels):
        arc_loss = self.arcface_loss(embeddings,labels)
        hard_triplets = self.triplet_miner(embeddings, labels)
        trip_loss = self.triplet_loss(embeddings, labels,hard_triplets)
        return self.lambda_arc * arc_loss + self.lambda_triplet * trip_loss


# ======== Training Epoch ======== #
def train_epoch(model, train_loader, criterion, optimizer, device, modality):
    model.train()
    total_loss = 0.0

    pbar = tqdm(train_loader, desc='Training')
    for sequences, lengths, labels in pbar:
        sequences, lengths, labels = sequences.to(device), lengths.to(device), labels.to(device)

        optimizer.zero_grad()
        if modality == 'PC':
                pc_batch = sequences.to(device)
                pc_len = lengths.to(device)
                rd_batch = None
                rd_len = None
                emb = model(pc_batch, pc_len, rd_batch, rd_len)
                emb = nn.functional.normalize(emb.float(), dim=1)
        else:
                rd_batch = sequences.to(device)
                rd_len = lengths.to(device)
                pc_batch = None
                pc_len = None
                emb = model(pc_batch, pc_len, rd_batch, rd_len)
                emb = nn.functional.normalize(emb.float(), dim=1)
        loss = criterion(emb, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / len(train_loader)

def evaluate(model, eval_loader, criterion, device, modality):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for sequences, lengths, labels in eval_loader:
            sequences = sequences.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)
            
            if modality == 'PC':
                    pc_batch = sequences.to(device)
                    pc_len = lengths.to(device)
                    rd_batch = None
                    rd_len = None
                    emb = model(pc_batch, pc_len, rd_batch, rd_len)
                    emb = nn.functional.normalize(emb.float(), dim=1)
            else:
                    rd_batch = sequences.to(device)
                    rd_len = lengths.to(device)
                    pc_batch = None
                    pc_len = None
                    emb = model(pc_batch, pc_len, rd_batch, rd_len)
                    emb = nn.functional.normalize(emb.float(), dim=1)

            loss = criterion(emb, labels)
            total_loss += loss.item()
    
    return total_loss / len(eval_loader)

# ============== Testing Helper Functions================ #
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

def knn_eval(model, train_dataset, eval_loader, device, collate_fn, output_dir='zero_shot_checkpoints', N=5, k=5, modality="RD", moco=False):
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
# ============================================================
# DATA LOADING
# ============================================================

def load_pc_dataset(batch_size=16):
    data1 = PantomimeDataset('./pantomime_foundational', environment='office')
    data2 = PantomimeDataset('./pantomime_foundational', environment='open')
    combined = ConcatDataset([data1, data2])

    train, val, test = torch.utils.data.random_split(
        combined,
        [int(0.5*len(combined)), int(0.1*len(combined)) ,len(combined) - int(0.5*len(combined)) - int(0.1*len(combined))]
    )

    train_loader = DataLoader(train, batch_size=batch_size,
                              shuffle=True, collate_fn=collate_point_clouds,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size,
                            shuffle=False, collate_fn=collate_point_clouds,
                            num_workers=4, pin_memory=True)
    test_loader = DataLoader(test, batch_size=batch_size,
                            shuffle=False, collate_fn=collate_point_clouds,
                            num_workers=4, pin_memory=True)
    return train, train_loader, val_loader, test_loader

def load_rd_dataset(dataset,batch_size=8):
    if dataset =='Soli':
        data = ConcatDataset([
            SoliHD5Dataset("./SOLI"),
        ])
    else:
        data = ConcatDataset([
            TinyRadarDataset("./processed/5G"),
            TinyRadarDataset("./processed/11G"),
        ])
        # Remove samples with label 0
        indices_to_keep = [i for i in range(len(data)) if data[i][-1] != -1]
        data = torch.utils.data.Subset(data, indices_to_keep)
    train, val, test = torch.utils.data.random_split(
        data,
        [int(0.5*len(data)), int(0.1*len(data)) ,len(data) - int(0.5*len(data)) - int(0.1*len(data))]
    )
    print(f'length of dataset = {len(train), len(val), len(test)}')
    train_loader = DataLoader(train, batch_size=batch_size,
                              shuffle=True, collate_fn=collate_sequences_padded,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size,
                            shuffle=False, collate_fn=collate_sequences_padded,
                            num_workers=4, pin_memory=True)
    test_loader = DataLoader(test, batch_size=batch_size,
                            shuffle=False, collate_fn=collate_sequences_padded,
                            num_workers=4, pin_memory=True)
    return train, train_loader, val_loader, test_loader


# ======== Main ======== #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arcface_s', type=float, default=30.0)
    parser.add_argument('--arcface_m', type=float, default=0.5)
    parser.add_argument('--dataset',type = str, default = 'Soli')
    parser.add_argument('--N', type=int, default=10, help='Number of samples per class for KNN')
    parser.add_argument('--K', type=int, default=5, help='Number of neighbors for KNN')
    args = parser.parse_args()

    config = {
        'batch_size': 32,
        'epochs': 25,
        'lr': 1e-5,
        'initial_margin': 0.1,
        'final_margin': 0.5,
        'lambda_arc': 1.0,
        'lambda_triplet': 0.5,
        'momentum': 0.9,
        'patience': 15,
        'lr_schedule': True,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    print("=" * 70)
    print("Fine-tuning Gesture Encoder (Triplet + ArcFace)")
    print("=" * 70)

    for k, v in config.items():
        print(f"   {k}: {v}")

    checkpoint_dir = 'Combined_Checkpoints_supervised'
    os.makedirs(checkpoint_dir, exist_ok=True)

    print("\n📚 Loading datasets...")
    # datasets
    modality = 'none'
    if args.dataset == 'Soli':
        train_dataset, train, val, test = load_rd_dataset(dataset='Soli',batch_size=16)
        modality = 'RD'
        collate = collate_sequences_padded
    elif args.dataset == 'Tiny_Radar':
         train_dataset, train, val, test = load_rd_dataset(dataset='Tiny_Radar',batch_size=16)
         modality = 'RD'
         collate = collate_sequences_padded
    else:
        train_dataset, train, val, test = load_pc_dataset(batch_size=16)
        modality = 'PC'
        collate = collate_point_clouds
    
    print("\n🧠 Creating model...")
    pc_encoder = PC_Encoder().to(config['device'])
    rd_encoder = RD_Encoder().to(config['device'])
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
    ).to(config['device'])

    print(f"   Model: {backbone.__class__.__name__}")
    print(f"   Parameters: {sum(p.numel() for p in backbone.parameters()):,}")
    print(f"   Device: {config['device']}")
    
    backbone.load_state_dict(torch.load('Combined_Checkpoints_improved/stage2_best.pth')['model'], strict=True)
    print("✅ Loaded pretrained weights")

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

    model = ClassifierHead(backbone).to(config['device'])

    if modality == 'RD':
        arcface_loss = losses.ArcFaceLoss(num_classes=12, embedding_size=128,scale=args.arcface_s, margin=args.arcface_m).to(config['device'])
    else:
        arcface_loss = losses.ArcFaceLoss(num_classes=21, embedding_size=128,scale=args.arcface_s, margin=args.arcface_m).to(config['device'])
    triplet_miner = miners.TripletMarginMiner(margin=config['initial_margin'], type_of_triplets="semihard")
    triplet_loss = losses.TripletMarginLoss(margin=config['initial_margin'])
    criterion = CombinedLoss(arcface_loss, triplet_loss, triplet_miner, config['lambda_arc'], config['lambda_triplet'])
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5) if config['lr_schedule'] else None

    print("\n🚀 Starting Fine-tuning...")
    best_loss = float('inf')
    patience_counter = 0
    current_margin = config['initial_margin']
    loss_list = []
    epoch_list =[]
    for epoch in range(1, config['epochs'] + 1):
        if (epoch) % 10 == 0 and epoch != config['epochs']:
            current_margin = min(
            criterion.triplet_loss.margin + (config['final_margin']-config['initial_margin'])/(config['epochs']/10), config['final_margin']
            )
        criterion.triplet_loss.margin = current_margin
        criterion.triplet_miner.margin = current_margin
        torch.cuda.empty_cache()
        print(f"\nEpoch {epoch}/{config['epochs']}, triplet margin: {current_margin:.4f}")

        train_loss = train_epoch(model, train, criterion, optimizer, config['device'], modality = modality)
        eval_loss = evaluate(model, val, criterion, config['device'], modality= modality)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Eval  - Loss: {eval_loss:.4f}, Normalized Eval Loss: {eval_loss/current_margin:.4f}")

        if scheduler:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"   LR: {current_lr:.6f}")
            scheduler.step()

        if eval_loss/current_margin < best_loss:
            best_loss = eval_loss/current_margin
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'eval_loss': eval_loss
            }, os.path.join(checkpoint_dir, f'{args.dataset}_with_MLP.pth'))
            print(f"✅ Saved best model (loss: {eval_loss:.2f})")
        else:
            patience_counter += 1
        
        if patience_counter >= config['patience']:
            print(f"\n⚠️ Early stopping triggered after {epoch} epochs")
            break
        epoch_list.append(epoch)
        loss_list.append(eval_loss/current_margin)
    plt.plot(epoch_list,loss_list)
    plt.xlabel('Epochs')
    plt.ylabel('Normalized Loss')
    plt.savefig(f'Combined_Results_supervised/{args.dataset}_{args.N}_{args.K}/Loss_Curve.png')
    print("\n" + "=" * 70)
    print(f"Training Complete! Best Eval Loss: {best_loss:.2f}")
    print("\n" + "=" * 70)
    print('Testing')
    acc = knn_eval(model, train_dataset, test, config['device'], collate_fn=collate,
                      output_dir=f'Combined_Results_supervised/{args.dataset}_{args.N}_{args.K}', N=args.N, k=args.K, modality=modality, moco=False)
    print(f"{args.dataset}-only KNN Accuracy: {acc:.2f}%")
    print("=" * 70)
    return best_loss


if __name__ == "__main__":
    try:
        best_loss = main()
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        raise