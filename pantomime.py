"""
PyTorch Dataset for Pantomime point cloud gesture recognition.
Loads preprocessed point cloud data in .npy format.
"""

import glob
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class PantomimeDataset(Dataset):
    """
    PyTorch Dataset that loads preprocessed Pantomime point cloud data from .npy files.

    Works with files produced by process_panto.py and compile_panto.py.
    Each .npy file contains point cloud sequence with shape (8, 64, 3).

    Returns:
        sequence: (T, N, 3) tensor where T=8 frames, N=64 points, 3=xyz
        label: int (gesture class 0-20)
    """

    def __init__(self, dataset_dir, normalize=True, environment=None):
        """
        Args:
            dataset_dir: Root directory containing processed .npy files
                        e.g., './processed_dataset' or './pantomime_foundational'
            normalize: Whether to normalize point clouds to [-1, 1]
            environment: Optional filter for specific environment ('office' or 'open')
                        If None, loads all environments
        """
        self.dataset_dir = Path(dataset_dir)
        self.normalize = normalize
        self.environment = environment
        
        # Initialize file_paths as empty list (will be populated if not using foundational)
        self.file_paths = []

        # Find all .npy files recursively
        if environment:
            # If using compiled foundational dataset
            pattern = f"pantomime_{environment}_X.npy"
            if (self.dataset_dir / pattern).exists():
                self._load_foundational(environment)
            else:
                # If using processed_dataset structure
                search_path = self.dataset_dir / environment / "**" / "*.npy"
                self.file_paths = sorted(glob.glob(str(search_path), recursive=True))
                self._extract_labels_from_paths()
        else:
            # Check if foundational datasets exist (both office and open)
            office_exists = (self.dataset_dir / "pantomime_office_X.npy").exists()
            open_exists = (self.dataset_dir / "pantomime_open_X.npy").exists()
            
            if office_exists or open_exists:
                # Load foundational format (combine both if available)
                samples_list = []
                labels_list = []
                
                if office_exists:
                    X = np.load(self.dataset_dir / "pantomime_office_X.npy")
                    y = np.load(self.dataset_dir / "pantomime_office_y.npy")
                    samples_list.append(X)
                    labels_list.append(y)
                    print(f"   Loaded office: {X.shape[0]} samples")
                
                if open_exists:
                    X = np.load(self.dataset_dir / "pantomime_open_X.npy")
                    y = np.load(self.dataset_dir / "pantomime_open_y.npy")
                    samples_list.append(X)
                    labels_list.append(y)
                    print(f"   Loaded open: {X.shape[0]} samples")
                
                # Concatenate all loaded data
                self.samples = np.concatenate(samples_list, axis=0)
                self.labels = np.concatenate(labels_list, axis=0)
                
                print(f"   Total combined samples: {self.samples.shape}")
            else:
                # Load all .npy files from processed_dataset structure
                self.file_paths = sorted(
                    glob.glob(str(self.dataset_dir / "**" / "*.npy"), recursive=True)
                )
                self._extract_labels_from_paths()

        # Validation check
        if len(self.file_paths) == 0 and not hasattr(self, 'samples'):
            raise ValueError(f"No .npy files found in {dataset_dir}")

        if hasattr(self, 'samples'):
            print(f"   Found {len(self.samples)} samples in {dataset_dir}/{environment if environment else 'all'}")
        else:
            print(f"   Found {len(self.file_paths)} samples in {dataset_dir}")
    def _load_foundational(self, environment):
        """Load from compiled foundational dataset format."""
        X_path = self.dataset_dir / f"pantomime_{environment}_X.npy"
        y_path = self.dataset_dir / f"pantomime_{environment}_y.npy"

        self.samples = np.load(X_path)  # (N, 8, 64, 3)
        self.labels = np.load(y_path)   # (N,)

        print(f"   Loaded foundational dataset:")
        print(f"   Samples shape: {self.samples.shape}")
        print(f"   Labels shape: {self.labels.shape}")

    def _extract_labels_from_paths(self):
        """Extract labels from directory structure."""
        self.labels = []
        
        for file_path in self.file_paths:
            path_parts = Path(file_path).parts
            
            try:
                # Find 'processed_dataset' index
                base_idx = path_parts.index('processed_dataset')
                
                # Path structure: processed_dataset/environment/SUBJECT/CLASS/file.npy
                # Class ID is at position base_idx + 3 (not +2!)
                if len(path_parts) > base_idx + 3:
                    class_folder = path_parts[base_idx + 3]  # Changed from +2 to +3
                    
                    if class_folder.isdigit():
                        # Pantomime classes are 1-21, convert to 0-20
                        label = int(class_folder) - 1
                        self.labels.append(label)
                    else:
                        print(f"Warning: Non-numeric class folder '{class_folder}' in {file_path}")
                        self.labels.append(0)
                else:
                    print(f"Warning: Unexpected path structure in {file_path}")
                    self.labels.append(0)
            except ValueError:
                # 'processed_dataset' not in path
                print(f"Warning: 'processed_dataset' not found in path {file_path}")
                self.labels.append(0)

    def __len__(self):
        if hasattr(self, 'samples'):
            return len(self.samples)
        return len(self.file_paths)

    def __getitem__(self, idx):
        """
        Returns:
            sequence: (T, N, 3) float tensor where T=8, N=64, (x,y,z)
            label: int
        """
        if hasattr(self, 'samples'):
            # Load from pre-compiled arrays
            sequence = self.samples[idx]  # (8, 64, 3)
            label = int(self.labels[idx])
        else:
            # Load from individual files
            npy_path = self.file_paths[idx]
            sequence = np.load(npy_path)  # (8, 64, 3)
            label = self.labels[idx]

        # Validate shape
        assert sequence.shape == (8, 64, 3), \
            f"Invalid shape {sequence.shape} in {npy_path if not hasattr(self, 'samples') else 'compiled dataset'}"

        # Convert to tensor
        sequence = torch.from_numpy(sequence).float()

        # Normalize to [-1, 1] if requested
        if self.normalize:
            # Normalize each coordinate dimension independently
            seq_min = sequence.min()
            seq_max = sequence.max()
            if seq_max > seq_min:
                sequence = 2.0 * (sequence - seq_min) / (seq_max - seq_min) - 1.0

        return sequence, label


# =============================================================================
# Collate Function for Point Clouds
# =============================================================================

def collate_point_clouds(batch):
    """
    Custom collate function for point cloud sequences.
    
    Args:
        batch: List of tuples [(sequence, label), ...]
               where sequence is (T, N, 3)
    
    Returns:
        sequences: Tensor (B, T, N, 3)
        labels: Tensor (B,)
        lengths: Tensor (B,) - all same length for Pantomime (T=8)
    """
    sequences = []
    labels = []
    
    for seq, label in batch:
        sequences.append(seq)
        labels.append(label)
    
    # Stack sequences (all have same shape for Pantomime)
    sequences = torch.stack(sequences)  # (B, T, N, 3)
    labels = torch.tensor(labels, dtype=torch.long)
    
    # All sequences have same length (8 frames)
    lengths = torch.full((len(batch),), 8, dtype=torch.long)
    
    return sequences, lengths, labels


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Testing PantomimeDataset")
    print("=" * 70)

    # Test 1: Load from processed_dataset structure
    processed_dir = Path("./processed_dataset")
    if processed_dir.exists():
        print("\n✓ Testing with processed_dataset structure...")
        try:
            ds = PantomimeDataset("./processed_dataset")
            seq, label = ds[0]
            print(f"   Samples: {len(ds)}")
            print(f"   Sequence shape: {seq.shape}")  # Should be (8, 64, 3)
            print(f"   Label: {label}")
            print(f"   Coordinate range: [{seq.min():.3f}, {seq.max():.3f}]")
        except Exception as e:
            print(f"   Error: {e}")

    # Test 2: Load from foundational dataset
    foundational_dir = Path("./pantomime_foundational")
    if foundational_dir.exists():
        print("\n✓ Testing with foundational dataset...")
        try:
            ds_office = PantomimeDataset(
                "./pantomime_foundational", environment="office"
            )
            seq, label = ds_office[0]
            print(f"   Office samples: {len(ds_office)}")
            print(f"   Sequence shape: {seq.shape}")
            
            ds_open = PantomimeDataset(
                "./pantomime_foundational", environment="open"
            )
            print(f"   Open samples: {len(ds_open)}")
        except Exception as e:
            print(f"   Error: {e}")

    # Test 3: Test collate function
    print("\n✓ Testing collate function...")
    if processed_dir.exists():
        try:
            ds = PantomimeDataset("./processed_dataset")
            from torch.utils.data import DataLoader
            
            loader = DataLoader(
                ds, batch_size=4, shuffle=True, collate_fn=collate_point_clouds
            )
            sequences, labels, lengths = next(iter(loader))
            
            print(f"   Batch sequences shape: {sequences.shape}")  # (4, 8, 64, 3)
            print(f"   Batch labels shape: {labels.shape}")  # (4,)
            print(f"   Batch lengths: {lengths}")  # [8, 8, 8, 8]
        except Exception as e:
            print(f"   Error: {e}")

    print("\n✓ Tests complete!")