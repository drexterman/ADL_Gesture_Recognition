"""
PyTorch Datasets for radar gesture recognition.
Supports TinyRadar (5G/11G), Soli, and original Soli HDF5 formats.
"""

import glob
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torchvision import transforms

# =============================================================================
# TinyRadar Dataset (for preprocessed 5G/11G data)
# =============================================================================


class TinyRadarDataset(Dataset):
    """
    PyTorch Dataset that loads preprocessed TinyRadar data from HDF5 files.

    Works with files produced by create_foundational_dataset.py.
    Each HDF5 file contains:
    - ch0: RDI sequence, shape (T, 64, 64) where T=40
    - label: Gesture label (int)
    - gesture_name (attribute): Human-readable gesture name

    Returns:
        sequence: (T, H, W) tensor
        label: int
    """

    def __init__(self, dataset_dir, normalize=True):
        """
        Args:
            dataset_dir: Root directory containing processed HDF5 files
                         e.g., './processed/5G' or './processed/11G'
            normalize: Whether to normalize data to [-1, 1]
        """
        self.dataset_dir = Path(dataset_dir)
        self.normalize = normalize

        # Find all .h5 files recursively
        self.file_paths = sorted(
            glob.glob(str(self.dataset_dir / "**" / "*.h5"), recursive=True)
        )

        if len(self.file_paths) == 0:
            raise ValueError(f"No .h5 files found in {dataset_dir}")

        print(f"   Found {len(self.file_paths)} samples in {dataset_dir}")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        """
        Returns:
            sequence: (T, H, W) float tensor
            label: int
        """
        h5_path = self.file_paths[idx]

        with h5py.File(h5_path, "r") as f:
            # Load RDI sequence: (T, 64, 64)
            sequence = f["ch0"][:]
            label = int(f["label"][()])

        # Convert to tensor
        sequence = torch.from_numpy(sequence).float()

        # Normalize to [-1, 1] if requested
        if self.normalize:
            seq_min = sequence.min()
            seq_max = sequence.max()
            if seq_max > seq_min:
                sequence = 2.0 * (sequence - seq_min) / (seq_max - seq_min) - 1.0

        return sequence, label


# =============================================================================
# Soli Dataset (simple loader for SOLI folder)
# =============================================================================


class SoliHD5Dataset(Dataset):
    """
    PyTorch Dataset that loads Soli radar data from HDF5 files in the SOLI folder.

    Each HDF5 file (e.g., 0_0_0.h5) contains:
    - ch0, ch1, ch2, ch3: Radar channels
    - label: Gesture label

    Returns:
        sequence: (T, H, W) tensor (uses ch0, reshaped/resized to 64x64)
        label: int
    """

    def __init__(self, dataset_dir, normalize=True):
        """
        Args:
            dataset_dir: Directory containing Soli HDF5 files (e.g., './SOLI')
            normalize: Whether to normalize data
        """
        self.dataset_dir = Path(dataset_dir)
        self.normalize = normalize

        # Find all .h5 files
        self.file_paths = sorted(glob.glob(str(self.dataset_dir / "*.h5")))

        if len(self.file_paths) == 0:
            raise ValueError(f"No .h5 files found in {dataset_dir}")

        print(f"   Found {len(self.file_paths)} samples in {dataset_dir}")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        """
        Returns:
            sequence: (T, H, W) float tensor
            label: int
        """
        h5_path = self.file_paths[idx]

        with h5py.File(h5_path, "r") as f:
            # Load ch0 (main channel)
            if "ch0" in f:
                data = f["ch0"][:]
            else:
                # Fallback: find first channel
                for key in f.keys():
                    if key.startswith("ch"):
                        data = f[key][:]
                        break

            # Get label
            label = f["label"][()]
            if hasattr(label, "__len__") and len(label.shape) > 0:
                label = int(label.flat[0])
            else:
                label = int(label)

        # Reshape if needed: Soli data is often (T, 1024) -> (T, 32, 32)
        if len(data.shape) == 2 and data.shape[1] == 1024:
            T = data.shape[0]
            data = data.reshape(T, 32, 32)

        # Resize to 64x64 if needed for consistency
        if len(data.shape) >= 2 and data.shape[-1] == 32:
            from skimage.transform import resize

            T = data.shape[0]
            resized = np.zeros((T, 64, 64), dtype=np.float32)
            for t in range(T):
                resized[t] = resize(
                    data[t], (64, 64), mode="reflect", anti_aliasing=True
                )
            data = resized

        sequence = torch.from_numpy(data.astype(np.float32)).float()

        if self.normalize:
            seq_min = sequence.min()
            seq_max = sequence.max()
            if seq_max > seq_min:
                sequence = 2.0 * (sequence - seq_min) / (seq_max - seq_min) - 1.0

        return sequence, label


# =============================================================================
# Original Soli HDF5 Dataset (with config file and multi-channel support)
# =============================================================================


class SoliHDF5DatasetOriginal(Dataset):
    """
    PyTorch Dataset that loads Soli radar data from HDF5 files with config-based splits.

    Each HDF5 file contains:
    - ch0, ch1, ch2, ch3: Radar channels, shape (frames, 1024)
    - label: Gesture label per frame

    The data is reshaped from 1024 vector to 32x32 image per channel.
    """

    def __init__(
        self,
        dataset_dir,
        config_file,
        split="train",
        num_channels=4,
        temporal_frames=4,
        normalize=True,
    ):
        """
        Args:
            dataset_dir: Directory containing HDF5 files
            config_file: JSON file with train/eval split
            split: 'train' or 'eval'
            num_channels: Number of radar channels to use (default 4)
            temporal_frames: Number of consecutive frames to stack (default 4)
            normalize: Whether to normalize data to [-1, 1]
        """
        self.dataset_dir = Path(dataset_dir)
        self.num_channels = num_channels
        self.temporal_frames = temporal_frames
        self.normalize = normalize

        # Load sample list
        with open(config_file, "r") as f:
            config = json.load(f)

        if split not in config:
            raise ValueError(
                f"Split '{split}' not found in config. Available: {list(config.keys())}"
            )

        # Filter out samples with zero frames
        all_samples = config[split]
        self.sample_ids = []
        skipped = 0

        for sample_id in all_samples:
            h5_path = self.dataset_dir / f"{sample_id}.h5"
            if h5_path.exists():
                try:
                    with h5py.File(h5_path, "r") as f:
                        if "ch0" in f and len(f["ch0"]) > 0:
                            self.sample_ids.append(sample_id)
                        else:
                            skipped += 1
                except:
                    skipped += 1

        self.split = split
        if skipped > 0:
            print(f"   Warning: Skipped {skipped} empty/invalid samples in {split} set")

    def __len__(self):
        return len(self.sample_ids)

    def _load_h5_sample(self, h5_path):
        """Load radar data and label from HDF5 file."""
        with h5py.File(h5_path, "r") as f:
            channels = []
            for ch_idx in range(self.num_channels):
                ch_name = f"ch{ch_idx}"
                if ch_name in f:
                    data = f[ch_name][:]
                    channels.append(data)

            radar_data = np.stack(channels, axis=1)
            frames, channels_dim, _ = radar_data.shape
            images = radar_data.reshape(frames, channels_dim, 32, 32)

            label = f["label"][()]
            if len(label.shape) > 0:
                label = label[0, 0] if label.shape[1] > 0 else label[0]

            return images, label

    def _stack_temporal_frames(self, images, frame_idx):
        """Stack temporal frames for a given frame index."""
        stacked = []
        total_frames = len(images)

        for t in range(self.temporal_frames):
            idx = min(frame_idx + t, total_frames - 1)
            stacked.append(images[idx])

        return np.concatenate(stacked, axis=0)

    def __getitem__(self, idx):
        """
        Returns:
            sequence_tensor: (T, C, H, W) tensor
            label: int
        """
        sample_id = self.sample_ids[idx]
        h5_path = self.dataset_dir / f"{sample_id}.h5"

        images, label = self._load_h5_sample(h5_path)

        if self.normalize:
            images = images * 2.0 - 1.0

        sequence = []
        for frame_idx in range(len(images)):
            stacked_frame = self._stack_temporal_frames(images, frame_idx)
            frame_tensor = torch.from_numpy(stacked_frame).float()
            sequence.append(frame_tensor)

        sequence_tensor = torch.stack(sequence)

        return sequence_tensor, label


# =============================================================================
# Collate Functions
# =============================================================================


def collate_sequences(batch):
    """
    Custom collate function to handle variable-length sequences.

    Returns:
        sequences: List of tensors (varying lengths)
        labels: Tensor of labels
        lengths: Tensor of sequence lengths
    """
    sequences = []
    labels = []
    lengths = []

    for seq, label in batch:
        sequences.append(seq)
        labels.append(label)
        lengths.append(len(seq))

    labels = torch.tensor(labels, dtype=torch.long)
    lengths = torch.tensor(lengths, dtype=torch.long)

    return sequences, labels, lengths


def collate_sequences_padded(batch):
    """
    Collate function that pads sequences to same length.

    Returns:
        sequences: Padded tensor (batch, max_length, H, W) or (batch, max_length, C, H, W)
        lengths: Tensor (batch,)
        labels: Tensor (batch,)
    """
    sequences = []
    labels = []
    lengths = []

    for seq, label in batch:
        sequences.append(seq)
        labels.append(label)
        lengths.append(len(seq))

    # Pad sequences to same length
    sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=0)

    labels = torch.tensor(labels, dtype=torch.long)
    lengths = torch.tensor(lengths, dtype=torch.long)

    return sequences_padded, lengths, labels


# =============================================================================
# ViT Dataset (for Vision Transformer training)
# =============================================================================


class ViTSoliDataset(SoliHDF5DatasetOriginal):
    def __init__(
        self, dataset_dir, config_file, split="train", seq_length=5, img_size=224
    ):
        super().__init__(dataset_dir, config_file, split)
        self.seq_length = seq_length
        self.ssl_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )
        self.resize = transforms.Resize((img_size, img_size))

    def __getitem__(self, idx):
        sequences, label = super().__getitem__(idx)

        frames = []
        for seq in sequences[: self.seq_length]:
            transformed = self.ssl_transform(seq)[:3, :, :]
            frames.append(transformed)

        if len(frames) < self.seq_length:
            frames += [torch.zeros_like(frames[0])] * (self.seq_length - len(frames))

        concatenated = torch.cat(frames, dim=2)
        view1 = self.resize(concatenated)
        concatenated2 = torch.cat(
            [self.ssl_transform(seq)[:3, :, :] for seq in sequences[: self.seq_length]],
            dim=2,
        )
        view2 = self.resize(concatenated2)

        return view1, view2, label


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Testing Dataset Classes")
    print("=" * 70)

    # Test TinyRadarDataset if processed data exists
    processed_5g = Path("./processed/5G")
    if processed_5g.exists():
        print("\n Testing TinyRadarDataset (5G)...")
        try:
            ds = TinyRadarDataset("./processed/5G")
            seq, label = ds[0]
            print(f"   Samples: {len(ds)}")
            print(f"   Sequence shape: {seq.shape}")
            print(f"   Label: {label}")
            print(f"   Value range: [{seq.min():.2f}, {seq.max():.2f}]")
        except Exception as e:
            print(f"   Error: {e}")

    # Test SoliHD5Dataset if SOLI data exists
    soli_dir = Path("./SOLI")
    if soli_dir.exists():
        print("\n Testing SoliHD5Dataset...")
        try:
            ds = SoliHD5Dataset("./SOLI")
            seq, label = ds[0]
            print(f"   Samples: {len(ds)}")
            print(f"   Sequence shape: {seq.shape}")
            print(f"   Label: {label}")
        except Exception as e:
            print(f"   Error: {e}")

    print("\n Tests complete!")
