"""
PyTorch DataLoader for PFN-Style Intervention Dataset

All data is fixed-size (padded to K=10, scaled by K/k).
No custom collate functions needed!
"""

import pickle
import torch
import numpy as np
from torch.utils.data import Dataset


class InterventionDataset(Dataset):
    """
    PyTorch Dataset for PFN-style encoded intervention data.

    All samples have fixed shape (300, 10) or (100, 10) due to PFN encoding.
    """

    def __init__(self, dataset_path, return_onehot=False):
        """
        Initialize the dataset.

        Args:
            dataset_path: Path to pickled dataset
            return_onehot: If True, return one-hot labels; else scalar labels
        """
        with open(dataset_path, "rb") as f:
            self.data = pickle.load(f)

        self.n_samples = len(self.data["intervention_nodes"])
        self.return_onehot = return_onehot
        self.max_nodes = self.data["observational_data"].shape[2]  # K

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        """
        Get a single sample.

        Returns:
            obs_data: (300, K) observational data (PFN-encoded)
            int_data: (100, K) interventional data (PFN-encoded)
            label: intervention node index (scalar or one-hot)
            n_nodes: actual number of nodes k (before padding)
            mask: (K,) boolean mask (True for real nodes, False for padding)
        """
        # Get PFN-encoded data (already padded and scaled)
        obs_data = self.data["observational_data"][idx]  # (300, K)
        int_data = self.data["interventional_data"][idx]  # (100, K)
        n_nodes = self.data["n_nodes_list"][idx]  # Original k

        # Create mask for valid nodes
        mask = np.zeros(self.max_nodes, dtype=bool)
        mask[:n_nodes] = True

        # Get label in requested format
        if self.return_onehot:
            label = self.data["intervention_labels"][idx]  # (K,) one-hot
            label = torch.FloatTensor(label)
        else:
            label = self.data["intervention_nodes"][idx]  # scalar
            label = torch.LongTensor([label])[0]

        # Convert to tensors
        obs_data = torch.FloatTensor(obs_data)
        int_data = torch.FloatTensor(int_data)
        mask = torch.BoolTensor(mask)
        n_nodes = torch.LongTensor([n_nodes])[0]

        return obs_data, int_data, label, n_nodes, mask
