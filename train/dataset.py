import torch
from torch.utils.data import Dataset
import polars as pl
import os
import glob
import gc
import numpy as np

class BitcoinDataset(Dataset):
    def __init__(self, file_list):
        """
        Args:
            file_list (list): List of paths to parquet files to load.
        """
        self.files = file_list
        self.data = None
        self.targets = None
        self._load_data()

    def _load_data(self):
        # print(f"Loading {len(self.files)} files into RAM...")
        dfs = []
        for f in self.files:
            try:
                # Read parquet
                df = pl.read_parquet(f)
                
                # Drop datetime if exists
                if "datetime" in df.columns:
                    df = df.drop("datetime")
                
                dfs.append(df)
            except Exception as e:
                print(f"Error loading {f}: {e}")
        
        if not dfs:
            print("No data loaded!")
            self.data = torch.empty(0)
            self.targets = torch.empty(0)
            return

        full_df = pl.concat(dfs)
        # print(f"Chunk rows: {full_df.height}")
        
        target_col = "target"
        feature_cols = [c for c in full_df.columns if c != target_col]
        
        # Features
        X = full_df.select(feature_cols).to_numpy()
        
        # Clamp outliers to prevent NaN loss
        # log_ret_norm (indices 1, 5, 9...): Clamp to [-20, 20]
        # vol_norm (indices 2, 6, 10...): Clamp to [0, 100]
        # RSI and OFI are already bounded.
        
        # Actually, simpler to clamp everything? No, RSI is 0-1.
        # Let's just clamp the whole tensor to [-100, 100] to be safe.
        X = np.clip(X, -100.0, 100.0)
        
        self.data = torch.tensor(X, dtype=torch.float32)
        
        # Targets
        y = full_df.select(target_col).to_numpy().flatten()
        self.targets = torch.tensor(y, dtype=torch.long)
        
        # Free memory
        del full_df
        del dfs
        gc.collect()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]
