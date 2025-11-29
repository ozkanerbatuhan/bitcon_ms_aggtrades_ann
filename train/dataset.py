import torch
from torch.utils.data import Dataset
import polars as pl
import numpy as np
import gc

class BitcoinDataset(Dataset):
    def __init__(self, file_list):
        self.files = file_list
        self.data = None
        self.targets = None
        self._load_data()

    def _load_data(self):
        dfs = []
        for f in self.files:
            try:
                df = pl.read_parquet(f)
                if "datetime" in df.columns:
                    df = df.drop("datetime")
                dfs.append(df)
            except Exception as e:
                print(f"Error loading {f}: {e}")
        
        if not dfs:
            self.data = torch.empty(0)
            self.targets = torch.empty(0)
            return

        full_df = pl.concat(dfs)
        
        target_col = "target"
        feature_cols = [c for c in full_df.columns if c != target_col]
        
        # 1. Convert to Numpy
        X = full_df.select(feature_cols).to_numpy()
        y = full_df.select(target_col).to_numpy().flatten()
        
        # 2. NaN / Inf Check & Cleaning (CRITICAL STEP)
        # Bazen log return sonsuz (inf) dönebilir. Bunları temizlemezsek model patlar.
        X = np.nan_to_num(X, nan=0.0, posinf=5.0, neginf=-5.0)
        
        # 3. Aggressive Clipping
        # Neural Network'ler -1 ile 1 arasını sever. 
        # -100/+100 çok fazlaydı, BatchNorm'u bozuyor olabilir.
        # -5/+5 sigma kuralına göre yeterlidir.
        X = np.clip(X, -5.0, 5.0)
        
        self.data = torch.tensor(X, dtype=torch.float32)
        self.targets = torch.tensor(y, dtype=torch.long)
        
        del full_df, dfs, X, y
        gc.collect()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]