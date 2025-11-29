import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
import glob
import gc
from dataset import BitcoinDataset
from model import SimpleMLP
from utils import calculate_metrics, export_weights_to_txt

# Hata ayıklama modu: NaN nerede oluşuyor gösterir (Eğitimi biraz yavaşlatır ama sorunu çözer)
torch.autograd.set_detect_anomaly(True)

def get_all_files(data_dir, start_year, end_year):
    files = []
    for year in range(start_year, end_year + 1):
        year_path = os.path.join(data_dir, f"yil={year}")
        if os.path.exists(year_path):
            month_dirs = sorted(glob.glob(os.path.join(year_path, "ay=*")))
            for md in month_dirs:
                f = os.path.join(md, "data.parquet")
                if os.path.exists(f):
                    files.append(f)
    return sorted(files)

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Files Setup
    all_files = get_all_files(r"c:\Users\ozkan\Desktop\aggTrades\training_data", args.start_year, args.end_year)
    if not all_files:
        print("No files found!")
        return

    val_count = max(1, int(len(all_files) * args.val_split))
    train_files = all_files[:-val_count]
    val_files = all_files[-val_count:]
    
    # Model Setup
    model = SimpleMLP().to(device)
    
    # Weights: Sınıf dengesizliği için
    weights = torch.tensor([1.0, 5.0, 5.0]).to(device) # 25 biraz agresifti, NaN riskini artırabilir. 5-10 arası daha güvenli.
    criterion = nn.CrossEntropyLoss(weight=weights)
    
    # LR Düşürüldü: 0.001 -> 0.0001 (Daha stabil öğrenme için)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_val_loss = float('inf')

    print("Starting Training...")
    
    for epoch in range(args.epochs):
        print(f"\n=== Epoch {epoch+1}/{args.epochs} ===")
        
        # --- TRAIN ---
        model.train()
        total_train_loss = 0.0
        train_batches = 0
        
        chunk_size = args.chunk_size
        for i in range(0, len(train_files), chunk_size):
            chunk_files = train_files[i : i + chunk_size]
            
            # Dataset Load
            try:
                dataset = BitcoinDataset(chunk_files)
            except Exception as e:
                print(f"Skipping corrupt chunk: {e}")
                continue
                
            if len(dataset) == 0: continue
            
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
            # % progress 
            progress_i =  0
            for inputs, targets in loader:
                inputs, targets = inputs.to(device), targets.to(device)
                progress_i += 1
                print(f"Progress: {progress_i}/{len(loader)}")
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                # Check for NaN loss explicitly
                if torch.isnan(loss):
                    print("PANIC: Loss is NaN! Skipping batch.")
                    continue
                
                loss.backward()
                
                # Gradient Clipping (Çok önemli)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                
                optimizer.step()
                
                total_train_loss += loss.item()
                train_batches += 1
            
            del dataset, loader
            gc.collect()
            
        avg_train_loss = total_train_loss / max(1, train_batches)
        print(f"Train Loss: {avg_train_loss:.4f}")

        # --- VALIDATION ---
        model.eval()
        total_val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for i in range(0, len(val_files), chunk_size):
                chunk_files = val_files[i : i + chunk_size]
                dataset = BitcoinDataset(chunk_files)
                if len(dataset) == 0: continue
                
                loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
                
                for inputs, targets in loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    total_val_loss += loss.item()
                    val_batches += 1
                
                del dataset, loader
                gc.collect()

        avg_val_loss = total_val_loss / max(1, val_batches)
        print(f"Val Loss: {avg_val_loss:.4f}")
        
        # Save Logic
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("Saved Best Model.")

    # Export
    print("Exporting weights...")
    if os.path.exists("best_model.pth"):
        model.load_state_dict(torch.load("best_model.pth"))
    
    export_weights_to_txt(model, "fpga_weights.txt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_year", type=int, default=2017)
    parser.add_argument("--end_year", type=int, default=2019)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--chunk_size", type=int, default=1)
    
    args = parser.parse_args()
    train(args)