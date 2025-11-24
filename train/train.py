import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
import glob
import gc
import random
from dataset import BitcoinDataset
from model import SimpleMLP
from utils import calculate_metrics, export_weights_to_txt

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
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if device.type == 'cpu':
        raise RuntimeError("CUDA not available! Aborting.")

    # 2. Discovery & Split
    print(f"Discovering files for {args.start_year}-{args.end_year}...")
    all_files = get_all_files(r"c:\Users\ozkan\Desktop\aggTrades\training_data", args.start_year, args.end_year)
    
    if not all_files:
        print("No files found!")
        return

    # Chronological Split
    val_count = max(1, int(len(all_files) * args.val_split))
    train_files = all_files[:-val_count]
    val_files = all_files[-val_count:]
    
    print(f"Total Files: {len(all_files)}")
    print(f"Train Files: {len(train_files)} | Val Files: {len(val_files)}")
    
    # 3. Model Setup
    model = SimpleMLP().to(device)
    weights = torch.tensor([1.0, 25.0, 25.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    best_val_loss = float('inf')
    
    # 4. Training Loop
    print("Starting Chunked Training...")
    
    for epoch in range(args.epochs):
        print(f"\n=== Epoch {epoch+1}/{args.epochs} ===")
        
        # --- TRAIN PHASE ---
        model.train()
        total_train_loss = 0.0
        train_batches = 0
        
        # Shuffle chunks? No, keep chronological for now, or shuffle chunks?
        # Shuffling chunks is better for SGD.
        # But let's keep it simple first.
        
        # Process Train Files in Chunks
        chunk_size = args.chunk_size
        for i in range(0, len(train_files), chunk_size):
            chunk_files = train_files[i : i + chunk_size]
            print(f"  [Train] Processing Chunk {i//chunk_size + 1}/{(len(train_files)-1)//chunk_size + 1} ({len(chunk_files)} files)...")
            
            # Load Chunk
            dataset = BitcoinDataset(chunk_files)
            if len(dataset) == 0: continue
            
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
            
            # Train on Chunk
            for inputs, targets in loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                
                # Gradient Clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                total_train_loss += loss.item()
                train_batches += 1
            
            # Cleanup
            del dataset, loader
            gc.collect()
            
        avg_train_loss = total_train_loss / max(1, train_batches)
        
        # --- VALIDATION PHASE ---
        model.eval()
        total_val_loss = 0.0
        val_batches = 0
        all_preds = []
        all_targets = []
        
        print("  [Val] Validating...")
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
                    
                    # Store for metrics (might be large, be careful)
                    # If Val set is huge, this might OOM.
                    # Let's store on CPU.
                    all_preds.append(outputs.cpu())
                    all_targets.append(targets.cpu())
                
                del dataset, loader
                gc.collect()

        avg_val_loss = total_val_loss / max(1, val_batches)
        
        # Metrics
        if all_preds:
            all_preds = torch.cat(all_preds)
            all_targets = torch.cat(all_targets)
            metrics = calculate_metrics(all_preds, all_targets)
            
            print(f"Summary:")
            print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
            print(f"Precision: {metrics['precision']}")
            print(f"Recall:    {metrics['recall']}")
            print(f"Confusion Matrix:\n{metrics['confusion_matrix']}")
        
        # Save Best (Handle NaN)
        if not (avg_val_loss != avg_val_loss) and avg_val_loss < best_val_loss: # Check for NaN
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("Saved Best Model!")
        
        # Always save last model as fallback
        torch.save(model.state_dict(), "last_model.pth")

    # 5. Export
    print("Exporting weights...")
    if os.path.exists("best_model.pth"):
        model.load_state_dict(torch.load("best_model.pth"))
    else:
        print("Warning: No best model found (NaN loss?), using last model.")
        model.load_state_dict(torch.load("last_model.pth"))
        
    export_weights_to_txt(model, "fpga_weights.txt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_year", type=int, default=2017)
    parser.add_argument("--end_year", type=int, default=2019)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=4096) # Increased batch size for GPU
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--chunk_size", type=int, default=1) # 1 file (month) at a time
    
    args = parser.parse_args()
    train(args)
