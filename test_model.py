import torch
from torch.utils.data import DataLoader
import glob
import os
import sys

# Add current directory to path to ensure imports work
sys.path.append(os.getcwd())

from train.model import SimpleMLP
from train.dataset import BitcoinDataset
from train.utils import calculate_metrics

def test():
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load Model
    model_path = "best_model.pth"
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found!")
        return

    print(f"Loading model from {model_path}...")
    model = SimpleMLP().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 3. Load Test Data
    # Try to find a file from 2018 or later if possible, otherwise use what's available
    # We look for any parquet file in training_data
    search_pattern = r"c:\Users\ozkan\Desktop\aggTrades\training_data\yil=2018\ay=01\data.parquet"
    files = glob.glob(search_pattern)
    
    if not files:
        print("No data files found in training_data!")
        return
        
    # Pick the last file (likely most recent data)
    test_file = files[-1]
    print(f"Testing on file: {test_file}")

    dataset = BitcoinDataset([test_file])
    if len(dataset) == 0:
        print("Dataset is empty.")
        return

    loader = DataLoader(dataset, batch_size=4096, shuffle=False)

    # 4. Inference
    all_preds = []
    all_targets = []

    print("Running inference...")
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            all_preds.append(outputs)
            all_targets.append(targets)

    # 5. Calculate Metrics
    if not all_preds:
        print("No predictions made.")
        return

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    
    metrics = calculate_metrics(all_preds, all_targets)

    print("\n=== Test Results ===")
    print(f"Precision (Wait/Buy/Sell): {metrics['precision']}")
    print(f"Recall    (Wait/Buy/Sell): {metrics['recall']}")
    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
    
    # Calculate Accuracy manually
    cm = metrics['confusion_matrix']
    accuracy = cm.trace() / cm.sum()
    print(f"\nOverall Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    test()
