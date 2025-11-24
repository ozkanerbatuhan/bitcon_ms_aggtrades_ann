import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, confusion_matrix

def calculate_metrics(outputs, targets):
    """
    Calculates Precision, Recall and Confusion Matrix.
    Args:
        outputs: Raw model outputs (logits) [Batch, Classes]
        targets: True labels [Batch]
    Returns:
        dict of metrics
    """
    # Get predictions
    _, preds = torch.max(outputs, 1)
    
    # Move to CPU for sklearn
    preds = preds.cpu().numpy()
    targets = targets.cpu().numpy()
    
    # Calculate metrics
    # average='weighted' or None to see per-class?
    # User cares about Class 1 (Buy) and Class 2 (Sell). Class 0 is Wait.
    # Let's get per-class metrics.
    
    precision = precision_score(targets, preds, average=None, zero_division=0)
    recall = recall_score(targets, preds, average=None, zero_division=0)
    cm = confusion_matrix(targets, preds, labels=[0, 1, 2])
    
    return {
        "precision": precision, # Array [P_0, P_1, P_2]
        "recall": recall,       # Array [R_0, R_1, R_2]
        "confusion_matrix": cm
    }

def export_weights_to_txt(model, path):
    """
    Exports model weights and biases to a flat text file for FPGA.
    Format: One value per line.
    Order: Layer1_Weight, Layer1_Bias, Layer2_Weight, ...
    """
    print(f"Exporting weights to {path}...")
    with open(path, "w") as f:
        for name, param in model.named_parameters():
            # Flatten the parameter tensor
            data = param.data.cpu().numpy().flatten()
            
            # Write header (optional, for debugging)
            f.write(f"# {name} shape={param.shape}\n")
            
            # Write values
            for val in data:
                f.write(f"{val:.8f}\n")
    print("Export complete.")
