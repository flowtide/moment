import torch
from momentfm import MOMENTPipeline
from torch.utils.data import DataLoader, Dataset
from momentfm.utils.data import load_from_tsfile
import numpy as np
import argparse
import sys
import time
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# Define the TS_Dataset class
class TS_Dataset(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        self.data, self.labels, meta_data = load_from_tsfile(file_path, return_meta_data=True)
        print(f"data.shape={self.data.shape}")
        self.n_samples, self.n_channels, self.series_length = self.data.shape
        mean = np.mean(self.data, axis=-1, keepdims=True)
        std = np.std(self.data, axis=-1, keepdims=True)
        std_adj = np.where(std == 0, 1, std)
        self.data = (self.data - mean) / std_adj
        self._length = self.n_samples
        self.n_classes = len(meta_data['class_values'])
        print(f"Dataset Loaded: {file_path} | Samples: {self.n_samples}, Channels: {self.n_channels}, Series Length: {self.series_length}, Classes: {self.n_classes}")
        
    def __len__(self):
        return self._length
    
    def __getitem__(self, idx):
        return self.data[idx], int(self.labels[idx])

# Function to load the model
def load_model(model_path, n_channels, num_classes, device='cuda'):
    """
    Load the MOMENTPipeline model from a saved state dictionary.
    
    Parameters
    ----------
    model_path : str
        Path to the saved .pth file containing the state dictionary.
    n_channels : int
        Number of channels in the time series data.
    num_classes : int
        Number of classes for classification.
    device : str
        Device to load the model onto ('cuda' or 'cpu').
    
    Returns
    -------
    model : MOMENTPipeline
        Loaded model with the saved weights.
    """
    # Initialize the model with the same configuration as during training
    model = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-large",
        model_kwargs={
            'task_name': 'classification',
            'n_channels': n_channels,
            'num_class': num_classes,
            'freeze_encoder': True,
            'freeze_embedder': True,
            'freeze_head': False,
            'enable_gradient_checkpointing': False,
            'reduction': 'mean',
        }
    ).to(device)
    
    model.init()

    # Load the state dictionary from the saved file
    state_dict = torch.load(model_path, map_location=device)
    
    # Debugging: Print shapes of the loaded state_dict
    print(f"Loaded state_dict head.linear.weight shape: {state_dict['head.linear.weight'].shape}")
    print(f"Loaded state_dict head.linear.bias shape: {state_dict['head.linear.bias'].shape}")
    
    # Load the weights into the model
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        print(f"Error loading state_dict: {e}")
        if state_dict['head.linear.weight'].shape[0] != num_classes:
            print(f"Adjusting model head to match saved num_classes={state_dict['head.linear.weight'].shape[0]}")
            model.head.linear = torch.nn.Linear(model.head.linear.in_features, state_dict['head.linear.weight'].shape[0])
            model.load_state_dict(state_dict)
    
    # Move the model to the specified device
    model.to(device)
    
    # Set the model to evaluation mode
    model.eval()
    
    print(f"Model loaded from {model_path}")
    return model

# Example usage
if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Load a MOMENT model and predict on a time series dataset.")
    parser.add_argument("ts_file", nargs='?', type=str, help="Path to the input .ts file")
    parser.add_argument("model_file", nargs='?', type=str, help="Path to the saved model .pth file")

    # Parse the arguments
    args = parser.parse_args()

    # Check if both arguments are provided
    if args.ts_file is None or args.model_file is None:
        parser.print_help()
        sys.exit(1)

    # Parameters
    model_path = args.model_file
    ts_file = args.ts_file
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the dataset to get n_channels and num_classes
    test_dataset = TS_Dataset(ts_file)
    n_channels = test_dataset.n_channels
    num_classes = test_dataset.n_classes
    
    print(f"n_channels={n_channels}, num_classes={num_classes}, Device: {device}")
    
    # Load the model
    loaded_model = load_model(model_path, n_channels, num_classes, device)
    
    # Test the loaded model on the full dataset with statistics
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # Define loss criterion
    criterion = torch.nn.CrossEntropyLoss()
    
    # Evaluate the model
    loaded_model.eval()
    test_loss = 0
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    t1 = time.time()

    with torch.no_grad():
        for batch_x, batch_labels in test_loader:
            batch_x = batch_x.to(device).float()
            batch_labels = batch_labels.to(device)
            output = loaded_model(x_enc=batch_x, reduction='mean')
            
            # Compute loss
            logits = output.logits
            loss = criterion(logits, batch_labels)
            test_loss += loss.item()
            
            # Get predictions and probabilities
            probabilities = torch.softmax(logits, dim=1)  # Convert logits to probabilities
            predictions = logits.argmax(dim=1)
            
            # Print prediction, true label, and probabilities
            probs_str = ", ".join([f"Class {i}: {prob:.3f}" for i, prob in enumerate(probabilities[0].cpu().numpy())])
            print(f"Predicted: {predictions.item()}, Label: {batch_labels.item()}, Probabilities: [{probs_str}]")
            
            total += batch_labels.size(0)
            correct += (predictions == batch_labels).sum().item()
            
            # Save labels and predictions for metrics
            all_labels.extend(batch_labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    # Calculate metrics with zero_division set to 0
    test_loss /= len(test_loader)
    accuracy = 100 * correct / total
    precision = precision_score(all_labels, all_predictions, average="weighted", zero_division=0)
    recall = recall_score(all_labels, all_predictions, average="weighted", zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average="weighted", zero_division=0)
    
    # Calculate confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_predictions)

    # Print metrics
    print(f"\nEvaluation Results:")
    print(f"Num Test Set: {len(test_loader)}")
    print(f"Test Loss: {test_loss:.3f}")
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(f"Confusion Matrix:\n{conf_matrix}")
    print(f"Time taken: {time.time() - t1:.2f} seconds, finished at {time.ctime()}")