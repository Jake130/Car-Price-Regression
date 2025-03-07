"""
Test Set Evaluation for Neural Networks

by Jake Kolster, Alex João Peterson Santos
March 7, 2025
for CS 453 Project

Usage:
python3 test.py <model dir> <state dict path> <hyperparameter 1...n>

The <model dir> is a relative path containing model_definition.py,
a file which defines hyperparameters and the model itself.

Results from the test are sent to <model dir>/<model>_test_results.csv.
"""

# imports
import os
import sys
import torch
import csv
import re

# check for arguments (needed for resolving model_definition.py)
if len(sys.argv) <= 1:
    raise ValueError("test.py: No model directory provided")

# resolve importing model_definition.py from subdir
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
model_dir = f"{project_root}/models/{sys.argv[1]}"
if not os.path.isdir(model_dir):
    raise FileNotFoundError(f"test.py: Invalid model directory provided: {model_dir}")
sys.path.insert(0, model_dir)
try:
    import model_definition
except ImportError:
    raise ImportError(f"test.py: Could not import model_definition from {model_dir}")


def show_cuda_info():
    """
    Print information about the CUDA device(s) in use.
    """
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")


def select_device() -> str:
    """
    Print information about available devices and select one for pytorch.
    """
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"Found {gpu_count} CUDA-enabled devices")
        show_cuda_info()
        device = "cuda"
    elif torch.backends.mps.is_available():
        print("CUDA is not available. Using MPS")
        device = "mps"
    else:
        print("CUDA and MPS are not available. Using CPU")
        device = "cpu"
    return device


def check_tensor(tensor, type):
    assert type in ["input", "label"]

    if type == "input" and not tensor.dtype == torch.float32:
        print(tensor)
        raise ValueError(f"Found input tensor with invalid data type {tensor.dtype}, expected torch.float32.")

    if type == "label" and not tensor.dtype == torch.float32:
        print(tensor)
        raise ValueError(f"Found label tensor with invalid data type {tensor.dtype}, expected torch.float32.")

    if torch.isnan(tensor).any():
        print(tensor)
        raise ValueError(f"Found tensor which contains NaN values.")

    if torch.isinf(tensor).any():
        print(tensor)
        raise ValueError(f"Found tensor which contains INF values.")


def evaluate(model, dataloader, loss_fn, device):
    """
    Evaluate the model on the test set.
    """
    model.eval()
    size = len(dataloader.dataset)
    if dataloader.dataset.type != "test":
        raise ValueError(f"test.py: Dataloader for evaluation is not of type 'test'")
    num_batches = len(dataloader)
    CORRECT_MARGIN = 500
    total_test_loss, correct_within_margin = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            # move to GPU
            X = X.to(device)
            y = y.to(device)

            # check for nan, inf, etc.
            check_tensor(X, "input")
            check_tensor(y, "label")

            # make prediction and run loss function
            pred = model(X)
            total_test_loss += loss_fn(pred, y).item()
            correct_within_margin += (torch.abs(pred - y) <= CORRECT_MARGIN).type(torch.float).sum().item()
    result = {
        "test total loss": total_test_loss,
        "test avg batch loss": total_test_loss / num_batches,
        "test num correct": correct_within_margin,
        "test accuracy": correct_within_margin / size * 100,
    }
    print(", ".join([f"{key}: {val}" for key, val in result.items()]))
    return result


def save_test_result(args, epoch, test_result):
    """
    Save the test loss and accuracy to a test result CSV file.
    """
    test_result_path = f"{model_dir}/{args.name}_ep{epoch:03}_test_result.csv"
    dict_to_write = args.hyperparameter_dict().copy()
    dict_to_write.update({"Epochs": epoch})
    dict_to_write.update(test_result)

    # overwrite existing results file if it exists
    with open(test_result_path, "w") as f:
        writer = csv.DictWriter(f, fieldnames=dict_to_write.keys())
        writer.writeheader()
        writer.writerow(dict_to_write)


def main():
    # check for valid state dict path
    if len(sys.argv) <= 2:
        raise ValueError("test.py: No state dict path provided")
    state_dict_path = sys.argv[2]
    if not os.path.isfile(state_dict_path) or not state_dict_path.endswith(".pth"):
        raise FileNotFoundError(f"test.py: Invalid state dict path provided: {state_dict_path}")

    print("Getting device")
    device = select_device()
    if device != "cuda":
        raise RuntimeError("test.py: CUDA device was not selected")

    print("Device information")
    show_cuda_info()

    print("Creating arguments object")
    args = model_definition.Argument(sys.argv[3:])
    print(args)

    print("Creating model")
    model = model_definition.create_model(args).to(device)

    print("Loading state dict")
    model.load_state_dict(torch.load(state_dict_path))

    print("Creating loss function")
    loss_fn = model_definition.create_loss_function(args)

    print("Creating test dataloader")
    test_dataloader = model_definition.create_dataloader(args, "test")

    print("Evaluation on test set:")
    test_result = evaluate(model, test_dataloader, loss_fn, device)

    print(f"Saving test result")
    # infer epoch from state dict filename
    try:
        match = re.search(r'ep(\d+)_weights.pth', state_dict_path)
        epoch = int(match.group(1))
    except Exception:
        raise ValueError(f"Could not infer epoch from state dict path: {state_dict_path}")
    save_test_result(args, epoch, test_result)

    print("Finished evaluation")
    exit(0)


if __name__ == "__main__":
    main()
