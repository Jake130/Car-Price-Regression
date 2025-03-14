"""
Training and Development Set Evaluation for Neural Networks

by Jake Kolster, Alex João Peterson Santos
March 7, 2025
for CS 453 Project

Usage:
python3 train.py <model dir> <hyperparameter 1...n>

The <model dir> is a relative path containing model_definition.py,
a file which defines hyperparameters and the model itself.

Results from the model are appended to <model dir>/records.csv.
Weights are saved to <model dir>/<model>_weights.pth.
"""

# imports
import os
import sys
import torch
import csv
import math
import torch.nn.functional as F

# constants
# how often to save validation performance and model weights
EPOCHS_CHECKPOINT = 2
# epochs until training stops
MAX_EPOCHS = 50
# margins (dollar amount) to consider a prediction correct
CORRECT_MARGINS = [0, 10, 100, 1000]
# margin to use for deciding whether to save state dict
SAVE_STATE_DICT_MARGIN = 10
# accuracy (within that margin) to save state dict
SAVE_STATE_DICT_ACCURACY = 0.99

assert MAX_EPOCHS % EPOCHS_CHECKPOINT == 0
assert SAVE_STATE_DICT_MARGIN in CORRECT_MARGINS

# check for arguments (needed for resolving model_definition.py)
if len(sys.argv) <= 1:
    raise ValueError("train.py: No model directory provided")

# resolve importing model_definition.py from subdir
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
model_dir = f"{project_root}/models/{sys.argv[1]}"
if not os.path.isdir(model_dir):
    raise FileNotFoundError(f"train.py: Invalid model directory provided: {model_dir}")
sys.path.insert(0, model_dir)
try:
    import model_definition
except ImportError:
    raise ImportError(f"train.py: Could not import model_definition from {model_dir}")


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

    if type == "label" and not tensor.dtype == torch.float32: # actually int might be more appropriate for price
        print(tensor)
        raise ValueError(f"Found label tensor with invalid data type {tensor.dtype}, expected torch.float32.")

    if torch.isnan(tensor).any():
        print(tensor)
        raise ValueError(f"Found tensor which contains NaN values.")

    if torch.isinf(tensor).any():
        print(tensor)
        raise ValueError(f"Found tensor which contains INF values.")


def train(model, dataloader, loss_fn, optimizer, device, scheduler=None, gradient_clipping_norm=0):
    """
    Define the training process for a single epoch.
    """
    size = len(dataloader.dataset)
    model.train()
    if dataloader.dataset.type != "train":
        raise ValueError(f"train.py: Dataloader for training is not of type 'train'")
    num_batches = len(dataloader)
    interval = math.ceil(num_batches / 10)
    total_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        # move to GPU
        X = X.to(device)
        y = y.to(device)

        # check for nan, inf, etc.
        check_tensor(X, "input")
        check_tensor(y, "label")

        optimizer.zero_grad()

        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()

        # gradient clipping (if specified)
        if gradient_clipping_norm != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clipping_norm)

        optimizer.step()

        loss = loss.item()
        total_loss += loss

        if batch == num_batches - 1:
            print(f"loss: {loss:>7f}  [{size:>5d}/{size:>5d}]")
        elif batch >= 1 and batch % interval == 0:
            current = batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    # apply learning rate schedule if available
    if scheduler is not None:
        scheduler.step()
        print(f"Updated LR: {scheduler.get_last_lr()}")

    result = {
        "train total loss": total_loss,
        "train avg batch loss": total_loss / num_batches
    }
    print(", ".join([f"{key}: {val}" for key, val in result.items()]))
    return result


def evaluate(model, dataloader, loss_fn, device):
    """
    Evaluate the model on the validation set.
    """
    model.eval()
    size = len(dataloader.dataset)
    if dataloader.dataset.type != "val":
        raise ValueError(f"train.py: Dataloader for evaluation is not of type 'val'")
    num_batches = len(dataloader)
    correct_within_margin = {margin: 0 for margin in CORRECT_MARGINS}
    total_test_loss = 0
    total_mse = 0
    total_mae = 0
    with torch.no_grad():
        for X, y in dataloader:
            # move to GPU
            X = X.to(device)
            y = y.to(device)

            # check for nan, inf, etc. in input and label tensors
            check_tensor(X, "input")
            check_tensor(y, "label")

            # make prediction
            pred = model(X)

            # performance metrics
            total_test_loss += loss_fn(pred, y).item()
            total_mse += F.mse_loss(pred, y).item()
            total_mae += F.l1_loss(pred, y).item()
            for margin in CORRECT_MARGINS:
                correct_within_margin[margin] += (torch.abs(pred - y) <= margin).type(torch.float).sum().item()

    result = {
        "val total loss": total_test_loss,
        "val avg batch loss": total_test_loss / num_batches,
        "val avg batch mse": total_mse / num_batches,
        "val avg batch mae": total_mae / num_batches,
    }
    for margin in CORRECT_MARGINS:
        result[f"val correct margin{margin}"] = correct_within_margin[margin]
        result[f"val accuracy margin{margin}"] = correct_within_margin[margin] / size
    for key, val in result.items():
        print(f"{key}: {val}")
    return result


def save_record(args, epochs, result):
    """
    Save the model's performance on the development set to the records CSV file.
    """
    record_path = f"{model_dir}/{args.generic_name}_records.csv"
    file_exists = os.path.isfile(record_path)

    # the CSV row will contain hyperparameters, epochs, and results
    dict_to_write = args.hyperparameter_dict().copy()
    dict_to_write.update({"Epochs": epochs})
    dict_to_write.update(result)

    # check that record file exists or create it
    with open(record_path, "a") as f:
        writer = csv.DictWriter(f, fieldnames=dict_to_write.keys())
        # write header if file is empty
        if not file_exists:
            writer.writeheader()
        # write record for this model
        writer.writerow(dict_to_write)


def save_curves(args, epoch, result):
    """
    Save the avg. training loss, avg. test loss, and accuracy of this epoch
    to a CSV file to help create a graph of the model's loss curves.
    """
    metrics_path = f"{model_dir}/curves/{args.name}_curves.csv"
    file_exists = os.path.isfile(metrics_path)

    # the CSV row will contain avg. training loss, avg. test loss, and accuracy
    dict_to_write = {"Epoch": epoch}
    dict_to_write.update(result)

    # check that the metrics file exists or create it
    with open(metrics_path, "a") as f:
        writer = csv.DictWriter(f, fieldnames=dict_to_write.keys())
        # write header if file is empty
        if not file_exists:
            writer.writeheader()
        # write metrics for this epoch
        writer.writerow(dict_to_write)


def save_state_dict(model, args, epoch):
    """
    Save the state dict of the model to the given model directory
    """
    state_dict_path = f"{model_dir}/saves/{args.name}_ep{epoch:03}_weights.pth"
    torch.save(model.state_dict(), state_dict_path)
    print(f"Saved weights state dict to {state_dict_path}")


def main():
    print("Getting device")
    device = select_device()
    if device != "cuda":
        raise RuntimeError("train.py: CUDA device was not selected")

    print("Device information")
    show_cuda_info()

    print("Creating arguments object")
    args = model_definition.Argument(sys.argv[2:])
    print(args)
    print(f"Running for {MAX_EPOCHS} epochs. Will write to records CSV and state dict every {EPOCHS_CHECKPOINT} epochs")

    print("Creating model")
    model = model_definition.create_model(args).to(device)

    print("Creating optimizer")
    optimizer = model_definition.create_optimizer(args, model)

    print("Creating loss function")
    loss_fn = model_definition.create_loss_function(args)

    # check if args has lr_schedule
    if hasattr(args, "lr_schedule") and args.lr_schedule > 0:
        print("Creating learning rate scheduler")
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_schedule)
    else:
        print("Not using learning rate schedule")
        scheduler = None
    
    if hasattr(args, "gradient_clipping_norm") and args.gradient_clipping_norm > 0:
        gradient_clipping_norm = args.gradient_clipping_norm
        print("Using gradient clipping norm")
    else:
        gradient_clipping_norm = 0
        print("Not using gradient clipping norm")

    print("Creating train and dev dataloaders")
    train_dataloader = model_definition.create_dataloader(args, "train")
    val_dataloader = model_definition.create_dataloader(args, "val")

    print("Beginning training")
    for t in range(MAX_EPOCHS):
        epoch = t + 1
        print(f"\nEpoch {epoch}")

        # one round of training and backpropagation
        result = {}
        train_result = train(model, train_dataloader, loss_fn, optimizer, device, scheduler, gradient_clipping_norm)
        result.update(train_result)

        # one round of evaluation on validation set
        print("Evaluation on val set:")
        val_result = evaluate(model, val_dataloader, loss_fn, device)
        result.update(val_result)

        # save records
        save_curves(args, epoch, result)

        # conditionally save state dict for good-performing models
        if epoch % EPOCHS_CHECKPOINT == 0:
            print("Recording performance and saving state dict")
            save_record(args, epoch, result)
            if result[f"val accuracy margin{SAVE_STATE_DICT_MARGIN}"] < SAVE_STATE_DICT_ACCURACY:
                print("Skipping state dict save due to low accuracy")
                continue
            save_state_dict(model, args, epoch)

    print("Finished training")
    exit(0)


if __name__ == "__main__":
    main()
