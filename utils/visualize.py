"""
Visualization Script for Training Loss, Development Loss, and Accuracy

by Alex JPS
2024-06-03
CS 472

Usage:
python3 visualize.py <model curves CSV file>
"""

# imports
import sys
import pandas as pd
import matplotlib.pyplot as plt
import re

# constants
MAX_EPOCHS = 20

def generate_subtitle(filename):
    # extract filename from path and remove extension
    filename = filename.split('/')[-1]

    # First extract the model name
    model_name_match = re.match(r'(?P<model_name>.+?)(?=_[0-9])_', filename)
    if not model_name_match:
        raise ValueError(f"Could not extract model name from filename: {filename}")
    model_name = model_name_match.group("model_name")
    
    # Extract hyperparameters
    if model_name == "basic_nn1":
        # Pattern: {}_{}x{}_bs{}_lr{}_sg{}_lk{}_curves.csv
        match = re.match((
            r'(?P<model_name>.+?)(?=_[0-9])_' # model name
            r'(?P<d>\d+)x(?P<w>\d+)_' # depth and width
            r'bs(?P<bs>\d+)_' # batch size
            r'lr(?P<lr>[\d\.]+)_' # learning rate
            r'sg(?P<sg>[\d\.]+)_' # SGD momentum
            r'lk(?P<lk>[\d\.]+)_curves\.csv' # leaky relu value
        ), filename)
    else:
        # Pattern: {}_{}x{}_bs{}_lr{}_sg{}_rl{}_curves.csv
        match = re.match((
            r'(?P<model_name>.+?)(?=_[0-9])_' # model name
            r'(?P<d>\d+)x(?P<w>\d+)_' # depth and width
            r'bs(?P<bs>\d+)_' # batch size
            r'lr(?P<lr>[\d\.]+)_' # learning rate
            r'sg(?P<sg>[\d\.]+)_' # SGD momentum
            r'rl(?P<rl>[\d\.]+)_curves\.csv' # use ReLU
        ), filename)

    if not match:
        raise ValueError(f"Could not extract hyperparameters from filename: {filename}")
    hp = match.groupdict()

    # Generate subtitle from extracted hyperparameters
    subtitle = f"{hp['d']}x{hp['w']}, {hp['bs']}-batches, learn rate {hp['lr']}, "
    subtitle += "AdamW, " if hp['sg'] == "-1" else f"SGD {hp['sg']}, "
    if hp['model_name'] == "basic_nn0":
        subtitle += "ReLU" if 'rl' in hp and hp['rl'] == "1" else "Sigmoid"
    if hp['model_name'] == "basic_nn1":
        subtitle += "Leaky ReLU" if 'lk' in hp and hp['lk'] == "1" else "ReLU"

    return subtitle


def plot_loss(filepath):
    # Read the CSV file
    data = pd.read_csv(filepath)
    plot(
        suptitle="Average training and validation loss per batch",
        title=generate_subtitle(filepath),
        epochs=data['Epoch'],
        black_line=data['val avg batch loss'],
        grey_line=data['train avg batch loss'],
        y_label='Loss',
        black_label='validation',
        grey_label='training',
        y_lim=None
    )

def plot_accuracy(filepath, margin):
    data = pd.read_csv(filepath)
    plot(
        suptitle=f"Proportion validation set predictions within ${margin}",
        title=generate_subtitle(filepath),
        epochs=data['Epoch'],
        black_line=data[f'val accuracy margin{margin}'],
        grey_line=None,
        y_label='Accuracy',
        black_label=None,
        grey_label=None,
        y_lim=(0, 1)
    )


def plot(suptitle, title, epochs, black_line, grey_line, y_label, black_label, grey_label, y_lim = None):
    # titles and spacing
    plt.suptitle(suptitle, fontsize=18, ha='center')
    plt.title(title, fontsize=14, pad=28, ha='center')
    plt.subplots_adjust(top=0.80)

    # limit number of epochs shown
    plt.xlim(0, MAX_EPOCHS)

    # styling
    if y_lim:
        plt.ylim(y_lim)
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    plt.xticks(range(0, MAX_EPOCHS + 1, 2) if MAX_EPOCHS < 20 else range(0, MAX_EPOCHS + 1, 5), fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, color='#dddddd', linewidth=1.5)
    ax = plt.gca()
    ax.yaxis.get_offset_text().set_fontsize(12)
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)

    # Plot training and development loss
    if grey_line is not None:
        plt.plot(epochs, grey_line, label=grey_label, color='grey', linewidth=4)
    if black_line is not None:
        plt.plot(epochs, black_line, label=black_label, color='black', linewidth=4)
    plt.legend(loc="upper right", fontsize=14)
    plt.show()


def bad_usage():
    print("Accuracy: python script.py <csv_file_path> -m <margin>")
    print("Loss: python script.py <csv_file_path> -l")
    exit(1)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        bad_usage()
    if sys.argv[2] == "-m":
        try:
            int(sys.argv[3])
        except:
            bad_usage()
        plot_accuracy(sys.argv[1], sys.argv[3])
        exit(0)
    if sys.argv[2] == "-l":
        plot_loss(sys.argv[1])
        exit(0)
    bad_usage()
