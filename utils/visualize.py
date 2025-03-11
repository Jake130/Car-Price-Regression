"""
Visualization Script for Training Loss, Development Loss, and Accuracy

by Alex JPS
2024-06-03
CS 472

Usage:
python3 visualize.py <model curves CSV file>
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt

def plot_loss(filepath):
    # Read the CSV file
    data = pd.read_csv(filepath)

    # Extract data
    epochs = data['Epoch']
    accuracy = data['Accuracy']
    training_loss = data['Average training loss']
    dev_loss = data['Average dev loss']

    # Create a figure and axis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # Plot accuracy
    ax1.plot(epochs, accuracy, marker='o', label='Accuracy', color='b')
    ax1.set_title('Accuracy over Epochs')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0, 1)
    ax1.legend()
    ax1.grid(True)

    # Plot training and development loss
    ax2.plot(epochs, training_loss, marker='o', label='Training Loss', color='r')
    ax2.plot(epochs, dev_loss, marker='o', label='Development Loss', color='g')
    ax2.set_title('Training and Development Loss over Epochs')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show plot
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
        plot_margin(sys.argv[1], sys.argv[3])
        exit(0)
    if sys.argv[2] == "-l":
        plot_loss(sys.argv[1])
        exit(0)
    bad_usage()
