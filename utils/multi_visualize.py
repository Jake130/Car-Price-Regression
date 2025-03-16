"""
Visualization Script for Training Loss, Development Loss, and Accuracy

by Alex João Peterson Santos
Updated March 16, 2025
CS 472
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import re
import matplotlib.colors as mc
import colorsys

# constants
MAX_EPOCHS = 25

def lighten_color(color, amount=0.5):
    """
    Lightens the given color.
    Input can be a matplotlib color name, hex string, or RGB tuple.
    The 'amount' controls how much lighter the color should be.
    """
    try:
        c = mc.cnames[color]
    except KeyError:
        c = color
    c = mc.to_rgb(c)
    h, l, s = colorsys.rgb_to_hls(*c)
    new_l = min(1, l + (1 - l) * amount)
    return colorsys.hls_to_rgb(h, new_l, s)

def extract_hp(filename):
    """
    Extracts hyperparameters from the CSV filename.
    Currently only supports basic_nn2 filenames.
    Expected pattern: 
      basic_nn2_<d>x<w>_bs<bs>_lr<lr>_sg<sg>_af<af>_gc<gc>_ls<ls>_curves.csv
    """
    filename = filename.split('/')[-1]
    model_name_match = re.match(r'(?P<model_name>.+?)(?=_[0-9])_', filename)
    if not model_name_match:
        raise ValueError(f"Could not extract model name from filename: {filename}")
    model_name = model_name_match.group("model_name")
    if model_name != "basic_nn2":
        raise ValueError("Currently only supports basic_nn2")
    pattern = (
        r'(?P<model_name>.+?)(?=_[0-9])_'  # model name
        r'(?P<d>\d+)x(?P<w>\d+)_'          # depth and width
        r'bs(?P<bs>\d+)_'                  # batch size
        r'lr(?P<lr>[\d\.]+)_'              # learning rate
        r'sg(?P<sg>-?[\d\.]+)_'            # SGD momentum (including negatives)
        r'af(?P<af>[\d\.]+)_'              # activation function
        r'gc(?P<gc>[\d\.]+)_'              # gradient clipping norm
        r'ls(?P<ls>[\d\.]+)_curves\.csv'    # learning rate schedule
    )
    match = re.match(pattern, filename)
    if not match:
        raise ValueError(f"Could not extract hyperparameters from filename: {filename}")
    hp = match.groupdict()
    return hp

def generate_subtitle(filename, exclude_key):
    """
    Generates a subtitle string from hyperparameters extracted from the CSV filename.
    For the hyperparameter given by exclude_key, the value is replaced with the literal word 'variable'.
    """
    hp = extract_hp(filename)
    hp[exclude_key] = "?"
    af = {
        "0": "Sigmoid",
        "1": "ReLU",
        "2": "Leaky ReLU",
        "?": "?"
    }[hp['af']]
    try:
        hp['gc'] = int(float(hp['gc']))
    except:
        pass
    # comparing_on = {
    #     "d": "depth",
    #     "w": "width",
    #     "bs": "batch size",
    #     "lr": "learning rate",
    #     "sg": "momentum",
    #     "af": "activation function",
    #     "gc": "gradient clipping",
    #     "ls": "learning rate schedule"
    # }
    subtitle = f"""
{hp['d']}x{hp['w']}, {hp['bs']}-batches, learn rate {hp['lr']} {"(no sched.)" if hp['ls'] == "0" else "on " + hp['ls'] + " sched."},
{"AdamW" if hp['sg'] == "-1" else f"SGD momentum {hp['sg']}"}, {af}, grad. clipping {hp['gc']}
"""
    return subtitle

def adjust_label(exclude_key, val):
    if exclude_key == "sg":
        return "AdamW" if val == "-1" else val
    if exclude_key == "af":
        return {
            "0": "Sigmoid",
            "1": "ReLU",
            "2": "Leaky ReLU"
        }[val]
    if exclude_key == "ls":
        return "No schedule" if val == "0" else val
    if exclude_key == "gc":
        try:
            val = int(float(val))
        except:
            pass
        return "No clipping" if val == 0 else val
    return val

def plot_multi_loss(files, exclude_key):
    plt.figure()
    plt.suptitle("Average Training Loss per Batch", fontsize=18)
    # Display the generated subtitle using the first file's hp dict
    subtitle = generate_subtitle(files[0], exclude_key)
    plt.title(subtitle, fontsize=14, pad=0, ha='center')
    plt.subplots_adjust(top=0.75)
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.xlim(0, MAX_EPOCHS)
    plt.xticks(range(0, MAX_EPOCHS + 1, 2) if MAX_EPOCHS < 20 else range(0, MAX_EPOCHS + 1, 5), fontsize=12)
    plt.grid(True, color='#dddddd', linewidth=1.5)
    plt.yticks(fontsize=12)
    ax = plt.gca()
    ax.yaxis.get_offset_text().set_fontsize(12)
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    
    colors = ['darkred', '#FF7518', 'navy', 'darkgreen']
    for i, file in enumerate(files):
        data = pd.read_csv(file)
        epochs = data['Epoch']
        if 'val avg batch loss' not in data.columns or 'train avg batch loss' not in data.columns:
            raise ValueError(f"Expected loss columns not found in {file}")
        val_loss = data['val avg batch loss']
        train_loss = data['train avg batch loss']
        # Use only the value of the differing hyperparameter as the label.
        hp = extract_hp(file)
        var_label = adjust_label(exclude_key, hp[exclude_key])
        full_color = colors[i % len(colors)]
        muted_color = lighten_color(full_color, amount=0.5)
        plt.plot(epochs, train_loss, label=f"{var_label}", color=full_color, linewidth=4)
        # plt.plot(epochs, train_loss, label=None, color=muted_color, linewidth=4)
    plt.legend(loc="upper right", fontsize=12)
    plt.show()

def plot_multi_accuracy(files, margin, exclude_key):
    plt.figure()
    plt.suptitle(f"Validation Accuracy within ${margin}", fontsize=18)
    subtitle = generate_subtitle(files[0], exclude_key)
    plt.title(subtitle, fontsize=14, pad=0, ha='center')
    plt.subplots_adjust(top=0.78)
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.xlim(0, MAX_EPOCHS)
    plt.ylim(0, 1)
    plt.xticks(range(0, MAX_EPOCHS + 1, 2) if MAX_EPOCHS < 20 else range(0, MAX_EPOCHS + 1, 5), fontsize=12)
    plt.grid(True, color='#dddddd', linewidth=1.5)
    plt.yticks(fontsize=12)
    ax = plt.gca()
    ax.yaxis.get_offset_text().set_fontsize(12)
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    
    colors = ['blue', 'orange', 'green', 'red']
    for i, file in enumerate(files):
        data = pd.read_csv(file)
        epochs = data['Epoch']
        colname = f'val accuracy margin{margin}'
        if colname not in data.columns:
            raise ValueError(f"Column {colname} not found in {file}")
        accuracy = data[colname]
        hp = extract_hp(file)
        var_label = adjust_label(exclude_key, var_label)
        full_color = colors[i % len(colors)]
        plt.plot(epochs, accuracy, label=var_label, color=full_color, linewidth=4)
    plt.legend(loc="upper right", fontsize=12)
    plt.show()

def plot_multi_error(files, exclude_key):
    plt.figure()
    plt.suptitle("Mean Absolute Error (MAE) on Validation Set", fontsize=18)
    subtitle = generate_subtitle(files[0], exclude_key)
    plt.title(subtitle, fontsize=14, pad=0, ha='center')
    plt.subplots_adjust(top=0.78)
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("MAE", fontsize=14)
    plt.xlim(0, MAX_EPOCHS)
    plt.xticks(range(0, MAX_EPOCHS + 1, 2) if MAX_EPOCHS < 20 else range(0, MAX_EPOCHS + 1, 5), fontsize=12)
    plt.grid(True, color='#dddddd', linewidth=1.5)
    plt.yticks(fontsize=12)
    ax = plt.gca()
    ax.yaxis.get_offset_text().set_fontsize(12)
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    
    colors = ['darkred', '#FF7518', 'navy', 'darkgreen']
    for i, file in enumerate(files):
        data = pd.read_csv(file)
        epochs = data['Epoch']
        if 'val avg batch mae' not in data.columns:
            raise ValueError(f"Column 'val avg batch mae' not found in {file}")
        mae = data['val avg batch mae']
        hp = extract_hp(file)
        var_label = adjust_label(exclude_key, hp[exclude_key])
        full_color = colors[i % len(colors)]
        plt.plot(epochs, mae, label=var_label, color=full_color, linewidth=4)
    plt.legend(loc="upper right", fontsize=12)
    plt.show()

def bad_usage():
    print("Usage:")
    print("  For Accuracy: python visualize.py <csv_file1> <csv_file2> [<csv_file3> <csv_file4>] -m <margin>")
    print("  For Loss:     python visualize.py <csv_file1> <csv_file2> [<csv_file3> <csv_file4>] -l")
    print("  For MAE:      python visualize.py <csv_file1> <csv_file2> [<csv_file3> <csv_file4>] -e")
    exit(1)

def parse_for_sorting(val):
    """
    Attempt to parse a string as float for numeric sorting.
    If it fails, return the string itself for alphabetical fallback.
    """
    try:
        return float(val)
    except ValueError:
        return val

if __name__ == '__main__':
    # Parse command-line arguments.
    args = sys.argv[1:]
    if not args or len(args) < 3:
        bad_usage()

    mode = None
    files = []
    margin = None

    if "-m" in args:
        flag_index = args.index("-m")
        if flag_index == len(args) - 1:
            bad_usage()
        margin = args[flag_index + 1]
        files = args[:flag_index]
        mode = "accuracy"
    elif "-l" in args:
        flag_index = args.index("-l")
        files = args[:flag_index]
        mode = "loss"
    elif "-e" in args:
        flag_index = args.index("-e")
        files = args[:flag_index]
        mode = "mae"
    else:
        bad_usage()

    # Reject if not between 2 and 4 CSV files.
    if len(files) < 2 or len(files) > 4:
        print("Please provide between 2 and 4 CSV files.")
        exit(1)

    # Determine the hyperparameters that differ across files.
    hp_list = [extract_hp(f) for f in files]
    diff_keys = [k for k in hp_list[0].keys() if not all(hp_list[0][k] == other[k] for other in hp_list)]
    if len(diff_keys) != 1:
        print("Error: Exactly one hyperparameter must differ across files.")
        print(f"Found differing keys: {diff_keys}")
        exit(1)
    exclude_key = diff_keys[0]

    files.sort(key=lambda f: parse_for_sorting(extract_hp(f)[exclude_key]))

    if mode == "accuracy":
        try:
            margin_int = int(margin)
        except:
            bad_usage()
        plot_multi_accuracy(files, margin_int, exclude_key)
    elif mode == "loss":
        plot_multi_loss(files, exclude_key)
    elif mode == "mae":
        plot_multi_error(files, exclude_key)
    else:
        bad_usage()
