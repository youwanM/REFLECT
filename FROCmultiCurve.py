import pickle
import matplotlib.pyplot as plt
import sys
from pathlib import Path

def load_pickled_plot(file_path):
    """Load a pickled Matplotlib Figure or Axes object."""
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def main(pickle_files):
    if len(pickle_files) != 4:
        print("Please provide exactly 4 pickle files.")
        sys.exit(1)

    fig, ax = plt.subplots(figsize=(8, 6))

    for file_path in pickle_files:
        obj = load_pickled_plot(file_path)

        # Handle both Figure and Axes objects
        if isinstance(obj, plt.Figure):
            for src_ax in obj.axes:
                for line in src_ax.get_lines():
                    ax.plot(line.get_xdata(), line.get_ydata(), label=f"{Path(file_path).stem}: {line.get_label() or 'curve'}")
        elif isinstance(obj, plt.Axes):
            for line in obj.get_lines():
                ax.plot(line.get_xdata(), line.get_ydata(), label=f"{Path(file_path).stem}: {line.get_label() or 'curve'}")
        else:
            print(f"Skipping {file_path}: unsupported type {type(obj)}")

    ax.set_title("Combined Curves from 4 Pickled Plots")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Usage: python overlay_pickled_plots.py plot1.pkl plot2.pkl plot3.pkl plot4.pkl
    main(sys.argv[1:])
