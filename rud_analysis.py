from matplotlib import pyplot
import re
from scipy.signal import savgol_filter
import matplotlib as mpl

def accuracy_analysis(file_path):

    with open(file_path, "r") as out_data:

        accuracy_line_count = 0
        layer_accuracy_data = [[] for i in range(12)]

        for line in out_data:
            line_parsed = re.split("  *", line)
            print(line_parsed)
            try:
                if line_parsed[1] == "accuracy":
                    layer_accuracy_data[accuracy_line_count % 12].append(float(line_parsed[2]))
                    accuracy_line_count += 1
            except:
                pass

        return layer_accuracy_data

data = accuracy_analysis("test_result_4_learn.txt")
print(data)

def plot_function(data):
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'pink', 'brown', 'orange', 'teal', 'coral', 'lightblue', 'lime', 'lavender', 'turquoise', 'darkgreen', 'tan', 'salmon', 'gold'])
    fig, ax = pyplot.subplots(3)
    for index, row in enumerate(data):
        ax[int(index/4)].plot(range(len(row)), row, label=f"{index+1}")
    for i in range(3):
        ax[i].legend()
    pyplot.savefig('figures/3subplots_learn4.png')


def smooth_plot(data):
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'pink', 'brown', 'orange', 'teal', 'coral', 'lightblue', 'lime', 'lavender', 'turquoise', 'darkgreen', 'tan', 'salmon', 'gold'])
    fig, ax = pyplot.subplots()
    for index, row in enumerate(data):
        smooth = savgol_filter(row, 101, 9)
        ax.plot(range(len(smooth)), smooth, label=f"{index+1}")
    ax.legend()
    pyplot.savefig('figures/smooth_4learn.png')

plot_function(data)
smooth_plot(data)

