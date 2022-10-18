from matplotlib import pyplot
import re
from scipy.signal import savgol_filter
import matplotlib as mpl

def accuracy_analysis(file_path):

    with open(file_path, "r") as out_data:

        accuracy_line_count = 0
        layer_train_accuracy_data = [[] for i in range(12)]
        layer_test_accuracy_data = [[] for i in range(12)]

        # this is because train and test reports alternate
        # entire train report followed by the entire test report
        test_train_toggle = True

        for line in out_data:
            line_parsed = re.split("  *", line)
            if accuracy_line_count % 12 == 0:
                test_train_toggle = not test_train_toggle
            print(line_parsed)
            try:
                if not test_train_toggle:
                    if line_parsed[1] == "accuracy":
                        layer_train_accuracy_data[accuracy_line_count % 12].append(float(line_parsed[2]))
                        accuracy_line_count += 1

                else:
                    if line_parsed[1] == "accuracy":
                        layer_test_accuracy_data[accuracy_line_count % 12].append(float(line_parsed[2]))
                        accuracy_line_count += 1

            except:
                pass
        print(f"accuracy line count: {accuracy_line_count}")
        return layer_train_accuracy_data, layer_test_accuracy_data

train_rep, test_rep = accuracy_analysis("test_result_two.txt")
# print(data)

def plot_function(data, name):
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'pink', 'brown', 'orange', 'teal', 'coral', 'lightblue', 'lime', 'lavender', 'turquoise', 'darkgreen', 'tan', 'salmon', 'gold'])
    fig, ax = pyplot.subplots(3)
    for index, row in enumerate(data):
        ax[int(index/4)].plot(range(len(row)), row, label=f"{index+1}")
    for i in range(3):
        ax[i].legend()
    pyplot.savefig(f'figures/{name}.png')


def smooth_plot(data, name):
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'pink', 'brown', 'orange', 'teal', 'coral', 'lightblue', 'lime', 'lavender', 'turquoise', 'darkgreen', 'tan', 'salmon', 'gold'])
    fig, ax = pyplot.subplots()
    for index, row in enumerate(data):
        print(len(row))
        smooth = savgol_filter(row, 101, 9)
        ax.plot(range(len(smooth)), smooth, label=f"{index+1}")
    ax.legend()
    pyplot.savefig(f'figures/{name}.png')

plot_function(train_rep, "train_3subplots")
plot_function(train_rep, "test_3subplots")

smooth_plot(train_rep, "train_smooth")
smooth_plot(train_rep, "test_smooth")

