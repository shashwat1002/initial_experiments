from matplotlib import pyplot
import re
from scipy.signal import savgol_filter
import matplotlib as mpl
from icecream import ic

LAYER_NUM = 13

def accuracy_analysis(file_path):

    with open(file_path, "r") as out_data:

        accuracy_line_count = 0
        layer_train_accuracy_data = [[] for i in range(LAYER_NUM)]
        layer_test_accuracy_data = [[] for i in range(LAYER_NUM)]

        main_data = [layer_train_accuracy_data, layer_test_accuracy_data]
        # this is because train and test reports alternate
        # entire train report followed by the entire test report
        test_train_toggle = True

        total_accuracy_count = 0
        for line in out_data:
            line_parsed = re.split("  *", line)

            try:
                if line_parsed[1] == "accuracy":
                    main_data[int(accuracy_line_count / LAYER_NUM) % 2][accuracy_line_count % LAYER_NUM].append(float(line_parsed[2]))
                    accuracy_line_count += 1
            except IndexError:
                pass

        # print(f"accuracy line count: {accuracy_line_count}")
        ic(len(layer_test_accuracy_data[0]))
        ic(len(layer_train_accuracy_data[0]))
        return layer_train_accuracy_data, layer_test_accuracy_data

train_rep, test_rep = accuracy_analysis("test_output_roberta_proper_1.txt")
# print(data)

def plot_function(data, name):
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'pink', 'brown', 'orange', 'teal', 'coral', 'lightblue', 'lime', 'lavender', 'turquoise', 'darkgreen', 'tan', 'salmon', 'gold'])
    fig, ax = pyplot.subplots(4)
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

pyplot.ylim([0.48, 0.75])
plot_function(train_rep, "train_roberta_proper_1")
plot_function(test_rep, "test_roberta_proper_1")

smooth_plot(train_rep, "train_roberta_proper_sooth_1")
smooth_plot(test_rep, "test_roberta_proper_sooth_1")

