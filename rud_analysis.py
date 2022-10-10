from matplotlib import pyplot
import re

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

data = accuracy_analysis("test_result_one.txt")
print(data)

def plot_function(data):
    
    fig, ax = pyplot.subplots()
    for index, row in enumerate(data):
        ax.plot(range(len(row)), row, label=f"{index+1}")
    ax.legend()
    pyplot.show()

plot_function(data)
