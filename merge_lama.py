import os
from settings import *
from pathlib import Path
import random


def process_file(path):

    lines = []
    with open(path, "r") as input_file:
        for line in input_file:
            lines.append(line)
    return lines




def process_directory(entry_dir_path_obj):
    tensor_and_target = []

    for entry in entry_dir_path_obj.iterdir():
        if entry.is_dir():
            tensor_and_target += process_directory(entry)
        else:
            tensor_and_target += process_file(entry.absolute())
    return tensor_and_target

def merge_dataset(dataset_path_entry):
    path_obj = Path(dataset_path_entry)

    all_lines = process_directory(path_obj)

    return all_lines


def create_test_dev_train_split_lama(dataset_path_entry):
    all_lines = merge_dataset(dataset_path_entry)

    random.shuffle(all_lines)

    total_size = len(all_lines)
    train_size = int(0.7 * total_size)
    dev_size = int(0.15 * total_size)

    train_split = all_lines[:train_size]
    dev_split = all_lines[train_size:train_size+dev_size]
    test_split = all_lines[train_size+dev_size:]

    return train_split, dev_split, test_split

def create_test_train_dev_files(dataset_path_entry):
    train_split, dev_split, test_split = create_test_dev_train_split_lama(dataset_path_entry)

    with open("lama_train.txt", "w") as train_file:
        for line in train_split:
            train_file.write(line)
            # train_file.write("\n")

    with open("lama_dev.txt", "w") as dev_file:
        for line in dev_split:
            dev_file.write(line)
            # dev_file.write("\n")

    with open("lama_test.txt", "w") as test_file:
        for line in test_split:
            test_file.write(line)
            # test_file.write("\n")


create_test_train_dev_files("LAMA_primed_negated")




