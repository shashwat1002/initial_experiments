import train
import data
import model_setup
import getopt
import sys
from settings import *
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import tempfile
import torch.distributed as dist
import torch.multiprocessing as mp


def run_gib_control_task(model_path, rank, world_size):
    """
    Control Task 1:
        - replace all the negation with a gibberish word
        - run inference using a trained model
    """

    dataset1 = data.NegLamaDataet(TRAIN_FILE_PATH, BERT_INPUT_SIZE)
    dataset2 = data.NegLamaDataet(TEST_FILE_PATH, BERT_INPUT_SIZE)

    model = DDP(model_setup.ExperimentModel(CONFIGURATION).to(DEVICE))
    model.load_state_dict(torch.load(model_path))

    # print("Normal run")
    train.test_model(dataset1, model=model)
    train.test_model(dataset2, model=model)

    dataset1 = data.NegLamaDataet(TRAIN_FILE_PATH, BERT_INPUT_SIZE, control_task=True)
    dataset2 = data.NegLamaDataet(TEST_FILE_PATH, BERT_INPUT_SIZE, control_task=True)

    print("Control run")
    train.test_model(dataset1, model=model)
    train.test_model(dataset2, model=model)


def main(rank, world_size):
    train.setup(rank, world_size)
    opt_vals, opt_left = getopt.getopt(sys.argv[1:], longopts=["model_file="], shortopts=None)
    model_path = ""
    for option_name, option_val in opt_vals:
        if option_name == "--model_file":
            model_path = option_val

    run_gib_control_task(model_path, rank, world_size)


if __name__ == "__main__":
    WORLD_SIZE = torch.cuda.device_count()
    print(f"number of devices: {WORLD_SIZE}")
    print("START", flush=True)
    mp.spawn(
        main, args=(WORLD_SIZE,), nprocs=WORLD_SIZE
    )