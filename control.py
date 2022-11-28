import train
import data
import model_setup
import getopt
import sys
from settings import *


def run_gib_control_task(model_path):
    """
    Control Task 1:
        - replace all the negation with a gibberish word
        - run inference using a trained model
    """

    dataset1 = data.NegLamaDataet(TRAIN_FILE_PATH, BERT_INPUT_SIZE)
    dataset2 = data.NegLamaDataet(TEST_FILE_PATH, BERT_INPUT_SIZE)

    model = model_setup.ExperimentModel(CONFIGURATION)
    model.load_state_dict(model_path)

    train.test_model(dataset1, model=model)
    train.test_model(dataset2, model=model)

def main():
    opt_vals, opt_left = getopt.getopt(sys.argv[1:], longopts=["model_file="])
    model_path = ""
    for option_name, option_val in opt_vals:
        if option_name == "--model_file":
            model_path = option_val

    run_gib_control_task(model_path)


if __name__ == "__main__":
    main()