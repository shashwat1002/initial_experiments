from transformers import RobertaTokenizer, RobertaConfig


TOKENIZER = RobertaTokenizer.from_pretrained('roberta-base')
CONFIGURATION = RobertaConfig.from_pretrained('roberta-base', output_hidden_states=True)
FREEZE_BERT = True
NUM_EPOCHS = 200
BATCH_SIZE = 100
DEVICE = "cuda"

TRAIN_FILE_PATH = "lama_train.txt"
TEST_FILE_PATH = "lama_test.txt"
DEV_FILE_PATH = "lama_dev.txt"
BERT_INPUT_SIZE = 128
MODEL_PATH = "model2.pth"
LEARNING_RATE = 1e-5
NUM_LAYERS = 13

sep_token = "</s>"
cls_token = "<s>"
pad_token = "<pad>"


OPTIONS = [
    'model_name=',
    'batch_size=',
    'learning_rate=',
    'num_epochs=',
]

def print_global_vars():
    print(MODEL_PATH)
    print(BATCH_SIZE)
    print(LEARNING_RATE)
    print(NUM_EPOCHS)
