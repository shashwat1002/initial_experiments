from transformers import BertTokenizer, BertConfig


HIDDEN_SIZE = 768
TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased')
CONFIGURATION = BertConfig(output_hidden_states=True)
CONFIGURATION.hidden_size = HIDDEN_SIZE
FREEZE_BERT = True
NUM_EPOCHS = 500
BATCH_SIZE = 50
DEVICE = "cuda"

TRAIN_FILE_PATH = "lama_train.txt"
TEST_FILE_PATH = "lama_test.txt"
DEV_FILE_PATH = "lama_dev.txt"
BERT_INPUT_SIZE = 128
