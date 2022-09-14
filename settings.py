from transformers import BertTokenizer, BertConfig


HIDDEN_SIZE = 768
TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased')
CONFIGURATION = BertConfig(output_hidden_states=True)
CONFIGURATION.hidden_size = HIDDEN_SIZE
FREEZE_BERT = True
NUM_EPOCHS = 100
BATCH_SIZE = 50
DEVICE = "cuda"