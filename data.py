from torch.utils.data import Dataset, DataLoader
import json
from transformers import BertTokenizer
from settings import *
import torch
from icecream import ic


def transform_lama_to_experiment_format(dictionary):
    sentence1 = dictionary["masked_sentences"]
    sentence2 = dictionary["masked_negations"]
    mask_replace = dictionary["obj_label"]

    sentence1 = sentence1[0].replace("[MASK]", mask_replace)
    sentence2 = sentence2[0].replace("[MASK]", mask_replace)

    sentences = [sentence1, sentence2]

    combinations = []

    for i in range(2):
        for j in range(2):
            sentence = f"{cls_token} " + sentences[i] + f" {sep_token} " + sentences[j] + f" {sep_token}"
            outcome = -1

            if i == j:
                outcome = 0
            else:
                outcome = 1

            combinations.append((sentence, outcome))

    return combinations

class NegLamaDataet(Dataset):
    def __init__(self, file_path, inputsize):

        self.inputsize = inputsize
        self.parsed_dictionaries = []

        with open(file_path, "r") as input_file:
            for line in input_file:
                try:
                    self.parsed_dictionaries.append(json.loads(line))
                except json.decoder.JSONDecodeError:
                    # empty line
                    pass


        self.transformed_inputs_text = [
            entry for dictionary in self.parsed_dictionaries for entry in transform_lama_to_experiment_format(dictionary)
        ]


        tokenized_and_padded_and_target = []

        for i in range(len(self.transformed_inputs_text)):
            og_input = self.transformed_inputs_text[i][0]
            target = self.transformed_inputs_text[i][1]
            tokenized = TOKENIZER.convert_tokens_to_ids(TOKENIZER.tokenize(og_input))
            len_tokenized = len(tokenized)

            pad = TOKENIZER.convert_tokens_to_ids(TOKENIZER.tokenize(f"{pad_token}"))

            tokenized_plus_pad = tokenized + pad * (inputsize - len_tokenized)
            tokenized_plus_pad = torch.tensor(tokenized_plus_pad).to(DEVICE)

            tokenized_and_padded_and_target.append((tokenized_plus_pad, target))


        self.tensor_and_target = tokenized_and_padded_and_target



    def __len__(self):
        return len(self.transformed_inputs_text)

    def __getitem__(self, index):
        # ic(self.transformed_inputs_text[index])
        return self.tensor_and_target[index]

# test_dataset = NegLamaDataet("LAMA_primed_negated/data/ConceptNet/high_ranked/ConceptNet.jsonl", 512)

# test_dataloader = DataLoader(test_dataset, batch_size=1)

# for batch in test_dataloader:
#     ic(batch)


# imdb_dataset = load_dataset('imdb')
# ic(imdb_dataset['train'][12])
# ic(TOKENIZER.tokenize(imdb_dataset['train'][12]['text']))