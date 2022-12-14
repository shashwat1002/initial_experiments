from torch.utils.data import Dataset, DataLoader, DistributedSampler
import json
from transformers import BertTokenizer
from settings import *
import torch
from icecream import ic
import shuffler
import string
import random


def transform_lama_to_experiment_format(dictionary, control_task=0):

    """
    control_task: takes one of three integers
        - 0: no control task, OG experiment
        - 1: control task of replacing negation particles with gibberish and running trained probes
        - 2: Hewitt based control task (elaborated later)

    The Experiment one generated mappings like so:
        - AA -> 0
        - AA' -> 1
        - A'A -> 1
        - A'A' -> 0

    This function transforms it like so when control_task=2:
        - AA -> 0
        - AA' -> 0
        - A'A -> 1
        - A'A' -> 1

    The purpose is a control task
    """

    sentence1 = dictionary["masked_sentences"]
    sentence2 = dictionary["masked_negations"]
    mask_replace = dictionary["obj_label"]

    sentence1 = sentence1[0].replace("[MASK]", mask_replace)
    sentence2 = sentence2[0].replace("[MASK]", mask_replace)

    sentences = [sentence1, sentence2]

    if control_task == 1:
        # replace negation particles with random gibberish
        s = string.ascii_lowercase[:26]
        l = list(s)
        random.Random(shuffler.SEED).shuffle(l)
        cipher = ''.join(l)
        sentences = list(shuffler.random_char_control(sentence1, sentence2, cipher))


    combinations = []

    for i in range(2):
        for j in range(2):
            sentence = f"{cls_token} " + sentences[i] + f" {sep_token} " + sentences[j] + f" {sep_token}"
            outcome = -1

            if control_task == 0:
                if i == j:
                    outcome = 0
                else:
                    outcome = 1
            elif control_task == 2:
                if i == 0:
                    outcome = 0
                else:
                    outcome = 1

            combinations.append((sentence, outcome))

    return combinations



class NegLamaDataet(Dataset):
    def __init__(self, file_path, inputsize, control_task=0):

        """
        Parameters:
            file_path: path of the file we'll be creating our dataset from
            inputsize: sequence length
            control_task: integer that takes one of 3 values
                - 0: no control task
                - 1: original control task of running ingerence on dataset with negation particles replaced with gibberish
                - 2: control task based on Hewitt et. al. with the mapping of pairs changed.
        """

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
            entry for dictionary in self.parsed_dictionaries for entry in transform_lama_to_experiment_format(dictionary, control_task)
        ]


        tokenized_and_padded_and_target = []

        for i in range(len(self.transformed_inputs_text)):
            og_input = self.transformed_inputs_text[i][0]
            target = self.transformed_inputs_text[i][1]
            tokenized = TOKENIZER.convert_tokens_to_ids(TOKENIZER.tokenize(og_input))
            len_tokenized = len(tokenized)

            pad = TOKENIZER.convert_tokens_to_ids(TOKENIZER.tokenize(f"{pad_token}"))

            tokenized_plus_pad = tokenized + pad * (inputsize - len_tokenized)
            tokenized_plus_pad = torch.tensor(tokenized_plus_pad) # this tensor will be in the CPU

            tokenized_and_padded_and_target.append((tokenized_plus_pad, target))


        self.tensor_and_target = tokenized_and_padded_and_target



    def __len__(self):
        return len(self.transformed_inputs_text)

    def __getitem__(self, index):
        # ic(self.transformed_inputs_text[index])
        return self.tensor_and_target[index]


def make_dataloader(dataset, batch_size):
    
    # so that there are no redundant datapoints across processes
    sampler = DistributedSampler(dataset, shuffle=False)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)



# test_dataset = NegLamaDataet("LAMA_primed_negated/data/ConceptNet/high_ranked/ConceptNet.jsonl", 512)

# test_dataloader = DataLoader(test_dataset, batch_size=1)

# for batch in test_dataloader:
#     ic(batch)


# imdb_dataset = load_dataset('imdb')
# ic(imdb_dataset['train'][12])
# ic(TOKENIZER.tokenize(imdb_dataset['train'][12]['text']))