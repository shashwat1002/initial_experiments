from torch.utils.data import Dataset, DataLoader, DistributedSampler
import json
from transformers import BertTokenizer, RobertaModel
from settings import *
import torch
from icecream import ic
import shuffler
import string
import random
import pickle
import os
import shutil
import timeit
ic.enable()

def transform_lama_to_experiment_format(dictionary, control_task=False):
    sentence1 = dictionary["masked_sentences"]
    sentence2 = dictionary["masked_negations"]
    mask_replace = dictionary["obj_label"]

    sentence1 = sentence1[0].replace("[MASK]", mask_replace)
    sentence2 = sentence2[0].replace("[MASK]", mask_replace)

    sentences = [sentence1, sentence2]

    if control_task:
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

            if i == j:
                outcome = 0
            else:
                outcome = 1

            combinations.append((sentence, outcome))

    return combinations

class NegLamaDataet(Dataset):
    def __init__(self, file_path, inputsize, control_task=False):

        """
        Parameters:
            file_path: path of the file we'll be creating our dataset from
            inputsize: sequence length
            control_task: is the dataset being created for general learning or the control task. The control replaced negation particles with random stuff
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


class SequenceRepDataset(Dataset):
    """
    Dataset that has the sequences after being passed through RoBERTa
    Targets are kept as is
    """

    def __init__(self, directory_path, dataset_orig=None):
        """
        dataset_orig: not None implies we'll have to create the representation dataset from it
        directory_path: this is supposed to be a directory which is supposed to have several datafiles, together they make the dataset
        """

        self.list_of_per_layer_representations_and_target = []
        self.directory_path = directory_path
        self.len = 0

        # this is to implement caching
        # the file corresponding to self.file_index is completely loaded in memory
        self.file_index = -1
        self.list_of_reps_and_targets = []

        if dataset_orig is not None:

            try:
                os.makedirs(self.directory_path)
            except FileExistsError:
                # reset
                shutil.rmtree(self.directory_path)
                os.makedirs(self.directory_path)


            bert_model = RobertaModel.from_pretrained('roberta-base', config=CONFIGURATION).to(DEVICE)
            bert_model.eval()

            on_gpu = [(item.to(DEVICE), torch.tensor(target).to(DEVICE)) for item, target in dataset_orig.tensor_and_target]
            self.len = len(on_gpu)
            file_index = -1
            file_path = ""
            points_on_file = 0

            file = None

            for index, (item, target) in enumerate(on_gpu):


                if index % DATAPOINTS_PER_REP_FILE == 0:

                    if file is not None:
                        file.close()

                    # move to next file
                    file_index += 1
                    file_path = os.path.join(directory_path, str(file_index))

                    file = open(file_path, "wb")

                    # number of datapoints that haven't yet been put on file
                    points_left = len(on_gpu) - index

                    if points_left > DATAPOINTS_PER_REP_FILE:
                        points_on_file = DATAPOINTS_PER_REP_FILE
                    else:
                        points_on_file = points_left


                    # first entry on the file is the number of datapoints on that file
                    pickle.dump(points_on_file, file)

                with torch.no_grad():
                    all_hidden_embeddings = bert_model(item.unsqueeze(0))['hidden_states']
                    ic(all_hidden_embeddings[0].size())
                    sentence_rep = [torch.mean(all_hidden_embeddings[i], dim=1, keepdim=False).squeeze(dim=0) for i in range(NUM_LAYERS)]
                ic(sentence_rep[0].size())


                pickle.dump((
                    [embedding_tensor.to("cpu") for embedding_tensor in sentence_rep], target.to("cpu")
                ), file)

            file.close()

        else:

            for filename in os.listdir(self.directory_path):
                file_path = os.path.join(self.directory_path, filename)
                with open(file_path, "rb") as file:
                    # first entry of each file is the number of datapoints on that file
                    self.len += pickle.load(file)


    def __len__(self):
        # return len(self.list_of_per_layer_representations_and_target)
        return self.len

    def __getitem__(self, index):
        to_ret = None
        file_index = int(index / DATAPOINTS_PER_REP_FILE)
        index_in_file = index % DATAPOINTS_PER_REP_FILE

        if file_index == self.file_index:
            return self.list_of_reps_and_targets[index_in_file]
        else:
            file_path = os.path.join(self.directory_path, str(file_index))

            with open(file_path, "rb") as f:
                datapoints_in_file = pickle.load(f)

                # print(f"here: {index}")
                for i in range(datapoints_in_file):
                    self.list_of_reps_and_targets.append(pickle.load(f))
            # ic(self.list_of_per_layer_representations_and_target[index][1].device)
            # return self.list_of_per_layer_representations_and_target[index]
            self.file_index = file_index
            return self.list_of_reps_and_targets[index_in_file]


dataset = NegLamaDataet(TEST_FILE_PATH, BERT_INPUT_SIZE)
rep_dataset = SequenceRepDataset(TEST_SENTENCE_REP_SCRATCH_PATH, dataset)

rep_dataset = SequenceRepDataset(TEST_SENTENCE_REP_SCRATCH_PATH)
rep_dataloader = DataLoader(rep_dataset, batch_size=800, shuffle=True)

start = timeit.default_timer()
batch = next(iter(rep_dataloader))
stop = timeit.default_timer()

ic(len(batch[0]))
ic(batch[1])
ic([item.size() for item in batch[0]])
ic(batch[1].size())

ic(stop - start)

# test_dataset = NegLamaDataet("LAMA_primed_negated/data/ConceptNet/high_ranked/ConceptNet.jsonl", 512)

# test_dataloader = DataLoader(test_dataset, batch_size=1)

# for batch in test_dataloader:
#     ic(batch)


# imdb_dataset = load_dataset('imdb')
# ic(imdb_dataset['train'][12])
# ic(TOKENIZER.tokenize(imdb_dataset['train'][12]['text']))