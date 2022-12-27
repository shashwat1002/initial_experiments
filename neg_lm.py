from transformers import RobertaForMaskedLM, RobertaTokenizer
from settings import *
import torch
from torch.utils.data import Dataset, DataLoader
from icecream import ic
import json
from tabulate import tabulate
from torch.nn import Softmax
from torch.distributions import Categorical


"""
The purpose of this file is to create a dataset for a crude experiment.

- We will collect top-k predictions for the mask token of pairs of sentences
- Each pair consists of one negated and one non-negated sentences
"""

DEVICE = "cuda"


class NegLAMADatasetMaskedLM(Dataset):
    """
    Dataset to do masked language modelling on the lama dataset
    """

    def __init__(self, file_path):

        self.parsed_dictionaries = []

        with open(file_path, "r") as input_file:
            for line in input_file:
                try:
                    self.parsed_dictionaries.append(json.loads(line))
                except json.decoder.JSONDecodeError:
                    # empty line
                    pass

        self.entries = [
            (dictionary["masked_sentences"][0].replace("[MASK]", "<mask>").replace(" .", "."),
            dictionary["masked_negations"][0].replace("[MASK]", "<mask>").replace(" .", "."),
            dictionary["obj_label"]) for dictionary in self.parsed_dictionaries
        ]

        # self.list_of_dictionaries = []

        # for trip in entries:
        #     sentence_tokenized = TOKENIZER.tokenize(trip[0])
        #     sentence_tokenized_indices = TOKENIZER.convert_tokens_to_ids(
        #         sentence_tokenized)

        #     neg_sentence_tokenized = TOKENIZER.tokenize(trip[1])
        #     neg_sentence_tokenized_indices = TOKENIZER.convert_tokens_to_ids(
        #         neg_sentence_tokenized)

        #     object_index = TOKENIZER.convert_tokens_to_ids(trip[2])
        #     # object_index = TOKENIZER.convert_tokens_to_ids(" "+trip[2])
        #     entry_dict = {
        #         "sentence": trip[0],
        #         "neg_sentence": trip[1],
        #         "sentence_tokenized": sentence_tokenized,
        #         "sentence_tokenized_indices": sentence_tokenized_indices,
        #         "neg_sentence_tokenized": neg_sentence_tokenized,
        #         "neg_sentence_tokenized_indices": neg_sentence_tokenized_indices,
        #         "object_index": object_index,
        #         "obj_label": trip[2]
        #     }

        #     self.list_of_dictionaries.append(entry_dict)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        return self.entries[index]


def send_mask_logits(model, sentence):
    inputs_tran = TOKENIZER(sentence, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        logits = model(**inputs_tran).logits

    mask_token_index = mask_token_index = (inputs_tran.input_ids == TOKENIZER.mask_token_id)[0].nonzero(as_tuple=True)[0]

    return logits[:, mask_token_index]

def top_k(k, logits_tensor):
    sorted_logits, indices = torch.sort(logits_tensor, descending=True)
    return sorted_logits[:k], indices[:k]


def test_model(model, dataset):

    dataloader = DataLoader(dataset, batch_size=1)

    input1 = next(iter(dataloader))

    masked_logits1 = send_mask_logits(model, input1[0])
    masked_logits2 = send_mask_logits(model, input1[1])

    # ic(masked_logits)
    # ic(masked_logits.size())

    preds, indices = top_k(10, masked_logits1)
    preds2, indices2 = top_k(10, masked_logits2)

    # ic(indices)
    ic(preds)
    ic(preds2)
    ic(indices.size())

    pred_tokens = TOKENIZER.convert_ids_to_tokens(indices.squeeze())
    ic(input1[2])
    ic(indices.squeeze()[0] == TOKENIZER.tokenize(" "+input1[2][0]))
    ic(indices.squeeze()[0] == indices.squeeze()[0])
    pred_tokens2 = TOKENIZER.convert_ids_to_tokens(indices2.squeeze())
    ic(pred_tokens)
    ic(pred_tokens2)


def deal_with_pair(model, input1):

    k = 20

    masked_logits1 = send_mask_logits(model, input1[0])
    masked_logits2 = send_mask_logits(model, input1[1])



    softmax_func = Softmax()
    preds1_prob = softmax_func(masked_logits1.squeeze())
    preds2_prob = softmax_func(masked_logits2.squeeze())
    preds1, indices1 = top_k(k, preds1_prob)
    preds2, indices2 = top_k(k, preds2_prob)

    categorical_preds1 = Categorical(preds1_prob)
    categorical_preds2 = Categorical(preds2_prob)

    entropy1 = categorical_preds1.entropy()
    entropy2 = categorical_preds2.entropy()

    # preds1_cat = Categorical(preds1_prob)

    change = not (indices1.squeeze()[0] == indices2.squeeze()[0])

    pred_tokens1 = TOKENIZER.convert_ids_to_tokens(indices1.squeeze())
    pred_tokens2 = TOKENIZER.convert_ids_to_tokens(indices2.squeeze())

    out_string = f"{input1[0]}\n{input1[1]}\n{str(change)}\n {entropy1}, {entropy2}\n{input1[2]}\n"

    data = []

    for i in range(k):
        data.append([pred_tokens1[i], preds1.squeeze()[i], pred_tokens2[i], preds2.squeeze()[i]])

    out_string += tabulate(data)


    return out_string


def main():
    model = RobertaForMaskedLM.from_pretrained(
        'roberta-base', config=CONFIGURATION).to(DEVICE)

    dataset = NegLAMADatasetMaskedLM(TRAIN_FILE_PATH)

    dataloader = DataLoader(dataset, batch_size=1)
    # iter_dataloader = iter(dataloader)

    for inp in dataloader:
        print(deal_with_pair(model, inp))
        print("\n")

main()