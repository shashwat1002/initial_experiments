from transformers import RobertaModel
from torch import nn
from settings import *
from data import *
from sklearn import metrics
from icecream import ic


INTERMEDIATE_1 = 50
INTERMEDIATE_2 = 2

class ExperimentModel(nn.Module):

    def __init__(self, bert_config):
        super().__init__()

        self.bert_layer = RobertaModel.from_pretrained('roberta-base', config=bert_config)
        bert_dim = bert_config.hidden_size
        self.bert_layer.eval()

        # freeze bert
        for param in self.bert_layer.parameters():
            param.requires_grad = False

        # freeze bert
        # if FREEZE_BERT:
        #     for param in self.bert_layer.parameters():
        #         torch.no_grad(param)

        self.classification1 = nn.Linear(bert_dim, 2)
        self.classification2 = nn.Linear(bert_dim, 2)
        self.classification3 = nn.Linear(bert_dim, 2)
        self.classification4 = nn.Linear(bert_dim, 2)
        self.classification5 = nn.Linear(bert_dim, 2)
        self.classification6 = nn.Linear(bert_dim, 2)
        self.classification7 = nn.Linear(bert_dim, 2)
        self.classification8 = nn.Linear(bert_dim, 2)
        self.classification9 = nn.Linear(bert_dim, 2)
        self.classification10 = nn.Linear(bert_dim, 2)
        self.classification11 = nn.Linear(bert_dim, 2)
        self.classification12 = nn.Linear(bert_dim, 2)
        self.classification13 = nn.Linear(bert_dim, 2)



        self.classification_layers = [
            self.classification1,
            self.classification2,
            self.classification3,
            self.classification4,
            self.classification5,
            self.classification6,
            self.classification7,
            self.classification8,
            self.classification9,
            self.classification10,
            self.classification11,
            self.classification12,
            self.classification13
        ]

    def forward(self, batch):

        # assuming batch_size * sentence_length
        with torch.no_grad():
           all_hidden_embeddings = self.bert_layer(batch)['hidden_states']
        # ic(all_hidden_embeddings[0].requires_grad)
        ic(len(all_hidden_embeddings))
        ic(all_hidden_embeddings[0].size())
        # all_hidden_embeddings has 12 elements
        # batch size of each tensor: batch_size x sentence_length x hidden_size

        CLS_INDEX = 0
        ic(all_hidden_embeddings[0][:, CLS_INDEX, :].size())
        sentence_rep_tensors = [torch.mean(all_hidden_embeddings[i], dim=1, keepdim=True).squeeze(dim=1) for i in range(len(self.classification_layers))]
        ic(sentence_rep_tensors[0].size())
        scores_across_layers = [module(sentence_rep_tensors[i]) for i, module in enumerate(self.classification_layers)]
        # ic(scores_across_layers)
        # ic(scores_across_layers[0].size())
        return scores_across_layers


class ExperimentModelDeep(nn.Module):

    def __init__(self, bert_config):
        super().__init__()

        self.bert_layer = RobertaModel.from_pretrained('roberta-base', config=bert_config)
        bert_dim = bert_config.hidden_size
        self.bert_layer.eval()

        # freeze bert
        for param in self.bert_layer.parameters():
            param.requires_grad = False

        # freeze bert
        # if FREEZE_BERT:
        #     for param in self.bert_layer.parameters():
        #         torch.no_grad(param)

        self.classification1 = nn.Sequential(
            nn.Linear(bert_dim, INTERMEDIATE_1),
            nn.ReLU(),
            nn.Linear(INTERMEDIATE_1, INTERMEDIATE_2)
        )
        self.classification2 = nn.Sequential(
            nn.Linear(bert_dim, INTERMEDIATE_1),
            nn.ReLU(),
            nn.Linear(INTERMEDIATE_1, INTERMEDIATE_2)
        )
        self.classification3 = nn.Sequential(
            nn.Linear(bert_dim, INTERMEDIATE_1),
            nn.ReLU(),
            nn.Linear(INTERMEDIATE_1, INTERMEDIATE_2)
        )
        self.classification4 = nn.Sequential(
            nn.Linear(bert_dim, INTERMEDIATE_1),
            nn.ReLU(),
            nn.Linear(INTERMEDIATE_1, INTERMEDIATE_2)
        )
        self.classification5 = nn.Sequential(
            nn.Linear(bert_dim, INTERMEDIATE_1),
            nn.ReLU(),
            nn.Linear(INTERMEDIATE_1, INTERMEDIATE_2)
        )
        self.classification6 = nn.Sequential(
            nn.Linear(bert_dim, INTERMEDIATE_1),
            nn.ReLU(),
            nn.Linear(INTERMEDIATE_1, INTERMEDIATE_2)
        )
        self.classification7 = nn.Sequential(
            nn.Linear(bert_dim, INTERMEDIATE_1),
            nn.ReLU(),
            nn.Linear(INTERMEDIATE_1, INTERMEDIATE_2)
        )
        self.classification8 = nn.Sequential(
            nn.Linear(bert_dim, INTERMEDIATE_1),
            nn.ReLU(),
            nn.Linear(INTERMEDIATE_1, INTERMEDIATE_2)
        )
        self.classification9 = nn.Sequential(
            nn.Linear(bert_dim, INTERMEDIATE_1),
            nn.ReLU(),
            nn.Linear(INTERMEDIATE_1, INTERMEDIATE_2)
        )
        self.classification10 = nn.Sequential(
            nn.Linear(bert_dim, INTERMEDIATE_1),
            nn.ReLU(),
            nn.Linear(INTERMEDIATE_1, INTERMEDIATE_2)
        )
        self.classification11 = nn.Sequential(
            nn.Linear(bert_dim, INTERMEDIATE_1),
            nn.ReLU(),
            nn.Linear(INTERMEDIATE_1, INTERMEDIATE_2)
        )
        self.classification12 = nn.Sequential(
            nn.Linear(bert_dim, INTERMEDIATE_1),
            nn.ReLU(),
            nn.Linear(INTERMEDIATE_1, INTERMEDIATE_2)
        )
        self.classification13 = nn.Sequential(
            nn.Linear(bert_dim, INTERMEDIATE_1),
            nn.ReLU(),
            nn.Linear(INTERMEDIATE_1, INTERMEDIATE_2)
        )



        self.classification_layers = [
            self.classification1,
            self.classification2,
            self.classification3,
            self.classification4,
            self.classification5,
            self.classification6,
            self.classification7,
            self.classification8,
            self.classification9,
            self.classification10,
            self.classification11,
            self.classification12,
            self.classification13
        ]

    def forward(self, batch):

        # assuming batch_size * sentence_length
        with torch.no_grad():
           all_hidden_embeddings = self.bert_layer(batch)['hidden_states']
        # ic(all_hidden_embeddings[0].requires_grad)
        ic(len(all_hidden_embeddings))
        ic(all_hidden_embeddings[0].size())
        # all_hidden_embeddings has 12 elements
        # batch size of each tensor: batch_size x sentence_length x hidden_size

        CLS_INDEX = 0
        ic(all_hidden_embeddings[0][:, CLS_INDEX, :].size())
        sentence_rep_tensors = [torch.mean(all_hidden_embeddings[i], dim=1, keepdim=True).squeeze(dim=1) for i in range(len(self.classification_layers))]
        ic(sentence_rep_tensors[0].size())
        scores_across_layers = [module(sentence_rep_tensors[i]) for i, module in enumerate(self.classification_layers)]
        # ic(scores_across_layers)
        # ic(scores_across_layers[0].size())
        return scores_across_layers



# test_model(test_dataset)
