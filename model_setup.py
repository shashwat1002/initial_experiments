from transformers import RobertaModel
from torch import nn
from settings import *
from data import *
from sklearn import metrics
from icecream import ic


INTERMEDIATE_1 = 300
INTERMEDIATE_2 = 2

class ExperimentModelDepercated(nn.Module):

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



class ClassificationProbes:
    """
    This class houses classification probes for all layers of centre model (bert or roberta)
    The probes expect the _sequence representation_ for sequence classification
    """

    def __init__(self, rep_dim, num_layers, num_classes=2):
        """
        rep_dim: dimensions of the sentence representation
        num_layers: number of layers, that is number of probes
        """
        super().__init__()
        self.probes = nn.ModuleList(
            [nn.Linear(rep_dim, num_classes) for i in range(num_layers)]
        )

    def forward(self, batch):
        """
        batch: [batch_size * rep_dim] * num_layers, it is a list of tensors each with dimensions batch_size * rep_dim
        a batch of sentence representations
        """

        scores_across_layers = [module(sentence_rep_tensors[i]) for i, module in enumerate(self.probes)]

        return scores_across_layers


class ExperimentModel:
    """
    This class will have a frozen bert and all the probes
    """

    def __init__(self, bert_config):
        super().__init__()
        self.bert_layer = RobertaModel.from_pretrained('roberta-base', config=bert_config)
        self.bert_layer.eval() # put it in eval mode so that dropout is not active

        for param in self.bert_layer.parameters():
            param.requires_grad = False

        self.probe_model = ClassificationProbes(bert_config.hidden_size, NUM_LAYERS, num_classes=2)

        self.pooling_method = self.mean_pooling

    def mean_pooling(self, batch):
        """
        expects batch to be a list of length num_layers
        each tensor is of dimensions batch_size x sentence_length x hidden_size

        returns: a list of num_layer elements each tensor is batch_size x hidden_size

        simply means everything in a sequence
        """

        return [torch.mean(batch[i], dim=1, keepdim=True).squeeze(dim=1) for i in range(NUM_LAYERS)]



    def decide_pooling_method(self, index=0):
        """
        Method to select a method for pooling to sequence representation
        default is mean
        """

        if index == 0:
            self.pooling_method = self.mean_pooling


    def forward(self, batch):
        """
        batch: batch_size * sequence_length
        is the batch of sequences essentially
        """

        with torch.no_grad():
           all_hidden_embeddings = self.bert_layer(batch)['hidden_states']

        # all_hidden_embeddings is a list of NUM_LAYERS elements
        # each tensor in it is batchsize * sequence_length * hidden_size

        sentence_rep_tensors = self.pooling_method(batch)

        scores_across_layers = self.probe_model(sentence_rep_tensors)

        return scores_across_layers




# test_model(test_dataset)
