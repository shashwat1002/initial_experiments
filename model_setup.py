from transformers import BertConfig, BertTokenizer, BertModel
from torch import nn
from settings import *
from data import *
from sklearn import metrics
from icecream import ic

torch.autograd.set_detect_anomaly(True)
ic.disable()
INTERMEDIATE_1 = 300
INTERMEDIATE_2 = 2
LAYER_NUM = 12

class ExperimentModel(nn.Module):

    def __init__(self, bert_config, bert_dim):
        super().__init__()

        self.bert_layer = BertModel(CONFIGURATION).to(DEVICE)

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
            # self.classification13
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



def train_epoch(model, loss_funs, optimizers, dataloader):
    model.train()

    epoch_loss = []
    for i in range(LAYER_NUM):
        epoch_loss.append(0)

    count = 0

    for batch in dataloader:
        batch[1].to(DEVICE)
        predictions = model(batch[0])
        predictions_tensor = torch.stack(predictions, dim=1).to(DEVICE)
        ic(predictions_tensor.device)
        ic(predictions_tensor.size())

        ic(batch[1].size())
        targets = batch[1].unsqueeze(dim=1).repeat(1, LAYER_NUM)
        ic(targets.size())
        # number of bert layers

        loss_list = [loss_funs[i](predictions_tensor[:, i, :], targets[:, i].to(DEVICE)) for i in range(LAYER_NUM)]
        ic(loss_list)

        # QUESTIONABLE STEP
        tot_loss = 0
        for i, loss in enumerate(loss_list):
            # tot_loss += loss
            optimizers[i].zero_grad()
            loss.backward(retain_graph=True)
            optimizers[i].step()
            epoch_loss[i] += loss.item()
        count += 1

    for i in range(LAYER_NUM):
        epoch_loss[i] /= count
    ic(epoch_loss)
    return epoch_loss

        # tot_loss.backward()

def train(model, loss_fun, optimizers, train_dataset, test_dataset, losses_file_path):

    for epoch in range(NUM_EPOCHS):
        dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
        epoch_losses = train_epoch(model, loss_fun, optimizers, dataloader)
        string_line = '\t'.join([str(i) for i in epoch_losses])
        print(string_line)
        with open(losses_file_path, "a") as losses_file:
            losses_file.write(string_line)
            losses_file.write('\n')
        torch.save(model, "model.pth")
        print(f"Train classification report: {epoch+1}")
        test_model(train_dataset)
        print(f"Test classification report: {epoch+1}")
        test_model(test_dataset)
        print(f"Epoch number: {epoch+1}")

def train_model(train_dataset, test_dataset, losses_file_path):

    bert_model = ExperimentModel(CONFIGURATION, HIDDEN_SIZE).to(DEVICE)
    loss_funs = []
    for i in range(LAYER_NUM):
        loss_fun = nn.CrossEntropyLoss()
        loss_funs.append(loss_fun)
    # optimizer = torch.optim.Adam(bert_model.parameters(), lr=1e-2)

    optimizers = []
    for layer in bert_model.classification_layers:
        optimizers.append(torch.optim.Adam(layer.parameters(), lr=1e-3))

    train(bert_model, loss_funs, optimizers, train_dataset, test_dataset, losses_file_path)

def test_model(dataset):


    dataloader = DataLoader(dataset, batch_size=1)
    predictions_all = []
    targets_all = []
    model = torch.load("model.pth")
    model.eval()

    for i in range(LAYER_NUM):
        predictions_all.append([])

    for batch in dataloader:
        batch[1].to(DEVICE)
        ic(batch[1].size())
        predictions = model(batch[0])
        # ic(predictions.size())
        predictions_tensor = torch.stack(predictions, dim=1).to(DEVICE)
        ic(predictions_tensor.size()) # 1, 13, 2
        predictions_argmax = torch.argmax(predictions_tensor, dim=2)
        ic(predictions_argmax.size()) # 1, 13


        for i in range(LAYER_NUM):
            predictions_all[i].append(predictions_argmax[:, i].item())

        ic(batch[1].item())
        ic(batch[1])
        ic(batch[1].size())
        targets_all.append(batch[1].item())



    # for prediction_total in predictions_all:
    #     ic(len(prediction_total))
    # ic(len(targets_all))

    for i in range(LAYER_NUM):
        print(f"Layer no. {i}")
        print(metrics.classification_report(targets_all, predictions_all[i]))


# test_dataset = NegLamaDataet("LAMA_primed_negated/data/ConceptNet/high_ranked/ConceptNet.jsonl", BERT_INPUT_SIZE)
train_dataset = NegLamaDataet(TRAIN_FILE_PATH, BERT_INPUT_SIZE)
test_dataset = NegLamaDataet(TEST_FILE_PATH, BERT_INPUT_SIZE)


# ic(bert_model(next(iter(test_dataloader))[0]))

train_model(train_dataset, test_dataset, "loss_out.txt")
# test_model(test_dataset)
