from model_setup import ExperimentModel
from settings import *
from data import *
from sklearn import metrics
from icecream import ic
from torch import nn

LAYER_NUM = 13


torch.autograd.set_detect_anomaly(True)
ic.disable()

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
        print(f"Starting epoch number: {epoch+1}", flush=True)
        epoch_losses = train_epoch(model, loss_fun, optimizers, dataloader)
        string_line = '\t'.join([str(i) for i in epoch_losses])
        print(string_line, flush=True)
        with open(losses_file_path, "a") as losses_file:
            losses_file.write(string_line)
            losses_file.write('\n')
        torch.save(model, MODEL_PATH)
        print(f"Train classification report: {epoch+1}", flush=True)
        test_model(train_dataset)
        print(f"Test classification report: {epoch+1}", flush=True)
        test_model(test_dataset)

def train_model(train_dataset, test_dataset, losses_file_path):

    bert_model = ExperimentModel(CONFIGURATION, HIDDEN_SIZE).to(DEVICE)
    loss_funs = []
    for i in range(LAYER_NUM):
        loss_fun = nn.CrossEntropyLoss()
        loss_funs.append(loss_fun)
    # optimizer = torch.optim.Adam(bert_model.parameters(), lr=1e-2)

    optimizers = []
    for layer in bert_model.classification_layers:
        optimizers.append(torch.optim.Adam(layer.parameters(), lr=LEARNING_RATE))

    train(bert_model, loss_funs, optimizers, train_dataset, test_dataset, losses_file_path)

def test_model(dataset):


    dataloader = DataLoader(dataset, batch_size=int(BATCH_SIZE*0.5))
    predictions_all = []
    targets_all = []
    model = torch.load(MODEL_PATH)
    model.eval()

    for i in range(LAYER_NUM):
        predictions_all.append([])
    with torch.no_grad():
        for batch in dataloader:
            batch[1] = batch[1].detach()
            batch[0] = batch[0].detach()
            batch[1].to(DEVICE)
            ic(batch[1].size()) # batch size, 1
            predictions = model(batch[0])
            predictions = [prediction.detach() for prediction in predictions]
            # ic(predictions.size())
            predictions_tensor = torch.stack(predictions, dim=1).to(DEVICE)
            ic(predictions_tensor.size()) # 1, 13, 2
            predictions_argmax = torch.argmax(predictions_tensor, dim=2)
            ic(predictions_argmax.size()) # batch, 13
            # predictions_argmax_flat = torch.flatten(predictions_argmax)
            # targets_flattened = torch.flatten(batch[1])


            for i in range(LAYER_NUM):
                predictions_all[i] += predictions_argmax[:, i].tolist()

            ic(batch[1])
            ic(batch[1].size())
            targets_all += batch[1].tolist()



    # for prediction_total in predictions_all:
    #     ic(len(prediction_total))
    # ic(len(targets_all))

    for i in range(LAYER_NUM):
        print(f"Layer no. {i}", flush=True)
        print(metrics.classification_report(targets_all, predictions_all[i]), flush=True)


# test_dataset = NegLamaDataet("LAMA_primed_negated/data/ConceptNet/high_ranked/ConceptNet.jsonl", BERT_INPUT_SIZE)
print("START", flush=True)
train_dataset = NegLamaDataet(TRAIN_FILE_PATH, BERT_INPUT_SIZE)
test_dataset = NegLamaDataet(TEST_FILE_PATH, BERT_INPUT_SIZE)


# ic(bert_model(next(iter(test_dataloader))[0]))

train_model(train_dataset, test_dataset, "loss_out.txt")