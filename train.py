from model_setup import ExperimentModel
from settings import *
import settings
from data import *
from sklearn import metrics
from icecream import ic
from torch import nn
import getopt
import sys
import time
from torch.nn.parallel import DistributedDataParallel as DDP
import tempfile
import torch.distributed as dist
import torch.multiprocessing as mp

LAYER_NUM = 13


torch.autograd.set_detect_anomaly(True)
ic.disable()


def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


def train_epoch(model, loss_funs, optimizers, dataloader, rank, world_size):
    model.train()

    epoch_loss = []
    for i in range(LAYER_NUM):
        epoch_loss.append(0)

    count = 0

    for batch in dataloader:
        batch[1].cuda(rank)
        predictions = model(batch[0])
        predictions_tensor = torch.stack(predictions, dim=1).cuda(rank)
        ic(predictions_tensor.device)
        ic(predictions_tensor.size())

        ic(batch[1].size())
        targets = batch[1].unsqueeze(dim=1).repeat(1, LAYER_NUM)
        ic(targets.size())
        # number of bert layers

        loss_list = [loss_funs[i](predictions_tensor[:, i, :], targets[:, i].cuda(rank)) for i in range(LAYER_NUM)]
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

def train(model, loss_fun, optimizers, train_dataset, test_dataset, rank, world_size):

    for epoch in range(settings.NUM_EPOCHS):
        dataloader = make_dataloader(train_dataset, rank, world_size, batch_size=settings.BATCH_SIZE)
        print(f"Starting epoch number: {epoch+1}", flush=True)
        epoch_losses = train_epoch(model, loss_fun, optimizers, dataloader)
        string_line = '\t'.join([str(i) for i in epoch_losses])
        print(string_line, flush=True)

        if rank == 0:
            # save stuff and test model only in the master process
            torch.save(model, settings.MODEL_PATH)
            dist.barrier() # make all processes wait for end of write
            print(f"Train classification report: {epoch+1}", flush=True)
            test_model(train_dataset)
            print(f"Test classification report: {epoch+1}", flush=True)
            test_model(test_dataset)

def train_model(rank, train_dataset, test_dataset, world_size):

    setup(rank, world_size) # for initialization of distributed stuff
    bert_model = ExperimentModel(CONFIGURATION, HIDDEN_SIZE).cuda(rank) # rank will have the specific GPU id

    bert_model_ddp = DDP(bert_model, device_ids=[rank])

    loss_funs = []
    for i in range(LAYER_NUM):
        loss_fun = nn.CrossEntropyLoss()
        loss_funs.append(loss_fun)
    # optimizer = torch.optim.Adam(bert_model.parameters(), lr=1e-2)

    optimizers = []
    for layer in bert_model_ddp.classification_layers:
        optimizers.append(torch.optim.Adam(layer.parameters(), lr=settings.LEARNING_RATE))

    train(bert_model, loss_funs, optimizers, train_dataset, test_dataset)

def test_model(dataset):


    # This function uses single GPU
    dataloader = DataLoader(dataset, batch_size=int(settings.BATCH_SIZE*0.5))
    predictions_all = []
    targets_all = []
    model = torch.load(settings.MODEL_PATH)
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


# # ic(bert_model(next(iter(test_dataloader))[0]))



def get_options():

    opt_vals, opt_left = getopt.getopt(sys.argv[1:], '', longopts=OPTIONS)
    for option_name, option_val in opt_vals:
        if option_name == '--model_name':
            settings.MODEL_PATH = f"{option_val}_{time.strftime('%d_%m_%Y_%H_%M')}.pth"
        elif option_name == '--batch_size':
            settings.BATCH_SIZE = int(option_val)
        elif option_name == '--learning_rate':
            settings.LEARNING_RATE = float(option_val)
        elif option_name == '--num_epochs':
            settings.NUM_EPOCHS = int(option_val)



get_options()
print_global_vars()

# for reproducibility reasons
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


WORLD_SIZE = torch.cuda.device_count()
print(f"number of devices: {WORLD_SIZE}")

if __name__ == "__main__":

    print("START", flush=True)
    train_dataset = NegLamaDataet(TRAIN_FILE_PATH, BERT_INPUT_SIZE)
    test_dataset = NegLamaDataet(TEST_FILE_PATH, BERT_INPUT_SIZE)

    mp.spawn(
        train_model, args=(train_dataset, test_dataset, WORLD_SIZE)
    )

