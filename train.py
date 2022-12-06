from model_setup import ExperimentModel
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
import os

LAYER_NUM = 13


torch.autograd.set_detect_anomaly(True)
ic.disable()


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


def train_epoch(model, loss_funs, optimizers, dataloader, rank, world_size):

    print(f"modelpath: {settings.MODEL_PATH}, epochs: {settings.NUM_EPOCHS}, learning_rate: {settings.LEARNING_RATE}", flush=True)
    
    epoch_loss = []
    for i in range(LAYER_NUM):
        epoch_loss.append(0)

    count = 0

    for batch in dataloader:
        batch = [batch[0].to(rank), batch[1].to(rank)]
        predictions = model(batch[0])
        predictions_tensor = torch.stack(predictions, dim=1).to(rank)
        ic(predictions_tensor.device)
        ic(predictions_tensor.size())

        ic(batch[1].size())
        targets = batch[1].unsqueeze(dim=1).repeat(1, LAYER_NUM)
        ic(targets.size())
        # number of bert layers

        loss_list = [loss_funs[i](predictions_tensor[:, i, :], targets[:, i].to(rank)) for i in range(LAYER_NUM)]
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

    dataloader = make_dataloader(train_dataset, batch_size=settings.BATCH_SIZE)
    print(f"modelpath: {settings.MODEL_PATH}, epochs: {settings.NUM_EPOCHS}, learning_rate: {settings.LEARNING_RATE}", flush=True)
    for epoch in range(settings.NUM_EPOCHS):
        dataloader.sampler.set_epoch(epoch)
        print(f"Starting epoch number: {epoch+1}, rank: {rank}", flush=True)

        if epoch != 0:
            print(f"about to load model for epoch two, rank: {rank}")
            map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
            model.load_state_dict(torch.load(settings.MODEL_PATH, map_location=map_location))

        epoch_losses = train_epoch(model, loss_fun, optimizers, dataloader, rank, world_size)
        string_line = '\t'.join([str(i) for i in epoch_losses])
        print(string_line, flush=True)
        if rank == 0:
            # save stuff and test model only in the master process
            torch.save(model.state_dict(), settings.MODEL_PATH)
            print(f"Train classification report: {epoch+1}", flush=True)
            test_model(train_dataset, rank, model=model)
            print(f"Test classification report: {epoch+1}", flush=True)
            test_model(test_dataset, rank, model=model)
        print(f"rank at end: {rank}")
        dist.barrier() # for sync

def train_model(rank, world_size):

    setup(rank, world_size) # for initialization of distributed stuff
    print(f"Rank: {rank}")
    get_options()
    print(f"modelpath: {settings.MODEL_PATH}, epochs: {settings.NUM_EPOCHS}, learning_rate: {settings.LEARNING_RATE}", flush=True)
    bert_model = ExperimentModel(CONFIGURATION).to(rank) # rank will have the specific GPU id

    bert_model_ddp = DDP(bert_model, device_ids=[rank])

    train_dataset = NegLamaDataet(TRAIN_FILE_PATH, BERT_INPUT_SIZE)
    test_dataset = NegLamaDataet(TEST_FILE_PATH, BERT_INPUT_SIZE)

    loss_funs = []
    for i in range(LAYER_NUM):
        loss_fun = nn.CrossEntropyLoss()
        loss_funs.append(loss_fun)
    # optimizer = torch.optim.Adam(bert_model.parameters(), lr=1e-2)

    optimizers = []
    # TODO: REQUIRES FURTHER LOOK FOR CORRECTNESS
    for layer in bert_model_ddp.module.classification_layers:
        optimizers.append(torch.optim.Adam(layer.parameters(), lr=settings.LEARNING_RATE))

    train(bert_model_ddp, loss_funs, optimizers, train_dataset, test_dataset, rank, world_size)

    dist.barrier()
    dist.destroy_process_group()

def test_model(dataset, rank=0, model=None):


    # This function uses single GPU
    print(f"modelpath: {settings.MODEL_PATH}, epochs: {settings.NUM_EPOCHS}, learning_rate: {settings.LEARNING_RATE}", flush=True)
    
    dataloader = DataLoader(dataset, batch_size=int(settings.BATCH_SIZE*0.5))
    predictions_all = []
    targets_all = []
    print("testtt")
    if model is None:
        model = DDP(ExperimentModel(settings.CONFIGURATION).to(rank), device_ids=[rank])
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        model.load_state_dict(torch.load(settings.MODEL_PATH, map_location=map_location))

    try:
        model_tmp = model.module
        model = model_tmp
    except AttributeError:
        pass


    for i in range(LAYER_NUM):
        predictions_all.append([])
    with torch.no_grad():
        for batch in dataloader:

            batch = [batch[0].to(rank), batch[1].to(rank)]

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
        print(metrics.classification_report(targets_all, predictions_all[i], digits=4), flush=True)


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




# for reproducibility reasons
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False




if __name__ == "__main__":

    WORLD_SIZE = torch.cuda.device_count()
    print(f"number of devices: {WORLD_SIZE}")
    print("START", flush=True)
    print_global_vars()

    mp.spawn(
        train_model, args=(WORLD_SIZE,), nprocs=WORLD_SIZE
    )

