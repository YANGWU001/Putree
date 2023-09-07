import logging
import pandas as pd
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import Adam
import pu_model
import pu_loss
import pu_dataset
from sklearn.metrics import classification_report
from sklearn import metrics
from tqdm import tqdm
import torch.optim.lr_scheduler as lr_scheduler

PULoss = pu_loss.PULoss
PUdata = pu_dataset.PUdata
PNdata = pu_dataset.PNdata


logging.basicConfig(format='% (asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='% m / %d /% Y % H: % M: % S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def train(model,train_loader, optimizer, prior, epoch, nnpu,device,scheduler,upper_model=None,local_bet =0):
    model.train()
    tr_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data.float())

        loss_fct = PULoss(prior=prior, nnpu=nnpu,device = device)
        loss = loss_fct(output.view(-1), target.type(torch.float))

        if upper_model is not None:
            soft_label = upper_model(data.float())
            cons = torch.sum(torch.abs(soft_label.view(-1)-output.view(-1))**2)/len(soft_label.view(-1))
            loss = loss+local_bet*cons

        loss.backward()
        optimizer.step()
    scheduler.step()

def train_pu_model(train_data,train_prior, nnpu=True, train_batch_size=10000, eval_batch_size=5000,
                   learning_rate=1e-4, num_train_epochs=10,device = "cpu",upper_model=None,local_bet = 0):
    train_set = PUdata(train_data)
    local_bet= local_bet
    dim = np.prod(list(train_set.x_data.shape[1:]))
    if upper_model is not None:
        upper_model = upper_model.to(device)
    model = pu_model.PUModel(dim=dim).to(device).float()
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=train_batch_size, shuffle=True)
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=0.005)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    for epoch in tqdm(range(1, num_train_epochs + 1)):
        train(model, train_loader, optimizer, train_prior, epoch, nnpu,device = device,scheduler = scheduler,upper_model = upper_model,local_bet = local_bet)
    return model
