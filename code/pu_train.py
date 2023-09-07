from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import argparse
import pandas as pd
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import Adam
import shutil
import pu_model
import pu_loss
import pu_dataset
from sklearn.metrics import classification_report
from sklearn import metrics
import math
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler



PULoss = pu_loss.PULoss
PUdata = pu_dataset.PUdata
PNdata = pu_dataset.PNdata


#from model import PUModel
#from loss import PULoss
#from dataset import PU_MNIST, PN_MNIST


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def train(args, model, device, train_loader, optimizer, prior, epoch,nnpu,scheduler):
    model.train()
    tr_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data.float())

        loss_fct = PULoss(prior = prior,nnpu = nnpu)
        loss = loss_fct(output.view(-1), target.type(torch.float))
        tr_loss += loss.item()

        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    scheduler.step()
    print("Train loss: ", tr_loss)


def test(args, model, device, test_loader, prior,nnpu):
    """Testing"""
    model.eval()
    test_loss = 0
    correct = 0
    num_pos = 0
    p = []
    gt = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data.float())
            test_loss_func = PULoss(prior = prior,nnpu = nnpu)
            test_loss += test_loss_func(output.view(-1), target.type(torch.float)).item() # sum up batch loss
            pred = torch.where(output <= 0.5, torch.tensor(0, device=device), torch.tensor(1, device=device)) 
            num_pos += torch.sum(pred == 1)
            correct += pred.eq(target.view_as(pred)).sum().item()
            p.append(pred)
            gt.append(target)
    pr = []
    for i in range(len(p)):
        pr+=[int(i[0]) for i in p[i]]
    gt2 = []
    for i in range(len(gt)):
        gt2+=list(gt[i])
    gt2 = [int(i) for i in gt2]

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    print('Percent of examples predicted positive: ', float(num_pos)/len(test_loader.dataset), '\n')

# Precision, Recall, Accuracy, and F1
    print("Accuracy:",round(metrics.accuracy_score(gt2, pr),4))
    print("Precision:",round(metrics.precision_score(gt2, pr),4))
    print("Recall:",round(metrics.recall_score(gt2, pr),4))
    print("F1:",round(metrics.f1_score(gt2, pr),4))
    print("AUC:",round(metrics.roc_auc_score(gt2, pr),4))


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--dataset", "-d",
                        default="unsw",
                        type=str,
                        choices=['unsw', 'nsl_kdd',"diabetes"],
                        help="dataset to train and test model")
    parser.add_argument("--nnpu",
                        action='store_true',
                        help="Whether to us non-negative pu-learning risk estimator.")
    parser.add_argument("--train_batch_size",
                        default=10000,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=100,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=1e-3,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs","-e",
                        default=100,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='how many batches to wait before logging training status')
    args = parser.parse_args()
    if args.dataset=="unsw":
        train_set = PUdata(csv_file="../../dataset/china_train2.csv")
        test_set =PNdata(csv_file="../../dataset/china_test2.csv")
    dim = np.prod(list(train_set.x_data.shape[1:]))
    out_path = "./jobs_main/china_data_result_2"
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    output_model_file = os.path.join(out_path, "pytorch_model.bin")

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    model = pu_model.PUModel(dim=dim).to(device).float()
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=10000, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=2000, shuffle=True)
    prior = train_set.get_prior()
    #prior = 0.2
    optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=0.005)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    nnpu = args.nnpu
    for epoch in range(1, args.num_train_epochs + 1):
        train(args, model, device, train_loader, optimizer, prior, epoch,nnpu,scheduler)
        test(args, model, device, test_loader, prior,nnpu)

    torch.save(model.state_dict(), output_model_file)

if __name__ == "__main__":
    main()
