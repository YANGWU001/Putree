from __future__ import print_function
import scipy.io
from gridmlssvr import GridMLSSVR
from mlssvrtrain import MLSSVRTrain
from mlssvrpredict import MLSSVRPredict
from rankpruning import RankPruning
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import lime.lime_tabular as lime_tabular
import lime.submodular_pick as submodular_pick
import torch
import pu_model
from train_pu import train_pu_model
from PULIME import get_split, gain, get_train_test
from pi_estimation import general_pi, actual_pi, rank_pi
from fusion_train import model_fusion, emb_pu
import random
from random import sample
from mask_recovery import augment_data, mask_data
from putree_model import PUTreeClassifier
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
import pickle
import networkx as nx
from sklearn.metrics import fbeta_score
from sklearn.model_selection import GridSearchCV
import argparse


def main(args):
    device = torch.device('cuda' if args.gpu else 'cpu')
    
    # Read dataset based on the provided argument
    if args.dataset == 'unsw':
        train_file = "../dataset/unsw_train.csv"
        test_file = "../dataset/unsw_test.csv"
    elif args.dataset == 'nsl_kdd':
        train_file = "../dataset/nsl_kdd_train.csv"
        test_file = "../dataset/nsl_kdd_test.csv"
    elif args.dataset == 'diabetes':
        train_file = "../dataset/diabetes_train.csv"
        test_file = "../dataset/diabetes_test.csv"
    

    test = pd.read_csv(test_file)
    train = pd.read_csv(train_file)
    validation = test.copy() # Set the validation set to mirror the test set for monitoring the training process.
    save_path = args.dataset + "_putree.png"
    putree = PUTreeClassifier(
        max_depth=args.max_depth,
        min_node_size=args.min_node_size,
        device=device,
        fusion_epoch=args.fusion_epoch,
        pu_epoch=args.pu_epoch,
        pi_selection = args.pi,
        bet=args.bet,
        local_bet=args.local_bet,
        aug=args.aug,
        fus=args.fus,
        save_path=save_path 
    )

    putree.fit(train, validation)
    prob, pred = putree.predict(test)
    putree.draw_PUTree()

    print("PUtree accuracy is:", metrics.accuracy_score(test.label, pred))
    print("PUtree precision is:", metrics.precision_score(test.label, pred))
    print("PUtree recall is:", metrics.recall_score(test.label, pred))
    print("PUtree f1 is:", metrics.f1_score(test.label, pred))
    print("PUtree f2 is:", fbeta_score(test.label, pred, beta=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PU Tree Classifier")
    # Add command-line arguments for your configurable parameters
    parser.add_argument("--max_depth", type=int, default=2, help="Max depth of the tree")
    parser.add_argument("--min_node_size", type=int, default=2000, help="Minimum node size")
    parser.add_argument("--fusion_epoch", type=int, default=30, help="Fusion epoch")
    parser.add_argument("--pu_epoch", type=int, default=100, help="PU epoch")
    parser.add_argument("--pi", type=str, default="gt", help="pi estimation method (gt, gen, rk)")
    parser.add_argument("--bet", type=float, default=0.1, help="Bet value")
    parser.add_argument("--local_bet", type=float, default=0.9, help="Local bet value")
    parser.add_argument("--aug", action="store_true", help="Enable data augmentation")
    parser.add_argument("--fus", action="store_true", help="Enable fusion")
    parser.add_argument("--dataset", type=str, default="unsw", help="Dataset choice (unsw, nsl_kdd, diabetes)")
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available")
    args = parser.parse_args()
    main(args)
