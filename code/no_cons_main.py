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
from ggg import model_fusion, emb_pu
import random
from random import sample
from mask_recovery import augment_data, mask_data
from main_file_full import PUTreeClassifier
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
import pickle
import networkx as nx
from sklearn.metrics import fbeta_score
from sklearn.model_selection import GridSearchCV

def draw_PUTree(node, save_path):
    """
    Recursively draw the PUTree and display node information.

    Args:
    - node (Node): The current node in the PUTree.
    - save_path (str): The file path to save the PNG image.
    """
    G = nx.DiGraph()
    node_labels = {}

    def traverse_tree(current_node):
        if current_node is None:
            return

        G.add_node(current_node.k)



        node_labels[current_node.k] = f"Node {current_node.k}\nDepth: {current_node.depth}\nSamples: {len(current_node.D)}\nClass Prior: {current_node.prior}\nActual Prior: {current_node.actual_p}        \nPath_fusion_id: {current_node.path_fusion_id}\nSibling: {current_node.sibling}\nFeature: {current_node.f}\nThreshold: {current_node.t}"

        if current_node.left is not None:
            G.add_edge(current_node.k, current_node.left.k)
            traverse_tree(current_node.left)

        if current_node.right is not None:
            G.add_edge(current_node.k, current_node.right.k)
            traverse_tree(current_node.right)

    traverse_tree(node)

    plt.figure(figsize=(20, 16))
    pos = nx.spring_layout(G)
    nx.draw_networkx(G, pos=pos, with_labels=False, node_size=800, node_color="lightblue", edge_color="gray", alpha=0.8)
    nx.draw_networkx_labels(G, pos=pos, labels=node_labels, font_size=10, font_color="black", verticalalignment="center")
    plt.axis("off")
    plt.title("PUTree")
    plt.savefig(save_path, format="png")
    plt.show()

def data_pre(train_path, test_path, root_pu_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test = pd.read_csv(test_path)
    train = pd.read_csv(train_path) # include original label
    X_train, y_train, X_test, y_test, features = get_train_test(train.drop(columns = ["original_label"]),test)
    prior = actual_pi(train)
    model = pu_model.PUModel(dim=X_train.shape[-1]).to("cpu").float()
    model.load_state_dict(torch.load(root_pu_path))
    explainer = lime_tabular.RecurrentTabularExplainer(X_train, feature_names=features,
                                                    discretize_continuous=True,
                                                    class_names=['P', 'N'],
                                                    discretizer='quartile',
                                                    mode='classification')

    mask_d, mask_l,top_f= mask_data(train,test,model)
    gamma_best, lambd_best, p_best, MSE_best, predictY_best = GridMLSSVR(mask_d,mask_l,5)
    alpha, b = MLSSVRTrain(mask_d, mask_l, gamma_best, lambd_best, p_best)
    return train,prior,test,mask_d, alpha, b, lambd_best, p_best, top_f,device


def cal_res(model, X_test,y_test,batch=4000):
    X_test= torch.from_numpy(X_test).float().to(device)
    y_test = torch.from_numpy(y_test).float().to(device)

    # Create a TensorDataset and a DataLoader
    dataset = TensorDataset(X_test, y_test)
    dataloader = DataLoader(dataset, batch_size=batch, shuffle=True)
    p = []
    gt = []
    for X_batch, y_batch in tqdm(dataloader, desc='Batch', leave=True):
        output,cons = model(X_batch)
        pred = torch.where(output <= 0.5, torch.tensor(0, device=device), torch.tensor(1, device=device)) 
        p.append(pred)
        gt.append(y_batch)
    pr = []
    for i in range(len(p)):
        pr+=[int(i[0]) for i in p[i]]
    gt2 = []
    for i in range(len(gt)):
        gt2+=list(gt[i])
    gt2 = [int(i) for i in gt2]

    Accuracy = metrics.accuracy_score(gt2, pr)
    Precision  = metrics.precision_score(gt2, pr)
    Recall = metrics.recall_score(gt2, pr)
    F1 = metrics.f1_score(gt2, pr)
    return Accuracy, Precision, Recall, F1,gt2,pr

def check_data(train,test,f,t):
    left_train = train[train[f]<=t]
    right_train = train[train[f]>t]
    left_test = test[test[f]<=t]
    right_test = test[test[f]>t]
    left_prior = actual_pi(left_train)
    right_prior = actual_pi(right_train)
    X_train_left, y_train_left, X_test_left, y_test_left, features_left = get_train_test(left_train.drop(columns = ["original_label"]),left_test)
    X_train_right, y_train_right, X_test_right, y_test_right, features_right = get_train_test(right_train.drop(columns = ["original_label"]),right_test)
    return left_train,left_prior,X_test_left,y_test_left, right_train,right_prior,X_test_right,y_test_right,X_train_left, y_train_left,X_train_right, y_train_right

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# test = pd.read_csv("../../../dataset/test_di_im.csv")
# train = pd.read_csv("../../../dataset/train_di.csv") # include original label
# X_train, y_train, X_test, y_test, features = get_train_test(train.drop(columns = ["original_label"]),test)
# prior = actual_pi(train)
# model = pu_model.PUModel(dim=X_train.shape[-1]).to("cpu").float()
# model.load_state_dict(torch.load("../../../model/PUtree/jobs_main/new_diabetes_result/pytorch_model.bin"))
# explainer = lime_tabular.RecurrentTabularExplainer(X_train, feature_names=features,
#                                                    discretize_continuous=True,
#                                                    class_names=['P', 'N'],
#                                                    discretizer='quartile',
#                                                    mode='classification')

# mask_d, mask_l,top_f= mask_data(train,test,model)
# gamma_best, lambd_best, p_best, MSE_best, predictY_best = GridMLSSVR(mask_d,mask_l,5)
# alpha, b = MLSSVRTrain(mask_d, mask_l, gamma_best, lambd_best, p_best)


#list1 = [mask_d, alpha, b, lambd_best, p_best, top_f]
#with open("aug_information.pkl", "wb") as file:
    #pickle.dump(list1, file)

#with open("aug_information.pkl", "rb") as file:
    #loaded_data = pickle.load(file)

# Assign the loaded data to variables
#mask_d, alpha, b, lambd_best, p_best, top_f = loaded_data


train,prior,test,mask_d, alpha, b, lambd_best, p_best, top_f,device = data_pre("../../../dataset/train.csv","../../../dataset/test.csv","../../../model/PUtree/jobs_main/china_result/pytorch_model.bin")
putree = PUTreeClassifier(max_depth=2, min_node_size=2000,mask_d=mask_d, alpha=alpha, b=b, 
                          lambd_best=lambd_best, p_best=p_best, top_f = top_f,
                          device = device,fusion_epoch=10,pu_epoch = 10,bet = 0.5,local_bet=0,aug=False)

putree.fit(train,prior,test)
prob, pred = putree.predict(test)

print("PUtree accuracy is :",metrics.accuracy_score(test.label, pred))
print("PUtree precision is :",metrics.precision_score(test.label, pred))
print("PUtree recall is :",metrics.recall_score(test.label, pred))
print("PUtree f1 is :",metrics.f1_score(test.label, pred))
print("PUtree f2 is :",fbeta_score(test.label, pred, beta=2))

# a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12 = check_data(train,test,putree.root.f,putree.root.t)
# a1,b1,c1,d1,gt1,pr1 = cal_res(putree.T[3], a3,a4)
# print(a1)
# print(b1)
# print(c1)
# print(d1)
# print(len(gt1))
# a2,b2,c2,d2,gt2,pr2 = cal_res(putree.T[5], a7,a8)
# print(a2)
# print(b2)
# print(c2)
# print(d2)
# print(len(gt2))

# print("split criter is ", putree.root.f)
# print("split threshold is ", putree.root.t)
# print("PUtree accuracy is :",metrics.accuracy_score(gt1+gt2, pr1+pr2))
# print("PUtree precision is :",metrics.precision_score(gt1+gt2, pr1+pr2))
# print("PUtree recall is :",metrics.recall_score(gt1+gt2, pr1+pr2))
# print("PUtree f1 is :",metrics.f1_score(gt1+gt2, pr1+pr2))

# Assuming you have an instance of PUTreeClassifier called 'putree'
save_path = "china_tree1.png"
draw_PUTree(putree.root, save_path)
