from __future__ import print_function
from rankpruning import RankPruning
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression


# class prior estimation
def general_pi(data):
    train_data = data.copy()
    X_train = train_data.drop(labels = ["label","original_label"],axis = 1)
    y_train = train_data["label"]
    clf = LogisticRegression()
    clf = clf.fit(X_train,y_train)
    result1 = pd.DataFrame(clf.predict_proba(X_train)).sort_values(by = [1],ascending = False)
    result1.columns = ["0","1"]
    result2 = result1[result1["0"]>0.5]
    mean_v = result2["0"].mean()
    pi  = result1[result1["0"]<mean_v].shape[0] / result1.shape[0]
    return pi

def actual_pi(data):
    num = data[data["label"]==1].shape[0]
    p = data[data["original_label"]==1].shape[0]-num
    n = data[data["original_label"]==0].shape[0]
    return max(p/(p+n),0.2)

def rank_pi(data):
    num = data[data["label"]==1].shape[0]
    if "original_label" in data.columns:
        X = np.array(data.drop(columns=["label","original_label"]))
    else:
        X = np.array(data.drop(columns=["label"]))
    y = np.array(data.label)
    for i in range(len(y)):
        if y[i]==-1:
            y[i]=0
    # Create and fit Rank Pruning object using any clf 
    # of your choice as long as it has predict_proba() defined
    rp = RankPruning(clf = LogisticRegression())
    # rp.fit(X, s, positive_lb_threshold=1-frac_pos2neg, negative_ub_threshold=frac_neg2pos) 
    rp.fit(X, y)
    pi = (num/(1-rp.rh1)-num)/(X.shape[0]-num)
    return pi