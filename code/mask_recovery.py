import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import random
from random import sample
from mlssvrpredict import MLSSVRPredict
from PULIME import get_split, gain, get_train_test

def mask_data(train,test,model):
    # Determine the number of top features to keep
    train2 = train.copy()
    test2 = test.copy()
    num_features = int(0.15 * (train2.shape[1]-2))  # remove label and original_label, so -2
    a = get_split(train2.drop(columns = ["original_label"]),test2,model) # LIME for root model
    top_features = []
    for i in a.loc[:(num_features-1),"split"]:
        top_features.append(i[0])
    top_features = list(set(top_features))
    # Print important features
    train2.drop(labels=["label","original_label"],axis = 1, inplace = True)
    data = train2.drop(top_features,axis = 1)
    labels = train2[top_features]
    length = data.shape[1]*10
    mask_d = data.sample(length,random_state = 42)
    mask_l = labels.iloc[mask_d.index.tolist(),:]

    return np.array(mask_d), np.array(mask_l),top_features

def augment_data(train, top_f, num_a=50, threshold=0.99,mask_d=None, alpha=None, b=None, lambd_best=None, p_best=None):
    m = train.copy()
    
    m.drop("label", axis=1, inplace=True)
    m.rename(columns={"original_label": "label"}, inplace=True)
    
    X = m.drop("label", axis=1)
    y = m["label"]
    
    clf = LogisticRegression()
    clf.fit(X, y)
    
    data = train.copy()
    num_p = data[data.label == 1]
    num_u = data[data.label == 0]
    num_aug = num_a
    
    good_p = pd.DataFrame(clf.predict_proba(num_p.drop(["label", "original_label"], axis=1)))
    good_p = good_p[good_p.iloc[:, 1] >= threshold]
    good_p_data = num_p.iloc[good_p.index]
    
    data1 = good_p_data.sample(num_aug, replace=True, random_state=42, ignore_index=True)
    data1 = data1.drop(["label", "original_label"], axis=1)
    
    random.seed(41)
    for i in range(num_aug):
        data1.iloc[i, sample(range(data1.shape[1]), 1)] = 0
    
    tstY = data1[top_f]
    tstX = data1.drop(top_f, axis=1)
    
    predictY, TSE, R2 = MLSSVRPredict(np.array(tstX), np.array(tstY), mask_d, alpha, b, lambd_best, p_best)
    
    aug_d = pd.DataFrame(predictY, columns=top_f)
    new_data = pd.concat([tstX, aug_d], axis=1)
    new_data2 = new_data.copy()
    new_data2 = new_data2[X.columns]
    
    a = pd.DataFrame(clf.predict_proba(new_data2))
    a = a[a.iloc[:,1]>=threshold]
    new_data2["label"] = 1
    new_data2["original_label"] = 1
    new_data2 = new_data2.iloc[a.index.tolist()]
    aug_data = pd.concat([new_data2, data], axis=0).reset_index(drop=True)
    return aug_data

