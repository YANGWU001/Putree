from putreelime import lime_tabular
from putreelime import submodular_pick
import numpy as np
import pandas as pd


# PULIME
def get_split(train,test,model,sibling=None):
    X_train, y_train, X_test, y_test, features = get_train_test(train,test)
    explainer = lime_tabular.RecurrentTabularExplainer(X_train, feature_names=features,
                                                   discretize_continuous=True,
                                                   class_names=['P', 'N'],
                                                   discretizer='quartile',
                                                   mode='classification')
    a = result_xai(explainer, X_test,model,sibling)
    a.split = a.split.apply(clean_split)
    a = get_indicator(a)
    return a
def get_train_test(train,test):
    train = train.sample(frac = 1, ignore_index=True,random_state=40)
    test = test.sample(frac = 1, ignore_index=True,random_state=41)
    X_train = train.drop(labels=['label'], axis=1)
    y_train = train['label']
    features = X_train.columns.tolist()
    X_train = np.array(X_train,dtype= "float32").reshape(X_train.shape[0],1,X_train.shape[1])
    y_train = np.array(y_train,dtype = "float32")
    X_test = test.drop(labels=['label'], axis=1)
    y_test = test['label']
    X_test = np.array(X_test,dtype= "float32").reshape(X_test.shape[0],1,X_test.shape[1])
    y_test = np.array(y_test,dtype = "float32")
    return X_train, y_train, X_test, y_test, features
def result_xai(explainer, X_test,model,sibling = None):
    sp_obj = submodular_pick.SubmodularPick(explainer, X_test[-100:], model, sibling,sample_size=10, num_features=10, num_exps_desired=5)
    W_pick=pd.DataFrame([dict(this.as_list(this.available_labels()[0])) for this in sp_obj.sp_explanations]).fillna(0)
    W_pick['prediction'] = [this.available_labels()[0] for this in sp_obj.sp_explanations]
    #Making a dataframe of all the explanations of sampled points
    W=pd.DataFrame([dict(this.as_list(this.available_labels()[0])) for this in sp_obj.explanations]).fillna(0)
    W['prediction'] = [this.available_labels()[0] for this in sp_obj.explanations]
    #Aggregate importances split by classes
    grped_coeff = W.groupby("prediction").mean()
    grped_coeff = grped_coeff.T
    grped_coeff["abs"] = np.abs(grped_coeff.iloc[:, 0])
    grped_coeff.sort_values("abs", inplace=True, ascending=False)
    grped_coeff = grped_coeff.sort_values("abs", ascending=False).drop("abs", axis=1)
    grped_coeff["split"] = grped_coeff.index
    grped_coeff.index = range(grped_coeff.shape[0])
    grped_coeff.columns = ["feature_importance","split"]
    return grped_coeff
def clean_split(a_string):
    if "<=" in a_string:
        value = float(a_string.split("<=")[1])
        if "<" in a_string.split("<=")[0].split("_t-")[0]:
            feature = a_string.split("<=")[0].split("_t-")[0].split("<")[1]
        else:
            feature = a_string.split("<=")[0].split("_t-")[0]
        return [feature.strip(),"<=", value]
    else:
        value = float(a_string.split(">")[1])
        feature = a_string.split(">")[0].split("_t-")[0]
        return [feature.strip(),">", value]
def get_indicator(data):
    for i in range(data.shape[0]):
        if data.iloc[i,0]>=0:
            data.iloc[i,1].append("N")
        else:
            data.iloc[i,1].append("P")
    return data

def gain(a,data):
    a["left"]=0
    a["right"]=0
    a["ratio"]=0
    for i in range(a.shape[0]):
        feature = a.split[i][0]
        thre = float(a.split[i][2])
        left = data[data[feature]<=thre]
        right = data[data[feature]>thre]
        a.loc[i,"left"] = left.shape[0]
        a.loc[i,"right"] = right.shape[0]
        if left.shape[0]!=0 and right.shape[0]!=0:
            a.loc[i,"ratio"] = left.shape[0]/right.shape[0]
    a = a[a["ratio"]!=0]
    a["ratio_difference"] = abs(a["ratio"]-1)
    a = a.sort_values("ratio_difference")
    a = a.reset_index(drop=True)
    return a
