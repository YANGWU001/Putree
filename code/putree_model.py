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
from mask_recovery import augment_data, mask_data
import torch
import pu_model
import random
from random import sample
from sklearn.base import BaseEstimator
from train_pu import train_pu_model
from PULIME import get_split, gain, get_train_test
from pi_estimation import general_pi, actual_pi, rank_pi
from fusion_train import model_fusion, emb_pu,GMUFusion
from mask_recovery import augment_data
import networkx as nx
import matplotlib.pyplot as plt



class Node:
    def __init__(self, k, D=None, M=None, f=None, t=None, prior=None, is_root = False, sibling= None,test_data= None,depth=0,path_fusion_id= None,actual_p=None):
        """
        Node class to represent nodes in the decision tree.

        Attributes:
        - k (int): A unique identifier for this node.
        - D (DataFrame): The data associated with this node.
        - M (model): The nnPU model trained on the data D.
        - f (str): The feature used for splitting the data at this node.
        - t (float): The threshold used for splitting the data at this node.
        - prior (float): The prior probability for positive class estimation.
        - is_root (bool): Whether the node is the root node or not.
        - sibling (Node): The sibling node.
        - test_data (DataFrame): Test data associated with this node.
        - depth (int): The depth of the node in the decision tree.
        - path_fusion_id (list): List of path fusion model identifiers associated with this node.
        - actual_p (float): The actual positive class prior for the node's data.
        """
        self.k = k
        self.D = D
        self.M = M
        self.f = f
        self.t = t
        self.prior = prior
        self.left = None
        self.right = None
        self.is_root = is_root
        self.sibling = sibling
        self.test_data = test_data
        self.depth = depth
        self.path_fusion_id = path_fusion_id
        self.actual_p= actual_p


class PUTreeClassifier(BaseEstimator):
    def __init__(self, max_depth=None, min_node_size=None,mask_d=None, alpha=None, b=None, lambd_best=None, p_best=None, top_f = None,device = "cpu",fusion_epoch=30,pu_epoch = 10,pi_selection = "gt",bet = 0,local_bet=0,aug=False,fus = False,save_path = "putree.png"):
        """
        PUTreeClassifier class for Positive-Unlabeled tree.

        Attributes:
        - root (Node): The root node of the  PUtree.
        - T (dict): The dictionary of path fusion models, keyed by the path node k.
        - mask_d (DataFrame): The root node mask recovery generator.
        - alpha (float): Alpha parameter.
        - b (float): Beta parameter.
        - lambd_best (float): Lambda parameter.
        - p_best (float): P parameter.
        - top_f (int): Top features parameter.
        - device (str): Device for training, either "cpu" or "gpu" (default: "cpu").
        - fusion_epoch (int): Number of fusion training epochs (default: 30).
        - pu_epoch (int): Number of PU model training epochs (default: 10).
        - pi_selection (str): Positive class prior selection method ("gt", "gen", or "rk"). Meaning ground truth, general method, rankpruning.
        - bet (float): Beta parameter for model fusion.
        - local_bet (float): Local beta parameter for PU model training.
        - aug (bool): Flag to enable data augmentation (default: False).
        - fus (bool): Flag to enable model fusion (default: False).
        - save_path (str): File path to save the PUtree visualization (default: "putree.png").
        - dataset (str): The dataset being used, either "unsw", "nsl", or "diabetes" (default: "unsw").
        """

        self.root = None
        self.T = {}
        self.max_depth = max_depth
        self.min_node_size = min_node_size
        self.mask_d = mask_d
        self.alpha = alpha
        self.b = b
        self.lambd_best= lambd_best
        self.p_best = p_best
        self.top_f = top_f
        self.device = device
        self.fusion_epoch = fusion_epoch
        self.pu_epoch = pu_epoch
        self.pi_selection = pi_selection
        self.bet = bet
        self.local_bet = local_bet
        self.aug = aug
        self.fus = fus
        self.save_path = save_path
        
    def fit(self, D,test_data):
        """
        Train the PUTreeClassifier.

        Args:
        - D (DataFrame): The training data.
        """
        if self.pi_selection=="gt":
            prior = actual_pi(D)
        elif self.pi_selection=="gen":
            prior = general_pi(D)
        elif selft.pi_selection=="rk":
            prior = rank_pi(D)
        self.root = Node(k=0, D=D,prior=prior, M=self.build_nnPU_model(D,prior).to(self.device), is_root= True,test_data=test_data, depth =0,actual_p=prior)
        if self.aug:
            mask_d, mask_l,top_f= mask_data(self.root.D,self.root.test_data,self.root.M.to("cpu"))
            gamma_best, lambd_best, p_best, MSE_best, predictY_best = GridMLSSVR(mask_d,mask_l,5)
            alpha, b = MLSSVRTrain(mask_d, mask_l, gamma_best, lambd_best, p_best)
            self.mask_d = mask_d
            self.alpha = alpha
            self.b = b
            self.lambd_best= lambd_best
            self.p_best = p_best
            self.top_f = top_f
        
        self.learn_PUTree(self.root)

    def learn_PUTree(self, node,depth=0,upper_model = None):
        """
        Recursive function to train the PUTreeClassifier.

        Args:
        - node (Node): The current node in the decision tree.
        """
        if self.termination_criterion(node):
            T_prime = self.train_path_fusion(node)
            self.T[node.k] = T_prime
        else:
            f, t = self.PUTreeLIME(node)
            node.f = f
            node.t = t

            Df_gt_t, Df_leq_t,test_data_gt_t,test_data_leq_t = self.split_data(node)
            if upper_model is None:
                upper_model = node.M
            node.right = Node(k=node.k * 2 + 1, D=Df_leq_t, test_data=test_data_leq_t,depth=depth + 1)
            node.left = Node(k=node.k * 2 + 2, D=Df_gt_t, test_data=test_data_gt_t,depth=depth + 1)

            if not self.termination_criterion(node.right) and not self.termination_criterion(node.left):
                if self.aug:
                    Df_leq_t_build = self.data_augmentation(Df_leq_t)
                else:
                    Df_leq_t_build = Df_leq_t.copy()
                Df_leq_t_build = Df_leq_t_build[Df_leq_t_build[f]<=t]
                gen_pi,gt_pi,rk_pi = self.positive_class_prior_estimation(Df_leq_t)
                if self.pi_selection=="gt":
                    pi_leq_t = gt_pi
                elif self.pi_selection=="gen":
                    pi_leq_t = gen_pi
                elif selft.pi_selection=="rk":
                    pi_leq_t = rk_pi
            
                Mf_leq_t = self.build_nnPU_model(Df_leq_t_build,pi_leq_t,upper_model).to(self.device)
                node.right = Node(k=node.k * 2 + 1, D=Df_leq_t, M=Mf_leq_t,prior= pi_leq_t, test_data=test_data_leq_t,depth=depth + 1,actual_p = gt_pi)
                
                if self.aug:
                    Df_gt_t_build = self.data_augmentation(Df_gt_t)
                else:
                    Df_gt_t_build = Df_gt_t.copy()
                Df_gt_t_build = Df_gt_t_build[Df_gt_t_build[f]>t]
                gen_pi,gt_pi,rk_pi = self.positive_class_prior_estimation(Df_gt_t)
                if self.pi_selection=="gt":
                    pi_gt_t = gt_pi
                elif self.pi_selection=="gen":
                    pi_gt_t = gen_pi
                elif selft.pi_selection=="rk":
                    pi_gt_t = rk_pi
                Mf_gt_t = self.build_nnPU_model(Df_gt_t_build,pi_gt_t,upper_model).to(self.device)
                node.left = Node(k=node.k * 2 + 2, D=Df_gt_t, M=Mf_gt_t, prior=pi_gt_t,sibling = node.right,test_data=test_data_gt_t,depth=depth + 1,actual_p = gt_pi)
                node.right.sibling = node.left

                self.learn_PUTree(node.right, depth + 1, upper_model = Mf_leq_t)
                self.learn_PUTree(node.left, depth + 1, upper_model = Mf_gt_t)
            
            elif not self.termination_criterion(node.right) and self.termination_criterion(node.left):
                T_prime = self.train_path_fusion(node.left)
                self.T[node.left.k] = T_prime

                if self.aug:
                    Df_leq_t_build = self.data_augmentation(Df_leq_t)
                else:
                    Df_leq_t_build = Df_leq_t.copy()
                Df_leq_t_build = Df_leq_t_build[Df_leq_t_build[f]<=t]
                gen_pi,gt_pi,rk_pi = self.positive_class_prior_estimation(Df_leq_t)
                if self.pi_selection=="gt":
                    pi_leq_t = gt_pi
                elif self.pi_selection=="gen":
                    pi_leq_t = gen_pi
                elif selft.pi_selection=="rk":
                    pi_leq_t = rk_pi
                Mf_leq_t = self.build_nnPU_model(Df_leq_t_build,pi_leq_t,upper_model).to(self.device)
                node.right = Node(k=node.k * 2 + 1, D=Df_leq_t, M=Mf_leq_t,prior= pi_leq_t, sibling = node.left, test_data=test_data_leq_t,depth=depth + 1,actual_p =gt_pi)
                self.learn_PUTree(node.right, depth + 1, upper_model = Mf_leq_t)
                
            elif self.termination_criterion(node.right) and not self.termination_criterion(node.left):
                T_prime = self.train_path_fusion(node.right)
                self.T[node.right.k] = T_prime
                if self.aug:
                    Df_gt_t_build = self.data_augmentation(Df_gt_t)
                else:
                    Df_gt_t_build = Df_gt_t.copy()
                Df_gt_t_build = Df_gt_t_build[Df_gt_t_build[f]>t]
                gen_pi,gt_pi,rk_pi = self.positive_class_prior_estimation(Df_gt_t)
                if self.pi_selection=="gt":
                    pi_gt_t = gt_pi
                elif self.pi_selection=="gen":
                    pi_gt_t = gen_pi
                elif selft.pi_selection=="rk":
                    pi_gt_t = rk_pi
                Mf_gt_t = self.build_nnPU_model(Df_gt_t_build,pi_gt_t,upper_model).to(self.device)
                node.left = Node(k=node.k * 2 + 2, D=Df_gt_t, M=Mf_gt_t, prior=pi_gt_t,sibling = node.right,test_data=test_data_gt_t,depth=depth + 1,actual_p = gt_pi)
                self.learn_PUTree(node.left, depth + 1, upper_model = Mf_gt_t)
            else:
                T_prime = self.train_path_fusion(node.left)
                self.T[node.left.k] = T_prime
                #T_prime = self.train_path_fusion(node.right)
                node.right.path_fusion_id= node.left.path_fusion_id
                self.T[node.right.k] = T_prime

            






    def termination_criterion(self, node):
        """
        Check if the termination criterion is met for a node.

        Args:
        - node (Node): The node to check.

        Returns:
        - (bool): True if the termination criterion is met, False otherwise.
        """
        # Terminate if maximum depth is reached
        if node.depth >= self.max_depth:
            return True

        # Terminate if the node size is smaller than the threshold
        if len(node.D) < self.min_node_size:
            return True

        # Terminate if the node is pure (all instances have the same class label)
        unique_labels = node.D['label'].unique()
        if len(unique_labels) == 1:
            return True

        return False

    def train_path_fusion(self, node):
        """
        Train path fusion on current path to get a classifier.

        Args:
        - node (Node): The current node.

        Returns:
        - (model): The trained path fusion model.
        """
        # Implement your path fusion training logic here
        # Set the fusion model to evaluation mode
        path_model = []
        nodes = []
        ids= []
        node_2 = []
  
        for _, instance in node.D.iloc[:1,].iterrows():
            path = self.get_path(self.root, instance)
            for i in path:
                #path_model.append(self.get_node(i).M)
                nodes.append(self.get_node(i))

        for path_node in nodes:
            if path_node.M is not None:
                path_model.append(path_node.M)
                ids.append(path_node.k)
                node_2.append(path_node)
        node_build= node_2[-1]
        node.path_fusion_id= ids
        #path_model = [x for x in path_model if x is not None]
        #if len(nodes)==1:
            #node_build = nodes[0]
        #else:
            #node_build = nodes[-2]
        fusion_data = node_build.D
        X_train, y_train, X_test, y_test, features = get_train_test(fusion_data.drop(columns = ["original_label"]),node_build.test_data)
        if len(path_model)==1:
            fusion = self.root.M.to(self.device)
        else:
            if self.fus:
                fusion = model_fusion(path_model, X_train, y_train,X_test, y_test,num_epochs = self.fusion_epoch,prior=node_build.actual_p,nnpu= True,device = self.device,bet = self.bet)
            else:
                fusion = path_model[-1].to(self.device)
        return fusion

    def PUTreeLIME(self, node):
        """
        Use PUTreeLIME to explain M and choose an optimal split (f, t).

        Args:
        - node (Node): The current node.

        Returns:
        - (str, float): The selected feature and threshold. 
        """

        # Implement your PUTreeLIME logic here
        if node.is_root:
            a = get_split(node.D.drop(columns = ["original_label"]),node.test_data,node.M.to("cpu")) # LIME
        else:
            if node.sibling.M is not None:
                a = get_split(node.D.drop(columns = ["original_label"]),node.test_data,node.M.to("cpu"),sibling=node.sibling.M.to("cpu")) # LIME+ uncertainty sampling
            else:
                a = get_split(node.D.drop(columns = ["original_label"]),node.test_data,node.M.to("cpu"))
        
        c = gain(a,node.D)
        be = c.loc[0,"split"]
        feature, threshold = be[0], be[2]
        return feature, threshold

    def split_data(self, node):
        """
        Split the data at a node based on the feature and threshold.

        Args:
        - node (Node): The current node.

        Returns:
        - (DataFrame, DataFrame): Two DataFrames representing the data where
          feature f values are greater than t and less than or equal to t.
        """
        f = node.f
        t = node.t
        Df_gt_t = node.D[node.D[f] > t]
        Df_leq_t = node.D[node.D[f] <= t]
        test_data_gt_t = node.test_data[node.test_data[f]>t]
        test_data_leq_t = node.test_data[node.test_data[f]<=t]
        return Df_gt_t, Df_leq_t, test_data_gt_t, test_data_leq_t

    def positive_class_prior_estimation(self, Data):
        """
        Estimate positive class prior for the split data.

        Args:
        - Data (DataFrame): used to predict its pi
        """
        # Implement your positive class prior estimation logic here
        gen_pi = general_pi(Data)
        gt_pi = actual_pi(Data)
        rk_pi = rank_pi(Data)
        return gen_pi,gt_pi,rk_pi

    def data_augmentation(self, data):
        """
        Perform data augmentation for the split data.

        Args:
        - Df_gt_t (DataFrame): The data where feature f values are greater than t.
        - Df_leq_t (DataFrame): The data where feature f values are less than or equal to t.

        Returns:
        - (DataFrame, DataFrame): Augmented versions of Df_gt_t and Df_leq_t.
        """
        # Implement your data augmentation logic here
        aug1 = augment_data(data, self.top_f, num_a=50, threshold=0.9,mask_d=self.mask_d, alpha=self.alpha, b=self.b, lambd_best=self.lambd_best, p_best=self.p_best)
        return aug1

    def build_nnPU_model(self, D,prior,upper_model = None):
        """
        Build an nnPU model for the data.

        Args:
        - D (DataFrame): The data to build the model on.

        Returns:
        - (model): The trained nnPU model.
        """
        # Implement your nnPU model building logic here
        model = train_pu_model(D,prior,device = self.device,num_train_epochs=self.pu_epoch,upper_model = upper_model,local_bet = self.local_bet)
        return model

    def predict(self, X):
        """
        Make predictions for instances in X.

        Args:
        - X (DataFrame): The instances to make predictions for.

        Returns:
        - (array-like): The predicted class labels.
        """
        predictions = []

        for _,instance in X.iterrows():
            path = self.get_path(self.root, instance)
            fusion_model = self.T[path[-1]]
            fusion_model.eval()
            instance2 = torch.tensor(np.array(instance.drop(labels="label"),dtype= "float32").reshape(1,1,instance.drop(labels="label").shape[0]))
            if isinstance(fusion_model, GMUFusion):
                prediction,cons = fusion_model(instance2.to(self.device))
            else:
                fusion_model = fusion_model.to(self.device)
                prediction = fusion_model(instance2.to(self.device))
            prediction = prediction.cpu().view(-1).detach().numpy()[0]
            predictions.append(prediction)
        pred = [0 if i <= 0.5 else 1 for i in predictions]
        return predictions, pred



   


    def get_path(self, node, instance):
        """
        Get the path an instance would take through the decision tree.

        Args:
        - node (Node): The current node.
        - instance (Series or array-like): The instance to get the path for.

        Returns:
        - (list): The path the instance would take through the decision tree.
        """
        path = [node.k]
        while node.left and node.right:
            if instance[node.f] <= node.t:
                node = node.right
            else:
                node = node.left
            path.append(node.k)
        return path
    def _traverse_tree(self, node, k):
        """
        Helper method to traverse the decision tree and find the node with the specified k value.

        Args:
        - node (Node): The current node in the traversal.
        - k (int): The k value of the node to retrieve.

        Returns:
        - (Node or None): The node with the specified k value, or None if not found.
        """
        if node is None:
            return None
        if node.k == k:
            return node
        left_node = self._traverse_tree(node.left, k)
        if left_node:
            return left_node
        right_node = self._traverse_tree(node.right, k)
        return right_node
    def get_node(self, k):
        """
        Get the node with the specified k value.

        Args:
        - k (int): The k value of the node to retrieve.

        Returns:
        - (Node or None): The node with the specified k value, or None if not found.
        """
        return self._traverse_tree(self.root, k)

    def draw_PUTree(self):
        """
        Recursively draw the PUTree and display node information in a tree structure.

        Args:
        - node (Node): The current node in the PUTree.
        - save_path (str): The file path to save the PNG image.
        """
        node = self.root
        save_path = self.save_path

        def traverse_tree(current_node, x, y, level):
            if current_node is None:
                return

            # Calculate node position
            node_x = x
            node_y = y

            # Draw the node
            plt.text(node_x, node_y, f"Node {current_node.k}\nDepth: {current_node.depth}\nSamples: {len(current_node.D)}\nClass Prior: {current_node.prior}\nActual Prior: {current_node.actual_p}\nPath_fusion_id: {current_node.path_fusion_id}\nSibling: {'Node ' + str(current_node.sibling.k) if current_node.sibling is not None else 'None'}\nFeature: {current_node.f}\nThreshold: {current_node.t}",
                    ha='center', va='center', bbox=dict(boxstyle='round,pad=0.3', edgecolor='blue', facecolor='lightblue'))

            # Draw edges to child nodes
            if current_node.left is not None:
                child_x_left = x + 3 / (2 ** level)
                child_y_left = y - 3
                plt.plot([node_x, child_x_left], [node_y, child_y_left], color='gray', linestyle='-', linewidth=2)
                traverse_tree(current_node.left, child_x_left, child_y_left, level + 1)

            if current_node.right is not None:
                child_x_right = x - 3 / (2 ** level)
                child_y_right = y - 3
                plt.plot([node_x, child_x_right], [node_y, child_y_right], color='gray', linestyle='-', linewidth=2)
                traverse_tree(current_node.right, child_x_right, child_y_right, level + 1)

        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(20, 16))
        ax.axis('off')  # Turn off axis

        # Start drawing from the root node
        traverse_tree(node, x=0, y=0, level=1)

        # Save the figure
        plt.savefig(save_path, format="png")
        plt.show()
