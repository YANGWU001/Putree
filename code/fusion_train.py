from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pu_loss
PULoss = pu_loss.PULoss
from sklearn import metrics
import torch.optim.lr_scheduler as lr_scheduler


class GMUFusion(nn.Module):
    def __init__(self, dim,beta=1,path_model = [],device = "cpu"):
        super(GMUFusion, self).__init__()
        self.dim = dim
        self.depth = len(path_model)
        self.beta = beta
        self.relu = nn.ReLU()
        self.fc = nn.Linear(300,1)
        self.path_model = path_model
        self.device = device
        self.linears = nn.ModuleList([nn.Linear(300, 1) for _ in range(self.depth)])
        self.gates = nn.ModuleList([nn.Linear(300 * self.depth, 1, bias=False) for _ in range(self.depth)])
        self.gate_2 = nn.ModuleList([nn.Linear(300, 300, bias=False) for _ in range(self.depth)])
        self.in_pu = nn.ModuleList([nn.Linear(self.dim + 300, self.dim, bias=False) for _ in range(self.depth)])



    def forward(self, x):

        cons = 0
        gat = 0
        ag = 0

          # Perform local consistency
        self.__dict__["A"+str(0)] = emb_pu(self.path_model[0], x, self.device)
        for i in range(1, self.depth):
          t1= self.__dict__["A"+str(i-1)]
          t2 = x
          t1 = torch.flatten(t1,start_dim=1)
          t2 = torch.flatten(t2,start_dim=1)
          t3 = torch.cat((t1, t2), dim=1)
          if len(t3.shape)==2:
            t3 = torch.unsqueeze(t3, 1)
          self.__dict__["A"+str(i)] = emb_pu(self.path_model[i], self.in_pu[i-1](t3), self.device)
          
          # Compute y values for each linear layer
        for i in range(self.depth):
          self.__dict__["y"+str(i)] = self.linears[i](self.__dict__["A"+str(i)])

          # Compute local consistency
        for i in range(self.depth - 1):
          cons += torch.abs(self.__dict__["y"+str(i+1)] - self.__dict__["y"+str(i)])**2
        cons = torch.sum(torch.flatten(cons)) / cons.shape[0]
        cons = self.beta * cons

          # Perform gating
        for i in range(self.depth):
          if i == 0:
               gat = self.__dict__["A"+str(i)]
          else:
               gat = torch.cat((gat, self.__dict__["A"+str(i)]), 1)

        for i in range(self.depth):
          ag += self.relu(self.gate_2[i](self.__dict__["A"+str(i)])) * F.sigmoid(self.gates[i](gat))

        y = self.fc(ag)
        y = F.sigmoid(y)
        return y, cons

def cal_res(model, X_test,y_test,batch=4000,device = "cpu"):
    model.eval()
    # Create a TensorDataset and a DataLoader
    dataset = TensorDataset(X_test, y_test)
    dataloader = DataLoader(dataset, batch_size=batch, shuffle=True)
    p = []
    gt = []
    for X_batch, y_batch in tqdm(dataloader, desc='Batch', leave=False):
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
    return Accuracy, Precision, Recall, F1

def emb_pu(model1,data, device):
    model1 = model1.to(device)
    model1.eval()
    embeddings = []
    for i in data:
        embedding = model1.fc4(model1.bn3(F.relu(model1.fc3(model1.bn2(F.relu(model1.fc2(model1.bn1(F.relu(model1.fc1(i))))))))))
        embeddings.append(embedding)
    return torch.cat(embeddings,dim=0)

def model_fusion(path_model, X_train, y_train, X_test, y_test,num_epochs=30, prior=0.5, nnpu=True, device=torch.device('cpu'), batch_size=4000,l2_lambda=0,bet = 0):
    # Convert numpy arrays to PyTorch tensors
    X_train = torch.from_numpy(X_train).float().to(device)
    y_train = torch.from_numpy(y_train).float().to(device)
    X_test= torch.from_numpy(X_test).float().to(device)
    y_test = torch.from_numpy(y_test).float().to(device)

    # Create a TensorDataset and a DataLoader
    dataset = TensorDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Create an instance of the GMU Fusion model
    dim = X_train.shape[-1]
    fusion_model = GMUFusion(dim = dim,beta=bet,path_model = path_model,device = device).to(device)
    

    # Define the optimization criterion and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.SGD(fusion_model.parameters(), lr=1e-3)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    best_f1= 0.0
    best_model =None
    best_epoch = -1

    # Outer training loop
    for epoch in range(num_epochs):
        # Inner loop over batches
        fusion_model.train()
        for batch_idx,(X_batch, y_batch) in tqdm(enumerate(dataloader), desc='Batch', leave=True):
            # Forward pass
            fused_output,cons = fusion_model(X_batch)

            # Compute the loss
            loss_fct = PULoss(prior=prior, nnpu=nnpu,l2_lambda=l2_lambda)
            loss = loss_fct(fused_output.flatten(), y_batch.type(torch.float))+cons

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(X_batch), len(dataloader.dataset),
                100. * batch_idx / len(dataloader), loss.item()))
        scheduler.step()    

        Accuracy, Precision, Recall, F1 = cal_res(fusion_model, X_test, y_test,device=device)
        print("Test_Accuracy:",Accuracy)
        print("Test_Precision:",Precision)
        print("Test_Recall:",Recall)
        print("Test_F1:",F1)

        if F1>best_f1:
          best_f1= F1
          best_model = fusion_model.state_dict()
          best_epoch = epoch

      
    best_fusion_model = GMUFusion(dim=dim, beta=bet, path_model=path_model, device=device).to(device)
    if best_model is not None:
      best_fusion_model.load_state_dict(best_model)
    else:
      best_fusion_model = fusion_model
    print("Best F1 score achieved at epoch:", best_epoch)
    return best_fusion_model
