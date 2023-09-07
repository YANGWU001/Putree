from torch import nn
import torch


class PULoss(nn.Module):
    """wrapper of loss function for PU learning"""

    def __init__(self, prior, loss = (lambda x: (-torch.log(torch.clamp(x, min=1e-8)))), gamma=1, beta=0, nnpu=False, l2_lambda=0,device = "cpu"):
        #loss=(lambda x: torch.sigmoid(-x))
        super(PULoss,self).__init__()
        if not 0 < prior < 1:
            raise NotImplementedError("The class prior should be in (0, 1)")
        self.prior = prior
        self.gamma = gamma
        self.beta = beta
        self.loss_func = loss#lambda x: (torch.tensor(1., device=x.device) - torch.sign(x))/torch.tensor(2, device=x.device)
        self.nnpu = nnpu
        self.positive = 1
        self.unlabeled = 0
        self.min_count = torch.tensor(1.)
        self.l2_lambda = l2_lambda
        self.device = device
    
    def forward(self, inp, target, test=False):
        assert(inp.shape == target.shape)    
        inp = inp.to(self.device)
        target = target.to(self.device)    
        positive, unlabeled = target == self.positive, target == self.unlabeled
        positive, unlabeled = positive.type(torch.float), unlabeled.type(torch.float)
        if inp.is_cuda:
            self.min_count = self.min_count.to(self.device)
        n_positive, n_unlabeled = torch.max(self.min_count, torch.sum(positive)), torch.max(self.min_count, torch.sum(unlabeled))
        #pos_ten = torch.ones(inp.shape)
        #neg_ten = torch.ones(inp.shape) * 0
        #y_positive = self.loss_func(inp, pos_ten) * positive
        #y_positive_inv = self.loss_func(inp, neg_ten) * positive
        #y_unlabeled = self.loss_func(inp, neg_ten) * unlabeled
        inp_inv = 1- inp
        y_positive = self.loss_func(inp) * positive
        y_positive_inv = self.loss_func(inp_inv) * positive
        y_unlabeled = self.loss_func(inp_inv) * unlabeled

        
        #y_positive = self.loss_func(positive*inp) * positive
        #y_positive_inv = self.loss_func(-positive*inp) * positive
        #y_unlabeled = self.loss_func(-unlabeled*inp) * unlabeled

        positive_risk = self.prior * torch.sum(y_positive)/ n_positive
        negative_risk = - self.prior *torch.sum(y_positive_inv)/ n_positive + torch.sum(y_unlabeled)/n_unlabeled

        if negative_risk < -self.beta and self.nnpu:
            pu_loss =  -self.gamma * negative_risk
        else:
            pu_loss =  positive_risk+negative_risk
        # L2 regularization
        l2_reg = torch.tensor(0., device=self.device)
        if self.l2_lambda > 0:
            for param in self.parameters():
                l2_reg += torch.norm(param, p=2)

        return pu_loss + self.l2_lambda * l2_reg
       
