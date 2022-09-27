from turtle import distance
import torch.nn as nn
import torch
import numpy as np

class CE_Loss(nn.Module):
    def __init__(self, c, device):
        super(CE_Loss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        #self.classifier = classifier.to(device)
        self.softmax = nn.Softmax(dim=1)
        self.Y_pred = 0
 
    def forward(self, inputs, targets,gamma2=None):  
        #self.Y_pred = self.classifier(inputs) # prediction before softmax
        self.Y_pred=inputs
        return self.ce_loss(self.Y_pred, targets)
    
    
class CE_GALoss_old(nn.Module):
    def __init__(self, c, device):
        super(CE_GALoss_old, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.nll_loss = nn.NLLLoss()
 
    def forward(self, inputs, targets, gamma2):   
        loss = self.ce_loss(inputs,targets) 
        loss+= self.nll_loss((1/gamma2-1)*inputs,targets)
        return loss
    
class CE_GALoss(nn.Module):
    def __init__(self, c, device):
        super(CE_GALoss, self).__init__()
        self.I = torch.eye(c).to(device)
        self.ce_loss = nn.CrossEntropyLoss()
        self.nll_loss = nn.NLLLoss()
 
    def forward(self, inputs, targets, gamma2):        
        Y = self.I[targets]
        loss = self.ce_loss(1/gamma2*Y*inputs + (1-Y)*inputs,targets) 
        loss+= self.nll_loss(inputs,targets)
        return loss

class BCE_GALoss(nn.Module):
    def __init__(self, c, device):
        super(BCE_GALoss, self).__init__()
        self.I = torch.eye(c).to(device)
        self.bce_loss = nn.BCELoss()
        #self.const = np.sqrt(c)
        self.const = (c-1)/2
        #self.mse_loss = nn.MSELoss(reduction='none')
        #self.classifier = classifier.to(device)
        #self.gamma2 = nn.Parameter(torch.ones(c)*0.9)
        #self.gamma2_min = gamma2_min
        #self.gamma2_max = gamma2_max
 
    def forward(self, inputs, targets,gamma2):        
        Y = self.I[targets]
        try:
            distances=-inputs
            loss = self.bce_loss(torch.exp(-distances),Y) 
            #loss+= torch.mean(Y*distances/gamma2)/self.c #to adapt to the mean computation of bce_loss, dividing by c and m
            loss+= torch.mean((self.const/gamma2-1)*Y*distances) # positive prediction weight is gamma * const
        except RuntimeError as e:
            print("min,max D",torch.min(inputs).item(), torch.max(inputs).item())
            print("min,max output",torch.min(torch.exp(inputs)).item(), torch.max(torch.exp(inputs)).item())
            print("nans output",torch.sum(torch.isnan(torch.exp(inputs))).item())
            print(f"{e},{e.__class_}")
        return loss      

class BCE_DUQLoss(nn.Module):
    
    def __init__(self, c, device):
        super(BCE_DUQLoss, self).__init__()
        self.bce_loss = nn.BCELoss()
        self.I = torch.eye(c).to(device)
        self.classifier = classifier.to(device)
        self.Y_pred = 0 #predicted class probabilities
        self.Y= 0
    
    def forward(self, inputs, targets, gamma2=None):
        self.Y = self.I[targets]
        self.Y_pred = torch.exp(inputs)
        loss = self.bce_loss(self.Y_pred, self.Y)
        return loss
