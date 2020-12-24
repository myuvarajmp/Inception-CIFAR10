from .inception import GoogLeNet
from .inceptionv3 import Inception3
import torch
import sys,os,shutil

def getModel( modelno,training=False, **kwargs):

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if modelno == 1:
        model = GoogLeNet()
        model.eval()
    if modelno == 2:
        model = Inception3()
        model.eval()

    if training :
        model.train()
        model = torch.nn.DataParallel(model)
        print("The model is in training mode")
    print("No of params in model is " , sum(p.numel() for p in model.parameters() if p.requires_grad))
    model  = model.to(device)
    print(f"model is loaded on GPU {next(model.parameters()).is_cuda}")
    return model
