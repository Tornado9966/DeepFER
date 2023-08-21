from __future__ import print_function
import os
import cv2
import pandas as pd
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from deep_emotion import FERModel
from data_loaders import PlainDataset, eval_data_dataloader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transformation = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])
dataset = PlainDataset(csv_file = '/test.csv', img_dir = 'test/', datatype = 'finaltest', transform = transformation)
test_loader =  DataLoader(dataset,batch_size=64,num_workers=0)

net = FERModel()
net.load_state_dict(torch.load('fermodel.pt'))
net.to(device)
net.eval()

#Model Evaluation on test data
classes = ('Angry', 'Disgust', 'Fear', 'Happy','Sad', 'Surprise', 'Neutral')
total = []
def evalOnTest(dataset):
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = net(data)
            pred = F.softmax(outputs,dim=1)
            classs = torch.argmax(pred,1)
            wrong = torch.where(classs != labels,torch.tensor([1.]).cuda(),torch.tensor([0.]).cuda())
            acc = 1- (torch.sum(wrong) / 64)
            total.append(acc.item())

    print('Accuracy of the network on the' + dataset + 'test images: %d %%' % (100 * np.mean(total)))