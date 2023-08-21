from __future__ import print_function
import numpy  as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms

from data_loaders import PlainDataset, eval_data_dataloader
from deep_emotion import FERModel
from generate_data import GenerateData

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def Train(epochs,train_loader,val_loader,criterion,optmizer,device):
    '''
    Training Loop
    '''
    print("===================================Start Training===================================")
    for e in range(epochs):
        train_loss = 0
        validation_loss = 0
        train_correct = 0
        val_correct = 0
        # Train the model #
        net.train()
        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)
            optmizer.zero_grad()
            outputs = net(data)
            loss = criterion(outputs,labels)

            l2_lambda = 0.0001
            l2_norm = sum(p.pow(2.0).sum()
                          for p in [net.fc1.weight, net.fc2.weight])
            
            loss = loss + l2_lambda * l2_norm
            
            loss.backward()
            optmizer.step()
            train_loss += loss.item()
            _, preds = torch.max(outputs,1)
            train_correct += torch.sum(preds == labels.data)

        # Validate the model #
        net.eval()
        for data,labels in val_loader:
            data, labels = data.to(device), labels.to(device)
            val_outputs = net(data)
            val_loss = criterion(val_outputs, labels)
            validation_loss += val_loss.item()
            _, val_preds = torch.max(val_outputs,1)
            val_correct += torch.sum(val_preds == labels.data)

        train_loss = train_loss/len(train_dataset)
        train_acc = train_correct.double() / len(train_dataset)
        validation_loss =  validation_loss / len(validation_dataset)
        val_acc = val_correct.double() / len(validation_dataset)
        print('Epoch: {} \tTraining Loss: {:.8f} \tValidation Loss {:.8f} \tTraining Accuracy {:.3f}% \tValidation Accuracy {:.3f}%'
                                                           .format(e+1, train_loss,validation_loss,train_acc * 100, val_acc*100))

    torch.save(net.state_dict(),'fermodel-{}-{}-{}.pt'.format(epochs,batchsize,lr))
    print("===================================Training Finished===================================")


if __name__ == '__main__':
    generate_dataset = GenerateData('data')
    generate_dataset.split_test()
    generate_dataset.save_images('train')
    generate_dataset.save_images('test')
    generate_dataset.save_images('val')

    epochs = 100
    lr = 0.005
    batchsize = 128

    net = FERModel()
    net.to(device)
    print("Model archticture: ", net)
    traincsv_file = 'train.csv'
    validationcsv_file = 'val.csv'
    train_img_dir = 'train/'
    validation_img_dir = 'val/'

    transformation= transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])
    train_dataset= PlainDataset(csv_file=traincsv_file, img_dir = train_img_dir, datatype = 'train', transform = transformation)
    validation_dataset= PlainDataset(csv_file=validationcsv_file, img_dir = validation_img_dir, datatype = 'val', transform = transformation)
    train_loader= DataLoader(train_dataset,batch_size=batchsize,shuffle = True,num_workers=0)
    val_loader=   DataLoader(validation_dataset,batch_size=batchsize,shuffle = True,num_workers=0)

    criterion= nn.CrossEntropyLoss()
    optmizer= optim.Adam(net.parameters(),lr= lr)
    Train(epochs, train_loader, val_loader, criterion, optmizer, device)
