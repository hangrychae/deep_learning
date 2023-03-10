# -*- coding: utf-8 -*-
"""Copy of last

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ZFrqQaNORWwZgUEFjm1cLIU2AeeTWBoP
"""

import os
import csv
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
import pandas as pd

#!pip install --upgrade --force-reinstall --no-deps kaggle

from google.colab import drive
drive.mount("/content/gdrive", force_remount=True)

!cp /content/gdrive/MyDrive/Colab\ Notebooks/hw1p2_student_data.zip /content
!cp /content/gdrive/MyDrive/Colab\ Notebooks/train_filenames_subset_8192_v2.csv /content
!cp /content/gdrive/MyDrive/Colab\ Notebooks/test_order.csv /content

!unzip /content/hw1p2_student_data.zip

class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        # TODO: Please try different architectures
        in_size = (1+2*32)*13
        layers = [            
            nn.Linear(in_size, 2048),
            nn.ReLU(),
            nn.Linear(2048, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(4096, 1024),
            nn.LeakyReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.Softplus(),
            nn.Dropout(p=0.1),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.BatchNorm1d(256)
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 40)
        ]
        self.laysers = nn.Sequential(*layers)

    def init_weight(self, model):
        if type(model) == nn.Linear:
          nn.init.kaiming_uniform_(model.weight, mode='fan_in', nonlinearity='relu')
          #nn.init.xavier_normal_(model.weight, gain=1.0)

    def forward(self, A0):
        x = self.laysers(A0)
        return x

class LibriTrain(torch.utils.data.Dataset):
    def __init__(self, data_path, sample=20000, shuffle=True, partition="dev-clean", csvpath=None):
        # sample represent how many npy files will be preloaded for one __getitem__ call
        self.sample = sample 
        
        self.X_dir = data_path + "/" + partition + "/mfcc/"
        self.Y_dir = data_path + "/" + partition +"/transcript/"

        self.X_names = os.listdir(self.X_dir)
        self.Y_names = os.listdir(self.Y_dir)
 
        # using a small part of the dataset to debug
        if csvpath:
          subset = self.parse_csv(csvpath)
          self.X_names = [i for i in self.X_names if i in subset]
          self.Y_names = [i for i in self.Y_names if i in subset]              
          
        if shuffle == True:
          XY_names = list(zip(self.X_names, self.Y_names))
          random.shuffle(XY_names)
          self.X_names, self.Y_names = zip(*XY_names)
          
        assert(len(self.X_names) == len(self.Y_names))
        self.length = len(self.X_names)
        
        self.PHONEMES = [
            'SIL',   'AA',    'AE',    'AH',    'AO',    'AW',    'AY',  
            'B',     'CH',    'D',     'DH',    'EH',    'ER',    'EY',
            'F',     'G',     'HH',    'IH',    'IY',    'JH',    'K',
            'L',     'M',     'N',     'NG',    'OW',    'OY',    'P',
            'R',     'S',     'SH',    'T',     'TH',    'UH',    'UW',
            'V',     'W',     'Y',     'Z',     'ZH',    '<sos>', '<eos>']
      
    @staticmethod
    def parse_csv(filepath):
        subset = []
        with open(filepath) as f:
            f_csv = csv.reader(f)
            for row in f_csv:
                subset.append(row[1])
        return subset[1:]

    def __len__(self):
        return int(np.ceil(self.length / self.sample))
        
    def __getitem__(self, i):
        sample_range = range(i*self.sample, min((i+1)*self.sample, self.length))
        
        X, Y = [], []
        for j in sample_range:
            X_path = self.X_dir + self.X_names[j]
            Y_path = self.Y_dir + self.Y_names[j]
            
            label = [self.PHONEMES.index(yy) for yy in np.load(Y_path)][1:-1]

            X_data = np.load(X_path)
            X_data = (X_data - X_data.mean(axis=0))/X_data.std(axis=0)
            X.append(X_data)
            Y.append(np.array(label))
            
        X, Y = np.concatenate(X), np.concatenate(Y)
        return X, Y
    
class LibriItemsTrain(torch.utils.data.Dataset):
    def __init__(self, X, Y, context = 0):
        assert(X.shape[0] == Y.shape[0])
        
        self.length  = X.shape[0]
        self.context = context

        if context == 0:
          self.X, self.Y = X, Y
        else:
          #X = np.pad(X, ((context,context), (0,0)), 'constant', constant_values=(0,0))
          #self.X, self.Y = X, Y
          # TODO: self.X, self.Y = ... 
          X = np.pad(X, ((context,context), (0,0)), 'constant', constant_values=(0,0))
          self.X, self.Y = X, Y
     


    def __len__(self):
        return self.length
        
    def __getitem__(self, i):
        if self.context == 0:
            xx = self.X[i].flatten()
            yy = self.Y[i]
        else:
            xx = self.X[i:(i + 2*self.context + 1)].flatten()
            yy = self.Y[i]
            # TODO xx, yy = ...
            pass
        return xx, yy



class LibriTest(torch.utils.data.Dataset):
    def __init__(self, data_path, sample=20000, shuffle=True, partition="dev-clean", csvpath=None):
        # sample represent how many npy files will be preloaded for one __getitem__ call
        self.sample = sample 
        
        self.X_dir = data_path + "/" + partition + "/mfcc/"
        self.X_names = os.listdir(self.X_dir)


        # using a small part of the dataset to debug
        if csvpath:
          self.X_names = list(pd.read_csv(csvpath).file)
        
        if shuffle == True:
          X_names = list(self.X_names)
          random.shuffle(X_names)
          self.X_names= X_names
        
        self.length = len(self.X_names)
        
        self.PHONEMES = [
            'SIL',   'AA',    'AE',    'AH',    'AO',    'AW',    'AY',  
            'B',     'CH',    'D',     'DH',    'EH',    'ER',    'EY',
            'F',     'G',     'HH',    'IH',    'IY',    'JH',    'K',
            'L',     'M',     'N',     'NG',    'OW',    'OY',    'P',
            'R',     'S',     'SH',    'T',     'TH',    'UH',    'UW',
            'V',     'W',     'Y',     'Z',     'ZH',    '<sos>', '<eos>']
      
      
    def __len__(self):
        return int(np.ceil(self.length / self.sample))
        
    def __getitem__(self, i):
        sample_range = range(i*self.sample, min((i+1)*self.sample, self.length))
        
        X = []
        for j in sample_range:
            X_path = self.X_dir + self.X_names[j]

            X_data = np.load(X_path)
            X_data = (X_data - X_data.mean(axis=0))/X_data.std(axis=0)
            X.append(X_data)
  
        X = np.concatenate(X)
        return X
    
class LibriItemsTest(torch.utils.data.Dataset):
    def __init__(self, X, context = 0):

        self.length  = X.shape[0]
        self.context = context

        if context == 0:
            self.X = X
        else:
            X = np.pad(X, ((context,context), (0,0)), 'constant', constant_values=(0,0))
            # TODO: self.X, self.Y = ... 
            self.X = X 
        
    def __len__(self):
        return self.length
        
    def __getitem__(self, i):
        if self.context == 0:
            xx = self.X[i].flatten()
        else:
            xx = self.X[i:(i + 2*self.context + 1)].flatten()
            # TODO xx, yy = ...
        return xx

def train(args, model, device, train_samples, optimizer, criterion, epoch, scheduler):
    model.train()
    for i in range(len(train_samples)):
        X, Y = train_samples[i]
        train_items = LibriItemsTrain(X, Y, context=args['context'])
        train_loader = torch.utils.data.DataLoader(train_items, batch_size=args['batch_size'], shuffle=True)

        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.float().to(device)
            target = target.long().to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % args['log_interval'] == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
        scheduler.step(loss)


def test(args, model, device, dev_samples):
    model.eval()
    pred_y_list = []
    true_y_list = []
    with torch.no_grad():
        for i in range(len(dev_samples)):
          X = dev_samples[i]

          # use this later for test
          test_items = LibriItemsTest(X, context=args['context'])
          
          test_loader = torch.utils.data.DataLoader(test_items, batch_size=args['batch_size'], shuffle=False)

          for data in test_loader:
              data = data.float().to(device)           
                
              output = model(data)
              pred_y = torch.argmax(output, axis=1)

              pred_y_list.extend(pred_y.tolist())

    
    return pred_y_list




def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Network().to(device)
    model.apply(model.init_weight)
    optimizer = optim.Adam(model.parameters(), lr=args['lr'])
    criterion = torch.nn.CrossEntropyLoss()
    scheduler1 = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, threshold=0.00001) 
    # If you want to use full Dataset, please pass None to csvpath
    train_samples = LibriTrain(data_path = args['LIBRI_PATH'], shuffle=True, partition="train-clean-100")
    test_samples = LibriTest(data_path = args['LIBRI_PATH'], shuffle=False, partition="test-clean", csvpath = "/content/test_order.csv")
    dev_samples = LibriTrain(data_path = args['LIBRI_PATH'], shuffle=False, partition="dev-clean")
  
    test_result = []
    for epoch in range(1, args['epoch'] + 1):

        train(args, model, device, train_samples, optimizer, criterion, epoch, scheduler1)
        test_result = test(args, model, device, test_samples)
        #test_result = test(args, model, device, dev_samples)
        #print('Dev accuracy ', test_result)
    result = pd.DataFrame()
    result['id'] = np.array(range(len(test_result)))
    result['label'] = np.array(test_result)
    result.to_csv("early_submission.csv", index = False)


if __name__ == '__main__':
    args = {
        'batch_size': 2048,
        'context': 32,
        'log_interval': 200,
        'LIBRI_PATH': '/content',
        'lr': 0.003,
        'epoch': 10
    }
    main(args)