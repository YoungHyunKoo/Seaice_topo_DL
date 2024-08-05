### PREDICT ONLY SEA ICE U & V

# Ignore warning
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import glob, os
import matplotlib.pyplot as plt
import numpy as np
import geopandas
import datetime as dt
from datetime import datetime
import pyproj

from tqdm import tqdm
import time

import pickle

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import TensorDataset, DataLoader, Dataset, random_split

from functions import *
from models import *


import argparse
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def parse_args() -> argparse.Namespace:
    """Get cmd line args."""
    
    # General settings
    parser = argparse.ArgumentParser(description='PyTorch Example')
    parser.add_argument(
        '--data-dir',
        type=str,
        default='D:\\IS2_topo_DL\\data\\',
        metavar='D',
        help='directory to download dataset to',
    )   
    parser.add_argument(
        '--model-dir',
        default='D:\\IS2_topo_DL\\model\\',
        help='Model directory',
    )
    parser.add_argument(
        '--cuda',
        type=bool,
        default=True,
        help='disables CUDA training',
    )
    
    # Training settings
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        metavar='N',
        help='input batch size for training (default: 16)',
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        metavar='N',
        help='number of epochs to train (default: 100)',
    )
    parser.add_argument(
        '--base-lr',
        type=float,
        default=0.001,
        metavar='LR',
        help='base learning rate (default: 0.01)',
    )
    parser.add_argument(
        '--sector',
        type=str,
        default='Ross',
        help='target sector in the Southern Ocean',
    )
    parser.add_argument(
        '--laps',
        type=int,
        default=4,
        help='The number of previous weeks considered in the prediction',
    )
    parser.add_argument(
        '--out-ch',
        type=int,
        default=1,
        help='Output channel of ICESat-2 (0: modal fb, 1: std fb)',
    )
    parser.add_argument(
        '--features',
        type=int,
        default=64,
        help='Number of the features in hidden layers',
    )
    parser.add_argument(
        '--hl',
        type=int,
        default=1,
        help='Number of hidden layers',
    )
    parser.add_argument(
        '--model',
        type=str,
        default='lstm',
        help='model name',
    )
    parser.add_argument(
        '--local_rank',
        type=int,
        default=0,
        help='local rank for distributed training',
    )

    args = parser.parse_args()
    if 'LOCAL_RANK' in os.environ:
        args.local_rank = int(os.environ['LOCAL_RANK'])
    args.cuda = args.cuda and torch.cuda.is_available()

    return args

    
##########################################################################################
def weights_init(m):
    torch.nn.init.xavier_uniform_(m.weight)
    if m.bias:
        torch.nn.init.zeros_(m.bias)
            
def main() -> None:
    
    ## Train parameters ##
    args = parse_args()
    
    data_path = args.data_dir
    model_dir = args.model_dir

    n_epochs = args.epochs
    batch_size = args.batch_size  # size of each batch
    lr = args.base_lr
    sector = args.sector
    laps = args.laps
    c = args.out_ch

    files = glob.glob(f'D:\\IS2_topo_DL\\data\\Data_{sector}_*.pkl')
    xx, yy, inputs, outputs = read_grid_input(files, c)
    
    ann_input, ann_output = make_rnn_input(inputs, outputs, laps = laps)

    train_input, val_input, train_output, val_output = train_test_split(ann_input, ann_output, test_size=0.3, random_state=42)
    
    train_input = torch.tensor(train_input, dtype=torch.float32)
    train_output = torch.tensor(train_output, dtype=torch.float32)
    val_input = torch.tensor(val_input, dtype=torch.float32)
    val_output = torch.tensor(val_output, dtype=torch.float32)

    train_dataset = TensorDataset(train_input, train_output)
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_dataset = TensorDataset(val_input, val_output)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    n_samples, _, in_channels = train_input.size()
    _, out_channels = train_output.size()
    print(f"##### TRAINING DATA IS PREPARED (Samples: {n_samples}; model: {args.model}) #####")

    features = args.features
    hidden_layers = args.hl
    net = LSTM(in_channels, out_channels, laps=laps, features=features, hidden_layers = hidden_layers)
    model_name = f"torch_{sector}_c{c}_lap{laps}_{args.model}_h{hidden_layers}_f{features}"
    print(model_name)
    
    if args.cuda:
        device = torch.device('cuda')
        device_name = 'gpu'
        net = nn.DataParallel(net)     
    else:            
        device = torch.device('cpu')
        device_name = 'cpu'

    print(device)
    net.to(device)

    loss_fn = nn.MSELoss() # nn.L1Loss() #nn.CrossEntropyLoss()
    optimizer = Adam(net.parameters(), lr)
    scheduler = ExponentialLR(optimizer, gamma=0.95)

    history = {'loss': [], 'val_loss': [], 'time': []}

    total_params = sum(p.numel() for p in net.parameters())
    print(f"Number of parameters: {total_params}")
    
    t0 = time.time()
    for epoch in range(n_epochs):
        
        train_loss = 0.0
        train_count = 0
        
        net.train()
        
        for (data, target) in train_loader:
            data = data.to(device)
            target = target.to(device)            
            pred = net(data)

            loss = loss_fn(pred*100, target*100)
            train_loss += loss.cpu().item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_count += 1            
        scheduler.step()
        
        ##### VALIDATION ######################
        net.eval()
        val_loss = 0
        val_count = 0
        for (data, target) in val_loader:
            data = data.to(device)
            target = target.to(device)
            pred = net(data)
            loss = loss_fn(pred*100, target*100)
            val_loss += loss.cpu().item()
            val_count += 1

        t1 = time.time() - t0
        history['loss'].append(train_loss/train_count)
        history['val_loss'].append(val_loss/val_count)
        history['time'].append(t1)
        
        print('Epoch {0} >> Train loss: {1:.4f}; Val loss: {2:.4f} [{3:.2f} sec]'.format(str(epoch).zfill(3),
                                                                                         train_loss/train_count, val_loss/val_count, t1))

        if (epoch > 20) & (np.nanmin(history['val_loss'][-5:]) > np.nanmean(history['val_loss'][-10:-5])):
            break
                
    torch.save(net.state_dict(), f'{model_dir}/{model_name}.pth')

    with open(f'{model_dir}/history_{model_name}.pkl', 'wb') as file:
        pickle.dump(history, file)

if __name__ == '__main__':
    main()