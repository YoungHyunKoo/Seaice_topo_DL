 ### PREDICT ONLY SEA ICE U & V

# Ignore warning
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import math
from datetime import datetime
from tqdm import tqdm
import time
import pickle

import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR

import torch.distributed as dist
from torch.utils import collect_env
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
# from torch_geometric.loader import DataLoader
 
# from torch.utils.tensorboard import SummaryWriter

from models import *
from functions import *

import argparse
import os    
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    filepath: str,
) -> None:
    """Save model checkpoint."""
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(state, filepath)


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
    
    try:
        # Set automatically by torch distributed launch
        parser.add_argument(
            '--local-rank',
            type=int,
            default=0,
            help='local rank for distributed training',
        )
    except:
        pass
    
    args = parser.parse_args()
    if 'LOCAL_RANK' in os.environ:
        args.local_rank = int(os.environ['LOCAL_RANK'])
    args.cuda = args.cuda and torch.cuda.is_available()

    return args
                

##########################################################################################

import torch.distributed as dist

from dgl import save_graphs, load_graphs
from dgl.data.utils import makedirs, save_info, load_info
import dgl
from dgl.data import DGLDataset

###############################################################################
# Data Loader Preparation
# -----------------------
#
# We split the dataset into training, validation and test subsets. In dataset
# splitting, we need to use a same random seed across processes to ensure a
# same split. We follow the common practice to train with multiple GPUs and
# evaluate with a single GPU, thus only set `use_ddp` to True in the
# :func:`~dgl.dataloading.pytorch.GraphDataLoader` for the training set, where 
# `ddp` stands for :func:`~torch.nn.parallel.DistributedDataParallel`.
#

from dgl.data import split_dataset
from dgl.dataloading import GraphDataLoader

def get_dataloaders(dataset, batch_size=32, shuffle = False, frac_list = [0.7, 0.3, 0.0]):
    # Use a 80:10:10 train-val-test split
    train_set, val_set, test_set = split_dataset(dataset,
                                                 frac_list=frac_list,
                                                 shuffle=True,
                                                 random_state=42)
    train_loader = GraphDataLoader(train_set, use_ddp=False, batch_size=batch_size, shuffle=shuffle)
    val_loader = GraphDataLoader(val_set, use_ddp=False, batch_size=batch_size, shuffle=shuffle)
    # test_loader = GraphDataLoader(test_set, batch_size=batch_size)

    return train_set, val_set

###############################################################################
# To ensure same initial model parameters across processes, we need to set the
# same random seed before model initialization. Once we construct a model
# instance, we wrap it with :func:`~torch.nn.parallel.DistributedDataParallel`.
#

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Adam

def main():
    
    now = datetime.now()
    args = parse_args()

    current_time = now.strftime("%Y-%m-%d %H:%M:%S")
    print(f"Current Time = {current_time} (GPU {args.local_rank})")
    
    model_dir = args.model_dir

    n_epochs = args.epochs
    batch_size = args.batch_size  # size of each batch
    lr = args.base_lr
    
    if args.cuda:
        device = torch.device('cuda')
        device_name = 'gpu'
        net = nn.DataParallel(net)     
    else:            
        device = torch.device('cpu')
        device_name = 'cpu'

    sector = args.sector
    dataset = gnn_input(url = glob.glob(f'D:\\IS2_topo_DL\\data\\Data_{sector}_*.pkl'))
    
    train_set, val_set = split_dataset(dataset, frac_list=[0.7, 0.3], shuffle=True, random_state=42)
    train_loader = GraphDataLoader(train_set, use_ddp=False, batch_size=batch_size, shuffle=True)
    val_loader = GraphDataLoader(val_set, use_ddp=False, batch_size=batch_size, shuffle=True)
    
    n_nodes = val_set[0][0].num_nodes()
    in_channels = val_set[0][0].ndata['feat'].shape[1]
    out_chanels = 1
    
    if args.local_rank == 0:
        print(f"## NODE: {n_nodes}; IN: {in_channels}; OUT: {out_channels}")
        print(f"## Total: {len(train_set)}; Val: {len(val_set)}")
        print("######## TRAINING/VALIDATION DATA IS PREPARED ########")
    
    features = args.features
    hidden_layers = args.hl
    if args.model_type == "gcn":
        model = GCN(in_channels, out_channels, features, hidden_layers)  # Graph convolutional network    

    model_name = f"torch_{sector}_c{c}_lap{laps}_{args.model}_h{hidden_layers}_f{features}"
    
    model.to(device)
    
    criterion = nn.MSELoss() #regional_loss() #nn.MSELoss() #nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr)
    scheduler = ExponentialLR(optimizer, gamma=0.95)
    
    total_params = sum(p.numel() for p in model.parameters())
    if args.local_rank == 0:
        print(model_name)
        print(f"MODEL: {args.model_type}; Number of parameters: {total_params}")
    
    history = {'loss': [], 'val_loss': [], 'time': []}
    ti = time.time()
    
    for epoch in range(n_epochs):
        t0 = time.time()
        model.train()
        # The line below ensures all processes use a different
        # random ordering in data loading for each epoch.
        
        ##### TRAIN ###########################
        train_loss = 0
        train_count = 0
        for bg, target in train_loader:
            bg = bg.to(device)
            feats = bg.ndata['feat']                
            pred = model(bg, feats)            
            loss = criterion(pred*100, target*100)
            train_loss += loss.cpu().item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_count += 1
        scheduler.step()
        
        ##### VALIDATION ######################
        val_loss = 0
        val_count = 0
        for bg, target in val_loader:
            bg = bg.to(device)
            feats = bg.ndata['feat']                
            pred = model(bg, feats)            
            loss = criterion(pred*100, target*100)                
            val_loss += loss.cpu().item()
            val_count += 1
            
        history['loss'].append(train_loss/train_count)
        history['val_loss'].append(val_loss/val_count)
        history['time'].append(time.time() - ti)
        
        t1 = time.time() - t0
        if args.local_rank == 0:
            if epoch % 1 == 0:            
                print('Epoch {0} >> Train loss: {1:.4f}; Val loss: {2:.4f} [{3:.2f} sec]'.format(str(epoch).zfill(3), train_loss/train_count, val_loss/val_count, t1))
            if epoch == n_epochs-1:
                print('Epoch {0} >> Train loss: {1:.4f}; Val loss: {2:.4f} [{3:.2f} sec]'.format(str(epoch).zfill(3), train_loss/train_count, val_loss/val_count, t1))
                
                torch.save(model.state_dict(), f'{model_dir}/{model_name}.pth')
                with open(f'{model_dir}/history_{model_name}.pkl', 'wb') as file:
                    pickle.dump(history, file)

###############################################################################
if __name__ == '__main__':
    main()

