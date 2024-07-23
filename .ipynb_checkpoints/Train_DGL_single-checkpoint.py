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

from DGL_model import *
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
        '--model-dir',
        default='../model',
        help='Model directory',
    )
    parser.add_argument(
        '--log-dir',
        default='./logs/torch_unet',
        help='TensorBoard/checkpoint directory',
    )
    parser.add_argument(
        '--checkpoint-format',
        default='checkpoint_unet_{epoch}.pth.tar',
        help='checkpoint file format',
    )
    parser.add_argument(
        '--checkpoint-freq',
        type=int,
        default=10,
        help='epochs between checkpoints',
    )
    parser.add_argument(
        '--no-cuda',
        # action='store_true',
        default=False,
        help='disables CUDA training',
    )    
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        metavar='S',
        help='random seed (default: 42)',
    )
    
    # Training settings
    parser.add_argument(
        '--batch-size',
        type=int,
        default=24,
        metavar='N',
        help='input batch size for training (default: 16)',
    )
    parser.add_argument(
        '--phy',
        type=str,
        default='nophy',
        help='filename of dataset',
    )
    parser.add_argument(
        '--data',
        type=str,
        default='mat',
        help='type of dataset (pkl or mat)',
    )
    parser.add_argument(
        '--out-ch',
        type=int,
        default=5,
        help='Number of output channels (6: include all or 3: u, v, h)',
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
        default=0.01,
        metavar='LR',
        help='base learning rate (default: 0.01)',
    )
    parser.add_argument(
        '--model-type',
        type=str,
        default="gcn",
        help='types of the neural network model (e.g. unet, cnn, fc)',
    )
    parser.add_argument(
        '--mesh',
        type=int,
        default=10000,
        help='meshsize of the finite element of ISSM model (select 5000, 10000, or 20000)',
    )
    
    parser.add_argument(
        '--backend',
        type=str,
        default='nccl',
        help='backend for distribute training (default: nccl)',
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
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    return args
                

##########################################################################################

import torch.distributed as dist

from dgl import save_graphs, load_graphs
from dgl.data.utils import makedirs, save_info, load_info
import dgl
from dgl.data import DGLDataset

class ISSM_train_dataset(DGLDataset):
    def __init__(self, filename):
        super().__init__(name="pig", url = filename)
        
    def process(self):
        glist, _ = load_graphs(self.url)
        self.graphs = glist
        
    def __getitem__(self, i):
        return self.graphs[i]
    
    def __len__(self):
        return len(self.graphs)
    
class ISSM_val_dataset(DGLDataset):
    def __init__(self, filename):
        super().__init__(name="pig", url = filename)
        
    def process(self):
        glist, _ = load_graphs(self.url)
        self.graphs = glist
        
    def __getitem__(self, i):
        return self.graphs[i]
    
    def __len__(self):
        return len(self.graphs)
    
class ISSM_test_dataset(DGLDataset):
    def __init__(self, filename):
        super().__init__(name="pig", url = filename)
        
    def process(self):
        glist, _ = load_graphs(self.url)
        self.graphs = glist
        
    def __getitem__(self, i):
        return self.graphs[i]
    
    def __len__(self):
        return len(self.graphs)

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

def get_dataloaders(dataset, seed, batch_size=32, shuffle = False, frac_list = [0.7, 0.3, 0.0]):
    # Use a 80:10:10 train-val-test split
    train_set, val_set, test_set = split_dataset(dataset,
                                                 frac_list=frac_list,
                                                 shuffle=True,
                                                 random_state=seed)
    train_loader = GraphDataLoader(train_set, use_ddp=False, batch_size=batch_size, shuffle=shuffle)
    # val_loader = GraphDataLoader(val_set, use_ddp=False, batch_size=batch_size, shuffle=shuffle)
    # test_loader = GraphDataLoader(test_set, batch_size=batch_size)

    return train_loader #, test_loader

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
    
    seed = args.seed
    
    # if args.cuda:
    #     torch.cuda.set_device(args.local_rank)
    #     torch.cuda.manual_seed(args.seed)
    
    model_dir = args.model_dir

    n_epochs = args.epochs
    batch_size = args.batch_size  # size of each batch
    lr = args.base_lr

    phy = args.phy ## PHYSICS OR NOT
    
    if args.no_cuda:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')
    
    torch.cuda.empty_cache()
    
    mesh = args.mesh
    
    if args.data == "mat":
        train_files, val_files, test_files = generate_list(region = "PIG")
        train_set = GNN_PIG_Dataset(train_files)
        val_set = GNN_PIG_Dataset(val_files)
        test_set = GNN_PIG_Dataset(test_files)
    elif args.data == "bin":
        train_set = ISSM_train_dataset(f"../data/DGL_train_dataset_{version}_m{mesh:05d}.bin")
        val_set = ISSM_val_dataset(f"../data/DGL_val_dataset_{version}_m{mesh:05d}.bin")
        test_set = ISSM_test_dataset(f"../data/DGL_test_dataset_{version}_m{mesh:05d}.bin")
    
    # train_loader = GraphDataLoader(train_set, use_ddp=True, batch_size=batch_size, shuffle=False)
    # val_loader = GraphDataLoader(val_set, batch_size=batch_size, shuffle=False)
    
    train_loader = get_dataloaders(train_set, seed, batch_size, True, frac_list = [0.7, 0.3, 0.0])
    val_loader = get_dataloaders(val_set, seed, batch_size, True, frac_list = [1.0, 0.0, 0.0])
    
    n_nodes = val_set[0].num_nodes()
    in_channels = val_set[0].ndata['feat'].shape[1] - 2 #-1
    edge_feat_size = val_set[0].edata['slope'].shape[1]
    if args.out_ch > 0:
        out_channels = args.out_ch
    else:
        out_channels = val_set[0].ndata['label'].shape[1]
    
    if args.local_rank == 0:
        print(f"## NODE: {n_nodes}; IN: {in_channels}; OUT: {out_channels}; EDGE FEATURES: {edge_feat_size}")
        print(f"## Total: {len(train_set)}; Train: {len(train_loader)*batch_size}; Val: {len(val_loader)*batch_size}")
        print("######## TRAINING/VALIDATION DATA IS PREPARED ########")
    
    hidden_channels = 128
    if args.model_type == "gcn":
        model = GCN(in_channels, out_channels, hidden_channels)  # Graph convolutional network    
    elif args.model_type == "mlp":
        model = MLP(in_channels, out_channels, hidden_channels)  # Fully connected network
    elif args.model_type == "gat":
        model = GAT(in_channels, out_channels, hidden_channels)  # Graph convolutional network 
    elif args.model_type == "egcn":
        model = EGCN(in_channels, out_channels-2, hidden_channels, edge_feat_size) # Equivariant Graph convolutional network
    elif args.model_type == "sage":
        model = SAGE(in_channels, out_channels, hidden_channels) # Equivariant Graph convolutional network
    elif args.model_type == "cheb":
        model = ChebGCN(in_channels, out_channels, hidden_channels)  # Equivariant Graph convolutional network
    else:
        print("Please put valid model name!!")
        # model = GCN(in_channels, out_channels, 128)  # Fully connected network
    
    model_name = f"torch_dgl_PIG_{args.model_type}_{n_nodes}_lr{lr}_ch{out_channels}"
    
    torch.manual_seed(seed)
    
    model.to(device)
    # if args.no_cuda:
    #     model = DistributedDataParallel(model)
    # else:
    #     model = DistributedDataParallel(model, device_ids=[args.local_rank])
    
    criterion = nn.MSELoss() #regional_loss() #nn.MSELoss() #nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr)
    scheduler = ExponentialLR(optimizer, gamma=0.99)
    
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
        for bg in train_loader:
            bg = bg.to(device)
            feats = bg.ndata['feat'][:, 2:]
            coord_feat = bg.ndata['feat'][:, :2]
            edge_feat = bg.edata['weight'].float() #.repeat(1, 2)
            if out_channels == 3:
                labels = bg.ndata['label'][:, [0, 1, 4]] # version 2
            elif out_channels == 2:
                labels = bg.ndata['label'][:, [2, 4]]
            elif out_channels == 4:
                labels = bg.ndata['label'][:, [0,1,4,5]]
            elif out_channels == 5:
                labels = bg.ndata['label'][:, [0,1,3,4,5]]
            else:
                labels = bg.ndata['label'][:, :]
                
            pred = model(bg, feats)

            if args.model_type[:4] == "egcn":
                # pred = model(bg, feats, coord_feat, edge_feat)
                labels = torch.cat([labels[:, :2], labels[:, 2:]], dim=1)
            
            loss = criterion(pred[:, :out_channels]*100, labels[:, :out_channels]*100)
            train_loss += loss.cpu().item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_count += 1
        scheduler.step()
        
        ##### VALIDATION ######################
        val_loss = 0
        val_count = 0
        for bg in val_loader:
            bg = bg.to(device)
            feats = bg.ndata['feat'][:, 2:]
            coord_feat = bg.ndata['feat'][:, :2]
            edge_feat = bg.edata['weight'].float() #.repeat(1, 2)
            if out_channels == 3:
                labels = bg.ndata['label'][:, [0,1,4]] # version 2
            elif out_channels == 2:
                labels = bg.ndata['label'][:, [2, 4]]
            elif out_channels == 4:
                labels = bg.ndata['label'][:, [0,1,4,5]]
            elif out_channels == 5:
                labels = bg.ndata['label'][:, [0,1,3,4,5]]
            else:
                labels = bg.ndata['label'][:, :]
            
            with torch.no_grad():
                
                pred = model(bg, feats)
                
                if args.model_type[:4] == "egcn":
                # pred = model(bg, feats, coord_feat, edge_feat)
                    labels = torch.cat([labels[:, :2], labels[:, 2:]], dim=1)
                # if args.model_type == "egcn":
                #     # pred = model(bg, feats, coord_feat, edge_feat)
                #     labels = torch.cat([labels, coord_feat], dim=1)
                # elif args.model_type == "egcn2":
                #     # pred = model(bg, feats, coord_feat, edge_feat)
                #     labels = torch.cat([labels, coord_feat], dim=1)                    


            loss = criterion(pred[:, :out_channels]*100, labels[:, :out_channels]*100)
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

