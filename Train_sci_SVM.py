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
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

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
        '--order',
        type=int,
        default=2,
        help='order of polynomial (default: 2)',
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
        '--kernel',
        type=str,
        default='rbf',
        help='kernel function for SVM (rbf, linear, polynomial) (default: rbf)',
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
        '--model',
        type=str,
        default='svm',
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

    sector = args.sector
    laps = args.laps
    c = args.out_ch

    files = glob.glob(f'D:\\IS2_topo_DL\\data\\Data_{sector}_*.pkl')
    xx, yy, inputs, outputs = read_grid_input(files, c)
    
    ann_input, ann_output = make_mlp_input(inputs, outputs, laps = laps)

    train_input, val_input, train_output, val_output = train_test_split(ann_input, ann_output, test_size=0.3, random_state=42)
    
    # train_input = torch.tensor(train_input, dtype=torch.float32)
    # train_output = torch.tensor(train_output, dtype=torch.float32)
    # val_input = torch.tensor(val_input, dtype=torch.float32)
    # val_output = torch.tensor(val_output, dtype=torch.float32)

    # train_dataset = TensorDataset(train_input, train_output)
    # train_loader = DataLoader(train_dataset, batch_size=batch_size)
    # val_dataset = TensorDataset(val_input, val_output)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size)
    print(train_input.shape)
    n_samples, in_channels = train_input.shape
    _, out_channels = train_output.shape
    print(f"##### TRAINING DATA IS PREPARED (Samples: {n_samples}; model: {args.model}) #####")

    if args.model == "lr":
        model = linear_model.LinearRegression()
        poly = PolynomialFeatures(degree = args.order)
        train_input = poly.fit_transform(train_input)
        val_input = poly.fit_transform(val_input)
        model_name = f"torch_{sector}_c{c}_lap{laps}_{args.model}{args.order}"
    elif args.model == "svm":
        model = SVR(kernel=args.kernel) # "rbf", "linear", "Polynomial"
        train_output = train_output[:, 0]
        val_output = val_output[:, 0]        
        model_name = f"torch_{sector}_c{c}_lap{laps}_{args.model}_{args.kernel}"
        
    print(model_name)

    model.fit(train_input, train_output)

    train_pred = model.predict(train_input)
    train_loss = RMSE(train_output, train_pred)
    train_R = corr(train_output, train_pred)

    val_pred = model.predict(val_input)
    val_loss = RMSE(val_output, val_pred)
    val_R = corr(val_output, val_pred)

    print('>> Train RMSE: {0:.4f}; R: {1:.4f}'.format(train_loss, train_R))
    print('>> Val   RMSE: {0:.4f}; R: {1:.4f}'.format(val_loss, val_R))

    with open(f'{model_dir}/{model_name}.pkl','wb') as f:
        pickle.dump(model, f)

if __name__ == '__main__':
    main()