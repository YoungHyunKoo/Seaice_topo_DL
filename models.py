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

import pickle

import torch
    
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import dgl
from dgl.data import DGLDataset

def read_grid_input(files, c):
    first = True
    for file in tqdm(files):
    
        with open(file, 'rb') as f:
            [xx, yy, input0, output0] = pickle.load(f)
            input0 = input0[12:] # From April
            output0 = output0[12:] # From April
    
        if first:
            inputs = input0
            outputs = output0
            first = False
        else:
            inputs = np.concatenate((inputs, input0), axis = 0)
            outputs = np.concatenate((outputs, output0), axis = 0)

    inputs = inputs[:,:9]
    outputs = outputs[:,c:c+1]
    print("Grid files are read!")
    
    return xx, yy, inputs, outputs

def normalize_input(inputs, c):
    vmax = [+20, +20, 1, 280, +20, +20, 30, +0.5, 0.7, 1]
    vmin = [-20, -20, 0, 230, -20, -20,  0, -0.5, 0.0, 0]
    if vmin[c] < 0:
        norm_inputs = 2*(inputs - vmin[c]) / (vmax[c] - vmin[c]) - 1
    else:
        norm_inputs = (inputs - vmin[c]) / (vmax[c] - vmin[c])
    
    return norm_inputs

def regularize(norm_data):
    norm_data[norm_data > 1] = 1
    norm_data[norm_data < -1] = -1
    return norm_data

def make_mlp_grid(inputs, laps = 4):
    
    # Input & output should be entire images for CNN
    n_samples, var_ip, row, col = np.shape(inputs)
    
    first = True
    for n in range(0, n_samples-laps):
    
        ann_input0 = np.zeros([row*col, var_ip * laps])
        
        for v in range(0, var_ip):
            for i in range(0, laps):
                ann_input0[:, v*laps+i] = normalize_input(inputs[n+i, v].flatten(), v)
    
        if first:
            ann_input = ann_input0
            first = False
        else:
            ann_input = np.concatenate((ann_input, ann_input0), axis = 0)
        
    return ann_input

def make_mlp_input(inputs, outputs, laps = 4):
    
    # Input & output should be entire images for CNN
    n_samples, var_ip, row, col = np.shape(inputs)
    _, var_op, _, _ = np.shape(outputs)

    mask = ~np.isnan(inputs).any(axis=(0,1))
    
    first = True
    for n in range(0, n_samples-laps):
        sic = inputs[n+laps, 2, :, :]
        valid = np.where((sic > 0.8) & (outputs[n+laps, 0] > 0) & (outputs[n+laps, 0] <= 1) & (mask))
    
        n_valid = valid[0].shape[0]
    
        ann_input0 = np.zeros([n_valid, var_ip * laps])
        ann_output0 = np.zeros([n_valid, var_op])
        
        for v in range(0, var_ip):
            for i in range(0, laps):
                ann_input0[:, v*laps+i] = regularize(normalize_input(inputs[n+i, v][valid], v))
        for v in range(0, var_op):
            ann_output0[:, v] = regularize(outputs[n+laps, v][valid])
    
        if first:
            ann_input = ann_input0
            ann_output = ann_output0
            first = False
        else:
            ann_input = np.concatenate((ann_input, ann_input0), axis = 0)
            ann_output = np.concatenate((ann_output, ann_output0), axis = 0)

    print(f"MLP dataset is ready for lap time {laps}")
    return ann_input, ann_output

def make_gnn_input(inputs, outputs, laps = 4):
    # Input & output should be entire images for CNN
    n_samples, var_ip, row, col = np.shape(inputs)
    _, var_op, _, _ = np.shape(outputs)

    mask = ~np.isnan(inputs).any(axis=(0,1))
    
    first = True
    for n in range(0, n_samples-laps):
        sic = inputs[n+laps, 2, :, :]
        valid = np.where((sic > 0.9) & (outputs[n+laps, 0] > 0) & (outputs[n+laps, 0] <= 1) & (mask))
    
        n_valid = valid[0].shape[0]
    
        ann_input0 = np.zeros([n_valid, var_ip * laps])
        ann_output0 = np.zeros([n_valid, var_op])
        
        for v in range(0, var_ip):
            for i in range(0, laps):
                ann_input0[:, v*laps+i] = normalize_input(inputs[n+i, v][valid], v)
        for v in range(0, var_op):
            ann_output0[:, v] = outputs[n+laps, v][valid]
    
        if first:
            ann_input = ann_input0
            ann_output = ann_output0
            first = False
        else:
            ann_input = np.concatenate((ann_input, ann_input0), axis = 0)
            ann_output = np.concatenate((ann_output, ann_output0), axis = 0)

        
    return ann_input, ann_output

def make_cnn_input(inputs, outputs, laps = 4):
    # Input & output should be entire images for CNN
    n_samples, var_ip, row, col = np.shape(inputs)
    _, var_op, _, _ = np.shape(outputs)

    inputs[np.isnan(inputs)] = 0
    # outputs[np.isnan(outputs)] = 0
    
    first = True
    for n in tqdm(range(0, n_samples-laps)):
        # sic = inputs[n+laps, 2, :, :]
        # valid = np.where((sic > 0.8) & (outputs[n+laps, 0] > 0) & (outputs[n+laps, 0] <= 1) & (mask))
    
        # n_valid = valid[0].shape[0]
    
        ann_input0 = np.zeros([1, var_ip * laps, row, col])
        ann_output0 = np.zeros([1, var_op, row, col])
        
        for v in range(0, var_ip):
            for i in range(0, laps):
                ann_input0[0, v*laps+i] = normalize_input(inputs[n+i, v], v)
        for v in range(0, var_op):
            ann_output0[0, v] = outputs[n+laps, v]
    
        if first:
            ann_input = ann_input0
            ann_output = ann_output0
            first = False
        else:
            ann_input = np.concatenate((ann_input, ann_input0), axis = 0)
            ann_output = np.concatenate((ann_output, ann_output0), axis = 0)
            
    print(f"CNN dataset is ready for lap time {laps}")
    return ann_input, ann_output

## Dataset for train ===================================
class gnn_input(DGLDataset):
    def __init__(self, url=None, laps=1, w=1):
        super(gnn_input, self).__init__(name="graph", url=url, laps=laps, w=w)
        
    def process(self):
        self.graphs = []
        files = self.url
        laps = self.laps
        w = self.w
        n_nodes = (2*w+1)**2
        inputs, outputs = read_grid_input(files)
        
        # Input & output should be entire images for CNN
        n_samples, var_ip, row, col = np.shape(inputs)
        _, var_op, _, _ = np.shape(outputs)
        
        first = True

        if first:
            src = []
            dst = []
            weight = []
            slope = []

            for i in range(0, n_sample):        
                p1, p2 = np.where(elements == i)
                connect = []

                for p in p1:
                    for k in elements[p]:
                        if (k != i) and (k not in connect):
                            connect.append(k)
                            dist = ((xc[i]-xc[k])**2+(yc[i]-yc[k])**2)**0.5                                
                            weight.append(np.exp(-(dist/1000)))
                            slope.append([np.exp(-(dist/1000)), (base[0,i]-base[0,k])/dist, (surface[0,i]-surface[0,k])/dist,
                                         (vx[0,i]-vx[0,k])/dist, (vy[0,i]-vy[0,k])/dist]) 
                            src.append(int(i))
                            dst.append(int(k))

            src = torch.tensor(src)
            dst = torch.tensor(dst)
            weight = torch.tensor(weight)
            slope = torch.arctan(torch.tensor(slope))
        
        for n in range(0, n_samples-laps):
            sic = inputs[n+laps, 2, :, :]
            valid = np.where((sic > 0.8) & (outputs[n+laps, 0] > 0))
        
            n_valid = valid[0].shape[0]
        
            ann_input0 = np.zeros([n_valid, var_ip * laps])
            ann_output0 = np.zeros([n_valid, var_op])

            for k in np.range(n_valid):
                
                part = np.zeros(n_nodes, var_ip)
                i = valid[0][k]
                j = valid[1][k]

            g = dgl.graph((src, dst), num_nodes=(2*w+1)**2)
            g.ndata['feat'] = inputs
            g.ndata['label'] = outputs
            g.edata['weight'] = weight
            g.edata['slope'] = slope

            
                

            
            for v in range(0, var_ip):
                for i in range(0, laps):
                    ann_input0[:, v*laps+i] = normalize_input(inputs[n+i, v][valid], v)
            for v in range(0, var_op):
                ann_output0[:, v] = outputs[n+laps, v][valid]
        
            if first:
                ann_input = ann_input0
                ann_output = ann_output0
                first = False
            else:
                ann_input = np.concatenate((ann_input, ann_input0), axis = 0)
                ann_output = np.concatenate((ann_output, ann_output0), axis = 0)


        
        
        # # Region filtering
        # filename = f'D:\\ISSM\\Helheim\\Helheim_r100_030.mat'
        # test = sio.loadmat(filename)
        # mask = test['S'][0][0][11][0]

        first = True
        # "READING GRAPH DATA..."
        for filename in tqdm(files[:]):
            mesh = int(filename.split("_m")[1][:3])
            rate = int(filename.split("_r")[1][:3])
            test = sio.loadmat(filename)

            xc = test['S'][0][0][0]
            yc = test['S'][0][0][1]
            elements = test['S'][0][0][2]-1
            smb = test['S'][0][0][3]
            vx = test['S'][0][0][4]
            vy = test['S'][0][0][5]
            vel = test['S'][0][0][6]
            surface = test['S'][0][0][7]
            base = test['S'][0][0][8]
            H = test['S'][0][0][9]
            f = test['S'][0][0][10]
            # mask = test['S'][0][0][11]
            # ice = np.zeros(mask.shape) # Negative: ice; Positive: no-ice
            # ice[mask > 0] = 0.5 # ice = 0; no-ice = 1
            # ice = np.where(mask < 0, mask / 1000000, mask/10000)

            n_year, n_sample = H.shape
            
            if first:
                mesh0 = mesh
            elif mesh0 != mesh:
                first = True
                mesh0 = mesh

            if first:
                src = []
                dst = []
                weight = []
                slope = []

                for i in range(0, n_sample):        
                    p1, p2 = np.where(elements == i)
                    connect = []

                    for p in p1:
                        for k in elements[p]:
                            if (k != i) and (k not in connect):
                                connect.append(k)
                                dist = ((xc[i]-xc[k])**2+(yc[i]-yc[k])**2)**0.5                                
                                weight.append(np.exp(-(dist/1000)))
                                slope.append([np.exp(-(dist/1000)), (base[0,i]-base[0,k])/dist, (surface[0,i]-surface[0,k])/dist,
                                             (vx[0,i]-vx[0,k])/dist, (vy[0,i]-vy[0,k])/dist]) 
                                src.append(int(i))
                                dst.append(int(k))

                src = torch.tensor(src)
                dst = torch.tensor(dst)
                weight = torch.tensor(weight)
                slope = torch.arctan(torch.tensor(slope))
                first = False
            else:
                pass                    

            for t in range(0, n_year):
                # INPUT: x/y coordinates, melting rate, time, SMB, Vx0, Vy0, Surface0, Base0, Thickness0, Floating0
                inputs = torch.zeros([n_sample, 12])
                # OUTPUT: Vx, Vy, Vel, Surface, Thickness, Floating
                outputs = torch.zeros([n_sample, 6])

                ## INPUTS ================================================
                inputs[:, 0] = torch.tensor((xc[:, 0]-xc.min())/10000) # torch.tensor(xc[0, :]/10000) # torch.tensor((xc[:, 0]-xc.min())/(xc.max()-xc.min())) # X coordinate
                inputs[:, 1] = torch.tensor((yc[:, 0]-yc.min())/10000) # torch.tensor(yc[0, :]/10000) # torch.tensor((yc[:, 0]-yc.min())/(yc.max()-yc.min())) # Y coordinate
                inputs[:, 2] = torch.where(torch.tensor(f[0, :]) < 0, rate/100, 0) # Melting rate (0-100)
                inputs[:, 3] = torch.tensor(t/n_year) # Year
                inputs[:, 4] = torch.tensor(smb[t, :]/20) # Surface mass balance
                inputs[:, 5] = torch.tensor(vx[0, :]/10000) # Initial Vx
                inputs[:, 6] = torch.tensor(vy[0, :]/10000) # Initial Vx
                inputs[:, 7] = torch.tensor(vel[0, :]/10000) # Initial Vel
                inputs[:, 8] = torch.tensor(surface[0, :]/5000) # Initial surface elevation
                inputs[:, 9] = torch.tensor(base[0, :]/5000) # Initial base elevation
                inputs[:, 10] = torch.tensor(H[0, :]/5000) # Initial ice thickness
                inputs[:, 11] = torch.tensor(f[0, :]/5000) # Initial floating part
                # inputs[:, 11] = torch.tensor(ice[0, :]) # Initial ice mask

                ## OUTPUTS ===============================================
                outputs[:, 0] = torch.tensor(vx[t, :]/10000) # Initial Vx
                outputs[:, 1] =  torch.tensor(vy[t, :]/10000) # Initial Vx
                outputs[:, 2] = torch.tensor(vel[t, :]/10000) # Initial surface elevation
                outputs[:, 3] = torch.tensor(surface[t, :]/5000) # Initial base elevation
                outputs[:, 4] = torch.tensor(H[t, :]/5000) # Initial ice thickness
                outputs[:, 5] = torch.tensor(f[t, :]/5000) # Initial floating part 
                # outputs[:, 5] = torch.tensor(ice[t, :]) # Initial floating part 

                g = dgl.graph((src, dst), num_nodes=n_sample)
                g.ndata['feat'] = inputs
                g.ndata['label'] = outputs
                g.edata['weight'] = weight
                g.edata['slope'] = slope

                self.graphs.append(g)
        
    def __getitem__(self, i):
        return self.graphs[i]
    
    def __len__(self):
        return len(self.graphs)


class MLP(nn.Module):    
    def __init__(self, ch_input, ch_output, features = 128, hidden_layers = 4):
        super(MLP, self).__init__()
        
        modules = [nn.Linear(ch_input, features)]
        for i in range(hidden_layers):
            modules.append(nn.Linear(features, features))
            modules.append(nn.LeakyReLU())
        modules.append(nn.Linear(features, ch_output))
        modules.append(nn.ReLU())
        self.lin = nn.Sequential(*modules)

    def forward(self, in_feat):

        x = self.lin(in_feat)
        
        return x

