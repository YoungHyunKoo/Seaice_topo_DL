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
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
 
# from torch.utils.tensorboard import SummaryWriter

from torch_model import *

import argparse
import os    
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# try:
#     from torch.cuda.amp import GradScaler

#     TORCH_FP16 = True
# except ImportError:
#     TORCH_FP16 = False


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
        default='../data/', #'D:\\PINN\\data\\',
        metavar='D',
        help='directory to download dataset to',
    )
    parser.add_argument(
        '--data-file',
        type=str,
        default='train_cnn_2016_2022_v7.pkl',
        help='filename of dataset',
    )    
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
        '--date',
        type=int,
        default=2022,
        help='year to exclude during the training process',
    )
    parser.add_argument(
        '--sdate',
        type=int,
        default=2016,
        help='year to start training',
    )
    parser.add_argument(
        '--ratio',
        type=float,
        default=1.0,
        help='Ratio to include in training dataset',
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
        default=16,
        metavar='N',
        help='input batch size for training (default: 16)',
    )
    parser.add_argument(
        '--batches-per-allreduce',
        type=int,
        default=1,
        help='number of batches processed locally before '
        'executing allreduce across workers; it multiplies '
        'total batch size.',
    )
    parser.add_argument(
        '--val-batch-size',
        type=int,
        default=16,
        help='input batch size for validation (default: 16)',
    )
    parser.add_argument(
        '--phy',
        type=str,
        default='phy',
        help='filename of dataset',
    )
    parser.add_argument(
        '--phy-weight',
        type=float,
        default=1.0,
        help='relative weight for physics informed loss function',
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
        '--day-int',
        type=int,
        default=3,
        help='date interval to create inputs',
    )
    parser.add_argument(
        '--forecast',
        type=int,
        default=1,
        help='date to forecast',
    )
    parser.add_argument(
        '--predict',
        type=str,
        default="all",
        help='prediction outputs',
    )
    parser.add_argument(
        '--model-type',
        type=str,
        default="unet",
        help='types of the neural network model (e.g. unet, cnn, fc)',
    )
    
    parser.add_argument(
        '--backend',
        type=str,
        default='nccl',
        help='backend for distribute training (default: nccl)',
    )

    # Set automatically by torch distributed launch
    parser.add_argument(
        '--local-rank',
        type=int,
        default=0,
        help='local rank for distributed training',
    )
    
    args = parser.parse_args()
    if 'LOCAL_RANK' in os.environ:
        args.local_rank = int(os.environ['LOCAL_RANK'])
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    return args

def make_sampler_and_loader(args, train_dataset, shuffle = True):
    """Create sampler and dataloader for train and val datasets."""
    torch.set_num_threads(args.world_size)
    kwargs: dict[str, Any] = (
        {'num_workers': args.world_size, 'pin_memory': True} if args.cuda else {}
    )
    
    if args.cuda:
        kwargs['prefetch_factor'] = 8
        kwargs['persistent_workers'] = True
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=shuffle
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            **kwargs,
        )
    else:
        train_sampler = DistributedSampler(train_dataset)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            **kwargs,
        )

    return train_sampler, train_loader

def RMSE(prd, obs):
    err = torch.square(obs-prd)
    return torch.nanmean(err)**0.5

def MSE(prd, obs):
    err = torch.square(obs-prd)
    return torch.nanmean(err)

# def init_processes(backend):
#     dist.init_process_group(backend, rank=WORLD_RANK, world_size=WORLD_SIZE)
#     run(backend)
    
class Metric:
    """Metric tracking class."""

    def __init__(self, name: str):
        """Init Metric."""
        self.name = name
        self.total = torch.tensor(0.0)
        self.n = torch.tensor(0.0)

    def update(self, val: torch.Tensor, n: int = 1) -> None:
        """Update metric.

        Args:
            val (float): new value to add.
            n (int): weight of new value.
        """
        dist.all_reduce(val, async_op=False)
        self.total += val.cpu() / dist.get_world_size()
        self.n += n

    @property
    def avg(self) -> torch.Tensor:
        """Get average of metric."""
        return self.total / self.n
    
# def train(
#     epoch: int,
#     model: torch.nn.Module,
#     optimizer: torch.optim.Optimizer,
#     loss_func: torch.nn.Module,
#     train_loader: torch.utils.data.DataLoader,
#     train_sampler: torch.utils.data.distributed.DistributedSampler,
#     args
# ):
    
#     """Train model."""
#     model.train()
#     train_sampler.set_epoch(epoch)
    
#     mini_step = 0
#     step_loss = torch.tensor(0.0).to('cuda' if args.cuda else 'cpu')
#     train_loss = Metric('train_loss')
#     t0 = time.time()
    
#     with tqdm(
#         total=math.ceil(len(train_loader) / args.batches_per_allreduce),
#         bar_format='{l_bar}{bar:10}{r_bar}',
#         desc=f'Epoch {epoch:3d}/{args.epochs:3d}',
#         disable=not args.verbose,
#     ) as t:
#         for batch_idx, (data, target) in enumerate(train_loader):
#             mini_step += 1
#             ind = torch.sum(data.isnan(), dim=(1,2,3))
#             data = data[ind==0, :, :, :]
#             target = target[ind==0, :, :, :]
#             if args.cuda:
#                 data, target = data.cuda(), target.cuda()
            
#             if args.model_type == "casunet":
#                 output = model(data, data[:, 2:3])
#             else:
#                 output = model(data)
                
#             if args.phy == "phy":
#                 loss = loss_func(output, target, data[:, 2*args.day_int, :, :].cuda())
#             else:
#                 loss = loss_func(output, target)

#             with torch.no_grad():
#                 step_loss += loss
            
#             # loss = loss / args.batches_per_allreduce

#             # if (
#             #     mini_step % args.batches_per_allreduce == 0
#             #     or batch_idx + 1 == len(train_loader)
#             # ):
#             #     loss.backward()
#             # else:
#             #     with model.no_sync():  # type: ignore
#             #         loss.backward()

#             loss.backward()

#             if (
#                 mini_step % args.batches_per_allreduce == 0
#                 or batch_idx + 1 == len(train_loader)
#             ):
#                 optimizer.zero_grad()
#                 optimizer.step()
                
#                 train_loss.update(step_loss / mini_step)
#                 step_loss.zero_()

#                 t.set_postfix_str('loss: {:.4f}'.format(train_loss.avg))
#                 t.update(1)
#                 mini_step = 0

#     if args.log_writer is not None:
#         args.log_writer.add_scalar('train/loss', train_loss.avg, epoch)
        
#     return train_loss.avg


# def validate(
#     epoch: int,
#     model: torch.nn.Module,
#     loss_func: torch.nn.Module,
#     val_loader: torch.utils.data.DataLoader,
#     args
# ):
#     """Test the model."""
#     model.eval()
#     rmse = 0
#     val_loss = Metric('val_loss')

#     with tqdm(
#         total=len(val_loader),
#         bar_format='{l_bar}{bar:10}|{postfix}',
#         desc='             ',
#         disable=not args.verbose
#     ) as t:
#         with torch.no_grad():
#             for i, (data, target) in enumerate(val_loader):
#                 ind = torch.sum(data.isnan(), dim=(1,2,3))
#                 data = data[ind==0, :, :, :]
#                 target = target[ind==0, :, :, :]
#                 if args.cuda:
#                     data, target = data.cuda(), target.cuda()
                    
#                 if args.model_type == "casunet":
#                     output = model(data, data[:, 2:3])
#                 else:
#                     output = model(data)
                
#                 if args.phy == "phy":
#                     val_loss.update(loss_func(output, target, data[:, 2*args.day_int, :, :].cuda()))
#                 else:
#                     val_loss.update(loss_func(output, target))

#                 rmse += RMSE(target, output)*100

#                 t.update(1)
#                 if i + 1 == len(val_loader):
#                     t.set_postfix_str(
#                         'val_loss: {:.4f}'.format(rmse), #val_loss.avg
#                         refresh=False,
#                     )

#     if args.log_writer is not None:
#         args.log_writer.add_scalar('val/loss', val_loss.avg, epoch)
        
#     return val_loss.avg

def train(
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_func: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    train_sampler: torch.utils.data.distributed.DistributedSampler,
    landmask,
    args
):
    
    """Train model."""
    model.train()
    train_sampler.set_epoch(epoch)
    
    mini_step = 0
    step_loss = torch.tensor(0.0).to('cuda' if args.cuda else 'cpu')
    train_loss = [] #Metric('train_loss')
    t0 = time.time()

    for batch_idx, (data, target) in enumerate(train_loader):
        mini_step += 1
        ind = torch.sum(data.isnan(), dim=(1,2,3))
        data = data[ind==0, :, :, :]
        target = target[ind==0, :, :, :]
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        
        if args.model_type == "casunet":
            output = model(data, data[:, 2:3])
        else:
            output = model(data)
            
        if args.phy == "phy":
            loss = loss_func(output, target, data[:, 2*args.day_int, :, :].cuda())
        else:
            loss = loss_func(output, target)
        
        # loss_func = single_loss(landmask)
        # loss0 = loss_func(output[:, 0], target[:, 0])
        # loss1 = loss_func(output[:, 1], target[:, 1])
        # loss2 = loss_func(output[:, 2], target[:, 2])            

        with torch.no_grad():
            step_loss += loss

        loss = loss / args.batches_per_allreduce
        
        loss.backward()

        if (
            mini_step % args.batches_per_allreduce == 0
            or batch_idx + 1 == len(train_loader)
        ):
            
            optimizer.step()
            optimizer.zero_grad()
            
            train_loss.append(loss.item()) #.update(step_loss / mini_step)
            step_loss.zero_()

    if args.log_writer is not None:
        args.log_writer.add_scalar('train/loss', train_loss.avg, epoch)

    return torch.nanmean(torch.tensor(train_loss)) #train_loss.avg.item()


def validate(
    epoch: int,
    model: torch.nn.Module,
    loss_func: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    landmask,
    args
):
    """Test the model."""
    model.eval()
    rmse = torch.tensor([0., 0., 0.]).cuda()
    val_loss = Metric('val_loss')

    with torch.no_grad():
        for i, (data, target) in enumerate(val_loader):
            ind = torch.sum(data.isnan(), dim=(1,2,3))
            data = data[ind==0, :, :, :]
            target = target[ind==0, :, :, :]
            if args.cuda:
                data, target = data.cuda(), target.cuda()
                
            if args.model_type == "casunet":
                output = model(data, data[:, 2:3])
            else:
                output = model(data)
            
            if args.phy == "phy":
                val_loss.update(loss_func(output, target, data[:, 2*args.day_int, :, :].cuda()))
            else:
                val_loss.update(loss_func(output, target))

            for c in range(0, target.shape[1]):
                rmse[c] += MSE(target[:, c, landmask==0], output[:, c, landmask==0]) #*100

            # t.update(1)
            # if i + 1 == len(val_loader):
            #     t.set_postfix_str(
            #         'val_loss: {:.4f}'.format(rmse/(i+1)*100), #val_loss.avg
            #         refresh=False,
            #     )

    if args.log_writer is not None:
        args.log_writer.add_scalar('val/loss', val_loss.avg, epoch)
        
    return rmse / (i+1) * 100 #val_loss.avg.item()

def test(
    model: torch.nn.Module,
    loss_func: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    model_name,
    args
):
    """Test the model."""
    model.eval()
    val_loss = Metric('val_loss')

    with torch.no_grad():
        for i, (data, target) in enumerate(val_loader):
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            val_loss.update(loss_func(output, target))

            test_save = [data.to('cpu').detach().numpy(), target.to('cpu').detach().numpy(), output.to('cpu').detach().numpy()]

            # Open a file and use dump()
            with open(f'../results/test_{model_name}_{args.global_rank}_{i}.pkl', 'wb') as file:
                pickle.dump(test_save, file)
    
##########################################################################################

def main() -> None:    
    
    #### SETTING CUDA ENVIRONMENTS ####
    """Main train and eval function."""
    args = parse_args()

    torch.distributed.init_process_group(
        backend=args.backend,
        init_method='env://',
    )

    if args.cuda:
        torch.cuda.set_device(args.local_rank)
        torch.cuda.manual_seed(args.seed)
        # torch.backends.cudnn.benchmark = False
        # torch.backends.cudnn.deterministic = True
    
    if args.no_cuda:
        device = torch.device('cpu')
        device_name = 'cpu'
    else:
        device = torch.device('cuda')
        device_name = 'gpu'
        
    torch.cuda.empty_cache()
    
    args.verbose = dist.get_rank() == 0
    world_size = int(os.environ['WORLD_SIZE'])
    args.world_size = world_size

    if args.verbose:
        print('Collecting env info...')
        # print(collect_env.get_pretty_env_info())
        # print()

    for r in range(torch.distributed.get_world_size()):
        if r == torch.distributed.get_rank():
            print(
                f'Global rank {torch.distributed.get_rank()} initialized: '
                f'local_rank = {args.local_rank}, '
                f'world_size = {torch.distributed.get_world_size()}',
            )
        torch.distributed.barrier()
    
    args.global_rank = torch.distributed.get_rank()

    os.makedirs(args.log_dir, exist_ok=True)
    args.checkpoint_format = os.path.join(args.log_dir, args.checkpoint_format)
    # args.log_writer = SummaryWriter(args.log_dir) if args.verbose else None  
    args.log_writer = None if args.verbose else None  
    
    data_path = args.data_dir
    data_file = args.data_file
    model_dir = args.model_dir
    date = args.date
    sdate = args.sdate

    n_epochs = args.epochs
    batch_size = args.batch_size  # size of each batch
    val_batch = args.val_batch_size  # size of validation batch size
    lr = args.base_lr

    phy = args.phy ## PHYSICS OR NOT
    phy_w = args.phy_weight
    dayint = args.day_int
    forecast = args.forecast    
    
    #### READ DATA ##################################################################    
    data_ver = data_file[-6:-4]
    data_type = data_file[6:9]
    
    with open(data_path + data_file, 'rb') as file:
        xx, yy, days, months, years, cnn_input, cnn_output = pickle.load(file)   
    
    if data_type == "cic":
        # CICE data
        data_type = "cice"
        cnn_input = cnn_input[:,:,:, [0,1,2,3,4,5,6]]
        cnn_output = cnn_output[:,:,:, [0,1,3]]
        # Land mask data
        with open(data_path + f"landmask_physics_256.pkl", 'rb') as file:
            landmask = pickle.load(file)
    elif data_type == "cnn":
        data_type = "sat"
        # Satellite observation data
        # cnn_input = cnn_input[:,60:220, 90:250,[0,1,2,3,4,5]]
        # cnn_output = cnn_output[:,60:220, 90:250,:-1]
        # Land mask data
        # with open(data_path + f"landmask_320.pkl", 'rb') as file:
        #     landmask = pickle.load(file)
        #     landmask = torch.tensor(landmask)[30:286, 10:266]
        # with open(data_path + f"CAAmask_256.pkl", 'rb') as file:
        #     landmask = pickle.load(file)
        #     landmask = torch.tensor(landmask)[60:220, 90:250]
        
        cnn_input = cnn_input[:, :, :, [0,1,2,3,4,5]]
        cnn_output = cnn_output[:,:,:,:-1]
        with open(data_path + f"landmask_256.pkl", 'rb') as file:
            landmask = pickle.load(file)
            landmask = torch.tensor(landmask)
        
    if args.predict == "sic":
        cnn_output = cnn_output[:,:,:,2:3]
    elif args.predict == "sit":
        if data_ver == 'v4':
            cnn_output = cnn_output[:,:,:,3:4]
        else:
            print(f"SIT prediction is not available with {data_ver} data >>> Proceed with all prediction")
    elif args.predict == "uv":
        cnn_output = cnn_output[:,:,:,0:2]        

    landmask = torch.tensor(landmask) #[30:286, 10:266] # Land = 1; Ocean = 0;
    
    # cnn_input = cnn_input[:, :, :, :4] # Only U, V, SIC, SIT as input
    # cnn_input, cnn_output, days, months, years = convert_cnn_input2D(cnn_input, cnn_output, days, months, years, dayint, forecast)
    
    ## Add x y coordinates as inputs
    # if args.model_type != "lg":
    #     xx_n = (xx - xx.min())/(xx.max() - xx.min()).astype(np.float16)
    #     yy_n = (yy - yy.min())/(yy.max() - yy.min()).astype(np.float16)
    #     cnn_input = np.concatenate((cnn_input, np.repeat(np.array([np.expand_dims(xx_n, 2)]), cnn_input.shape[0], axis = 0).astype(np.float16)), axis = 3)
    #     cnn_input = np.concatenate((cnn_input, np.repeat(np.array([np.expand_dims(yy_n, 2)]), cnn_input.shape[0], axis = 0).astype(np.float16)), axis = 3)
    
    ## Convert numpy array into torch tensor
    cnn_input = torch.tensor(cnn_input, dtype=torch.float32)
    cnn_output = torch.tensor(cnn_output, dtype=torch.float32)
    
    mask1 = (years == date) # Test samples
    mask2 = (years >= sdate) # (days % 7 != 2) # Validation samples
    
    train_mask = (~mask1)&(mask2)
    val_mask = mask1
    
    train_input = cnn_input[train_mask] #cnn_input[(~mask1)&(~mask2), :, :, :]
    train_output = cnn_output[train_mask] #cnn_output[(~mask1)&(~mask2), :, :, :]
    val_input = cnn_input[val_mask] #cnn_input[(~mask1)&(mask2), :, :, :]
    val_output = cnn_output[val_mask] #cnn_output[(~mask1)&(mask2), :, :, :]
    # test_input = cnn_input[mask1, :, :, :]
    # test_output = cnn_output[mask1, :, :, :]    

    train_input = torch.permute(train_input, (0, 3, 1, 2)) * (landmask == 0)
    train_output = torch.permute(train_output, (0, 3, 1, 2)) * (landmask == 0)
    val_input = torch.permute(val_input, (0, 3, 1, 2)) * (landmask == 0)
    val_output = torch.permute(val_output, (0, 3, 1, 2)) * (landmask == 0)
    
    # print(train_input.size(), train_output.size(), val_input.size(), val_output.size()) 
    
    # train_dataset = TensorDataset(train_input, train_output)
    # val_dataset = TensorDataset(val_input, val_output)
    # test_dataset = TensorDataset(test_input, test_output)
    
    # Cehck sequential days ---------------------------------------
    seq_days = []
    step = 0
    for i in range(0, len(days)):
        if (days[i] ==1) & (years[i] != years[0]):
            step += days[i-1]
        seq_days.append(days[i] + step)

    seq_days = np.array(seq_days)
    train_days = seq_days[train_mask]
    val_days = seq_days[val_mask]
    # -------------------------------------------------------------
    
    train_dataset = SeaiceDataset(train_input, train_output, train_days, dayint, forecast, exact = True)
    # train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    val_dataset = SeaiceDataset(val_input, val_output, val_days, dayint, forecast, exact = True)
    
    if args.ratio < 1:
        generator = torch.Generator().manual_seed(0)
        train_dataset, _ = random_split(train_dataset, [args.ratio, 1-args.ratio], generator)
    
    n_samples = len(train_dataset) #.length
    val_samples = len(val_dataset) #.length
    in_channels, row, col = train_dataset[0][0].shape
    out_channels, _, _ = train_dataset[0][1].shape
    
    # n_samples, in_channels, row, col = train_input.size()
    # _, out_channels, _, _ = train_output.size()

    train_sampler, train_loader = make_sampler_and_loader(args, train_dataset, shuffle = True) 
    val_sampler, val_loader = make_sampler_and_loader(args, val_dataset, shuffle = False)
    if args.cuda:
        landmask = landmask.cuda() # Land = 1; Ocean = 0;
    
    # del cnn_input, cnn_output, train_input, train_output
    
    #############################################################################   
    if args.model_type == "unet":
        net = UNet(in_channels, out_channels)
    elif args.model_type == "mtunet":
        net = HF_UNet(in_channels, out_channels)
    elif args.model_type == "tsunet":
        net = TS_UNet(in_channels, out_channels, landmask, row) # Triple sharing
    elif args.model_type == "isunet":
        net = IS_UNet(in_channels, out_channels, landmask, row) # information sharing
    elif args.model_type == "hisunet":
        net = HIS_UNet(in_channels, out_channels, landmask, row, 3, phy) # hierarchical information sharing (attention blocks)
    elif args.model_type == "lbunet":
        net = LB_UNet(in_channels, out_channels, landmask)
    elif args.model_type == "ebunet":
        net = EB_UNet(in_channels, out_channels, landmask)
    elif args.model_type == "casunet":
        net = Cascade_UNet(in_channels, out_channels, landmask)
    elif args.model_type == "cnn":
        net = Net(in_channels, out_channels)
    elif args.model_type == "fc":
        net = FC(in_channels, out_channels)
    elif args.model_type == "lg": # linear regression
        net = linear_regression(in_channels, out_channels, row, col)
    else:
        net = UNet(in_channels, out_channels)

    model_name = f"torch_{args.model_type}_{data_type}{data_ver}_{args.predict}_{sdate}_{date}_r{args.ratio}_pw{phy_w}_{phy}_d{dayint}f{forecast}_gpu{world_size}"  
    print(model_name)
    
    # print(device)
    net.to(device)
    
    if args.no_cuda == False:
        net = torch.nn.parallel.DistributedDataParallel(
            net,
            device_ids=[args.local_rank],
        )

    if phy == "phy":
        loss_fn = physics_loss(landmask, phy_w) # nn.L1Loss() #nn.CrossEntropyLoss()
    elif phy == "nophy":
        if args.model_type == "fc":
            loss_fn = nn.L1Loss()
        else:
            if args.predict== "all":
                loss_fn = ref_loss(landmask) #custom_loss(landmask, args.forecast) #nn.MSELoss() #ref_loss(landmask) # nn.L1Loss() #nn.CrossEntropyLoss()            
            else:
                loss_fn = single_loss(landmask)

    optimizer = optim.Adam(net.parameters(), lr=lr)
    scheduler = ExponentialLR(optimizer, gamma=0.98)

    history = {'loss': [], 'val_loss': [], 'time': []}

    total_params = sum(p.numel() for p in net.parameters())
    if dist.get_rank() == 0:
        print(f"Number of parameters: {total_params}")
        print("Train sample: {0}, Val sample: {1}; IN: {2} OUT: {3} ({4} x {5})".format(n_samples, val_samples, in_channels, out_channels, row, col)) 
    
    
    for epoch in range(n_epochs):

        t0 = time.time()
        
        train_loss = 0.0
        train_cnt = 1
        val_cnt = 1
        
        net.train()
        
        train_loss = train(
            epoch,
            net,
            optimizer,
            loss_fn,
            train_loader,
            train_sampler,
            landmask,
            args
        )
        
        scheduler.step()
        val_loss = validate(epoch, net, loss_fn, val_loader, landmask, args)
        
        # ##### TRAIN ###########################
        # for batch_idx, (data, target) in enumerate(train_loader):

        #     ind = torch.sum(data.isnan(), dim=(1,2,3))
        #     data = data[ind==0, :, :, :]
        #     target = target[ind==0, :, :, :]
        #     if args.cuda:
        #         data, target = data.cuda(), target.cuda()

        #     output = net(data)
                
        #     if args.phy == "phy":
        #         loss = loss_fn(output, target, data[:, 2*args.day_int, :, :].cuda())
        #     else:
        #         loss = loss_fn(output, target)

        #     train_loss += loss.cpu().item()
        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()
        #     train_cnt += 1

        # ##### Validation ###########################
        # val_loss = 0
        # val_cnt = 0
        # net.eval()
        # with torch.no_grad():
        #     for i, (data, target) in enumerate(val_loader):
        #         ind = torch.sum(data.isnan(), dim=(1,2,3))
        #         data = data[ind==0, :, :, :]
        #         target = target[ind==0, :, :, :]
        #         if args.cuda:
        #             data, target = data.cuda(), target.cuda()
                    
        #         output = net(data)
    
        #         val_loss += RMSE(target, output)*100
        #         val_cnt += 1

        torch.cuda.empty_cache()

        history['loss'].append(train_loss/train_cnt)
        history['val_loss'].append(val_loss/val_cnt)
        history['time'].append(time.time() - t0)

        t1 = time.time() - t0
        if dist.get_rank() == 0:
            if epoch % 2 == 0:
                print('Epoch {0} >> Train loss: {1:.4f} [{2:.2f} sec]'.format(
                    str(epoch).zfill(3), train_loss/train_cnt, t1))
                print('          >> Val loss: {0:.4f}, {1:.4f}, {2:.4f}'.format(
                    val_loss[0], val_loss[1], val_loss[2]))
            
            # if epoch % args.checkpoint_freq == 0:
            #     save_checkpoint(net.module, optimizer, args.checkpoint_format.format(epoch=epoch))          
            
            if epoch == n_epochs-1:
                torch.save(net.state_dict(), f'{model_dir}/{model_name}.pth')

                with open(f'{model_dir}/history_{model_name}.pkl', 'wb') as file:
                    pickle.dump(history, file)
                    
        # if epoch > 100 and torch.nanmedian(torch.tensor(history['val_loss'][-10:])) >= torch.nanmedian(torch.tensor(history['val_loss'][-20:-10])):
        #     break # over-fitting
    
    torch.cuda.empty_cache()
    
    del train_dataset, train_loader, train_sampler
    
    # Test the model with the trained model ========================================
    val_years = years[mask1][val_dataset.valid]
    val_months = months[mask1][val_dataset.valid]
    val_days = days[mask1][val_dataset.valid]
    
    net.eval()
    
    if dist.get_rank() == 0:    
        for m in np.unique(val_years):
            # if m % world_size == dist.get_rank():
            
            idx = np.where(val_years == m)[0]
            valid = []
            
            # data = val_input[val_months==m, :, :, :]
            data = torch.zeros([len(idx), in_channels, row, col])
            target = torch.zeros([len(idx), out_channels, row, col]) #val_output[val_months==m, :, :, :]
            output = torch.zeros([len(idx), out_channels, row, col])
            
            with torch.no_grad():
                for j in range(0, len(idx)): #range(0, target.size()[0]):
                    data[j, :, :, :] = val_dataset[idx[j]][0]
                    check = torch.sum(torch.tensor(val_dataset[idx[j]][0]).isnan())
                    if check == 0:
                        if args.model_type == "casunet":
                            output[j, :, :, :] = net(val_dataset[idx[j]][0][None, :], val_dataset[idx[j]][0][None, :][:, 2:3])
                        else:
                            output[j, :, :, :] = net(val_dataset[idx[j]][0][None, :])                        
                        target[j, :, :, :] = val_dataset[idx[j]][1][None, :]
                        valid.append(j)              
            
            test_save = [data[valid].to('cpu').detach().numpy().astype(np.float16), target[valid].to('cpu').detach().numpy().astype(np.float16), 
                         output[valid].to('cpu').detach().numpy().astype(np.float16),
                         val_months[idx[valid]], val_days[idx[valid]]]
            # print(len(valid), data[valid].shape, target[valid].shape, output[valid].shape)

            # Open a file and use dump()
            with open(f'../results/test_{model_name}_{str(int(m)).zfill(2)}.pkl', 'wb') as file:
                pickle.dump(test_save, file)
                        
    if dist.get_rank() == 0:
        print("#### Validation done!! ####")     
    # ===============================================================================

if __name__ == '__main__':
    main()
