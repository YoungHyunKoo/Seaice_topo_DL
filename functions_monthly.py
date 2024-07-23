import pandas as pd
import glob, os
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import geopandas
import netCDF4
import h5py
import datetime as dt
import pyproj
from tqdm import tqdm
from pyproj import Proj, transform
from shapely.geometry import Polygon
import cartopy.crs as ccrs
from netCDF4 import date2num,num2date

from scipy.interpolate import griddata

import torch
import torch.nn as nn
import cdsapi
import xarray as xr
from urllib.request import urlopen

import pickle

global data_path
data_path = "D:\\PINN\\data"
# data_path = "C:\\Users\\yok223\\Research\\PINN\\data"

from scipy.ndimage.filters import gaussian_filter

def get_ice_motion(ncfile, i, sampling_size = 1):
# ncfile: input monthly ERA5 file (ncfile)
# field: input variable ('sst', 't2m', 'u10', 'v10')
# bounding_box: processed area (Ross Sea - Amundsen Sea)
# latlon_ib: geocoordinates of the iceberg (lat, lon)
# time_ib: date of the iceberg (datetime format)

    with netCDF4.Dataset(ncfile, 'r') as nc:
        keys = nc.variables.keys()
        fields = ['u', 'v']
    
        xs = np.array(nc.variables['x'])[::sampling_size]
        ys = np.array(nc.variables['y'])[::sampling_size]  
        xx, yy = np.meshgrid(xs, ys)
        lat = np.array(nc.variables['latitude'])[::sampling_size, ::sampling_size]
        lon = np.array(nc.variables['longitude'])[::sampling_size, ::sampling_size]
    
        days = np.array(nc.variables['time']).astype(float)
        
        for field in fields:                
    
            # data = np.zeros([len(idxs), xx.shape[0], yy.shape[0]])
    
            data = np.array(nc.variables[field][i][::sampling_size, ::sampling_size])
            # cm/s to km/day
            data[data == -9999] = np.nan                      
    
            data_mean = np.array([np.mean(data, axis = 0)])
    
            # df[field] = data_mean.flatten()
    
            if field == "u":
                u = data # data_mean
                # u[np.isnan(u)] = 0
            elif field == "v":
                v = data # data_mean
                # v[np.isnan(v)] = 0   
    
    u[np.isnan(u)] = 0
    v[np.isnan(v)] = 0
    
    # Apply Gaussian filter
    # u = gaussian_filter(u, sigma = 3)
    # v = gaussian_filter(v, sigma = 3)
    
    return xx, yy, lat, lon, u, v

def get_SIT(ncfile, xx, yy):
# ncfile: input CS2 file (ncfile)

    with netCDF4.Dataset(ncfile, 'r') as nc:
        lat = np.array(nc.variables['lat'])
        lon = np.array(nc.variables['lon'])
        hi = np.array(nc.variables['analysis_sea_ice_thickness'])[0]
    
    # EPSG:4326 (WGS84); EPSG:3408 (NSIDC EASE-Grid North - Polar pathfinder sea ice movement)
    inProj = Proj('epsg:4326')  
    outProj = Proj('epsg:3408')
    xx2,yy2 = transform(inProj,outProj,lat,lon)

    grid_sit = griddata((xx2.flatten(), yy2.flatten()), hi.flatten(), (xx, yy), method='linear')
    grid_sit[grid_sit < 0] = 0
    
    return grid_sit


def get_SIC(t1, xx, yy, dtype = "noaa", region = "SH"):
    ## Read SIC data (AMSR) ==================================================
    if dtype == "AMSR":
        h5file = data_path + "/{0}/SIC/AMSR_U2_L3_SeaIce25km_B04_{1}.he5".format(region, dt.datetime.strftime(t1, "%Y%m%d"))

        if os.path.exists(h5file):
            f = h5py.File(h5file)

            lat2 = f['HDFEOS']['GRIDS']['NpPolarGrid25km']['lat'][:]
            lon2 = f['HDFEOS']['GRIDS']['NpPolarGrid25km']['lon'][:]
            sic = f['/HDFEOS/GRIDS/NpPolarGrid25km/Data Fields/SI_25km_NH_ICECON_DAY'][:].astype(float)
            sic[sic <= 0] = 0
            sic[sic > 100] = 0

            # EPSG:4326 (WGS84); EPSG:3408 (NSIDC EASE-Grid North - Polar pathfinder sea ice movement)
            inProj = Proj('epsg:4326')  
            outProj = Proj('epsg:3408')
            xx2,yy2 = transform(inProj,outProj,lat2,lon2)
            grid_sic = griddata((xx2.flatten(), yy2.flatten()), sic.flatten(), (xx, yy), method='linear')
            grid_sic[np.isnan(grid_sic)] = 0
            return grid_sic * 0.01  # Change into 0-1

        else:
            print("Filename is NOT correct!")
            
    elif dtype == "noaa":
        ncfile = data_path + "/{0}/SIC_NOAA/seaice_conc_daily_{0}_{1}_f17_v04r00.nc".format(region, dt.datetime.strftime(t1, "%Y%m%d"))
        
        if os.path.exists(ncfile):
            with netCDF4.Dataset(ncfile, 'r') as nc:
                xx0 = np.array(nc.variables['xgrid'])
                yy0 = np.array(nc.variables['ygrid'])
                sic = np.array(nc.variables['cdr_seaice_conc'])[0] # CDR SIC
                # sic = np.array(nc.variables['nsidc_bt_seaice_conc'])[0] # BT SIC
                # sic = np.array(nc.variables['nsidc_nt_seaice_conc'])[0] # NT SIC

                sic[sic <= 0] = 0
                sic[sic > 1] = 0

                # ESPG:3411 (NSIDC Sea Ice Polar Stereographic North - SIC data)
                if region == "NH":
                    inProj = Proj('epsg:3411')
                    outProj = Proj('epsg:3408')
                elif region == "SH":
                    inProj = Proj('epsg:3412')
                    outProj = Proj('epsg:3409')
                xx1, yy1 = np.meshgrid(xx0, yy0)
                xx2,yy2 = transform(inProj,outProj,xx1,yy1)
                grid_sic = griddata((xx2.flatten(), yy2.flatten()), sic.flatten(), (xx, yy), method='linear')
                grid_sic[np.isnan(grid_sic)] = 0
            return grid_sic

        else:
            print("Filename is NOT correct!")

def retrieve_hourly_ERA5(year, months, days, region = "SH"):
    c = cdsapi.Client()
    # dataset to read
    dataset = 'reanalysis-era5-single-levels'
    # flag to download data
    download_flag = False
    variables = [
        '10m_u_component_of_wind', '10m_v_component_of_wind', 'instantaneous_10m_wind_gust',
        '2m_temperature', 'sea_ice_cover', 'surface_pressure'
    ]
    # api parameters 
    if region == "NH":
        params = {
            'format': 'netcdf',
            'product_type': 'reanalysis',
            'variable': variables,
            'year':[str(year)],
            'month': months,
            'day': days,
            'time': ['12:00'],
            'grid': [1, 0.5],
            'area': [90, -180, 40, 180]
            }
    
    elif region == "SH":
        params = {
            'format': 'netcdf',
            'product_type': 'reanalysis',
            'variable': variables,
            'year':[str(year)],
            'month': months,
            'day': days,
            'time': ['12:00'],
            'grid': [1, 0.5],
            'area': [-50, -180, -90, 180]
            }

    # retrieves the path to the file
    fl = c.retrieve(dataset, params)

    # load into memory
    with urlopen(fl.location) as f:
        ds = xr.open_dataset(f.read())

    return ds

def retrieve_monthly_ERA5(year, month, region = "SH"):
    c = cdsapi.Client()
    # dataset to read
    dataset = 'reanalysis-era5-single-levels-monthly-means'
    # flag to download data
    download_flag = False
    variables = [
        '10m_u_component_of_wind', '10m_v_component_of_wind', 'instantaneous_10m_wind_gust',
        '2m_temperature', 'sea_ice_cover', 'surface_pressure'
    ]
    # api parameters 
    if region == "NH":
        params = {
            'format': 'netcdf',
            'product_type': 'monthly_averaged_reanalysis',
            'month': [str(month).zfill(2)],
            'year': [str(year)],
            'area': [90, -180, 40, 180],
            'variable': variables,
            'time': '00:00',
        }
    
    elif region == "SH":

        params = {
            'format': 'netcdf',
            'product_type': 'monthly_averaged_reanalysis',
            'month': [str(month).zfill(2)],
            'year': [str(year)],
            'area': [-50, -180, -90, 180],
            'variable': variables,
            'time': '00:00',
        }

    # retrieves the path to the file
    fl = c.retrieve(dataset, params)

    # load into memory
    with urlopen(fl.location) as f:
        ds = xr.open_dataset(f.read())

    return ds

def rotate_vector(u, v, lon, ref_lon = 0):
    angle = (lon-ref_lon)*np.pi/180 # rotation angle (radian)
    u2 = u*np.cos(angle) - v*np.sin(angle)
    v2 = u*np.sin(angle) + v*np.cos(angle)
    return u2, v2

def get_ERA5(ds, i, xx, yy, region = "NH", ref_lon = 0):
    lat3, lon3 = np.meshgrid(ds.latitude, ds.longitude)
    inProj = Proj('epsg:4326')
    if region == "NH":
        if ref_lon == 0:
            outProj = Proj('epsg:3408')
        elif ref_lon == -45:
            outProj = Proj('proj4: +proj=stere +lon_0=-45 +lat_0=90 +k=1 +R=6378273 +no_defs')
    elif region == "SH":
        outProj = Proj('epsg:3409')
        
    xx3,yy3 = transform(inProj,outProj,lat3,lon3)
    t2m = np.array(ds.t2m[i]).transpose()
    u10 = np.array(ds.u10[i]).transpose()
    v10 = np.array(ds.v10[i]).transpose()
    sic = np.array(ds.siconc[i]).transpose()
    i10 = np.array(ds.i10fg[i]).transpose()
    
    u10, v10 = rotate_vector(u10, v10, lon3, ref_lon)
    
    grid_t2m = griddata((xx3.flatten(), yy3.flatten()), np.array(t2m).flatten(), (xx, yy), method='linear')
    grid_u10 = griddata((xx3.flatten(), yy3.flatten()), np.array(u10).flatten(), (xx, yy), method='linear')
    grid_v10 = griddata((xx3.flatten(), yy3.flatten()), np.array(v10).flatten(), (xx, yy), method='linear')
    grid_sic = griddata((xx3.flatten(), yy3.flatten()), np.array(sic).flatten(), (xx, yy), method='linear')
    grid_i10 = griddata((xx3.flatten(), yy3.flatten()), np.array(i10).flatten(), (xx, yy), method='linear')
    
    grid_t2m[np.isnan(grid_t2m)] = 0
    grid_u10[np.isnan(grid_u10)] = 0
    grid_v10[np.isnan(grid_v10)] = 0
    grid_sic[np.isnan(grid_sic)] = 0
    grid_i10[np.isnan(grid_i10)] = 0
    
    return grid_t2m, grid_u10, grid_v10, grid_sic, grid_i10

def calculate_div(u, v, dx = 25, dy = 25):
    # DIV = du / dx + dv / dy
    div = np.zeros(u.shape) * np.nan
    dudx = ((u[:, 2:, 2:] - u[:, 2:, :-2]) + (u[:, 1:-1, 2:] - u[:, 1:-1, :-2]) + (u[:, :-2, 2:] - u[:, :-2, :-2])) / (3*dx*2)
    dvdy = ((v[:, 2:, 2:] - v[:, :-2, 2:]) + (v[:, 2:, 1:-1] - v[:, :-2, 1:-1]) + (v[:, 2:, :-2] - v[:, :-2, :-2])) / (3*dy*2)
    div[:, 1:-1, 1:-1] = dudx + dvdy
    return div

def calculate_shr(u, v, dx = 25, dy = 25):
    # SHR = ((du/dx - dv/dy)**2 + (du/dy + dv/dx)**2) ** 0.5
    shr = np.zeros(u.shape) * np.nan
    dudx = ((u[:, 2:, 2:] - u[:, 2:, :-2]) + (u[:, 1:-1, 2:] - u[:, 1:-1, :-2]) + (u[:, :-2, 2:] - u[:, :-2, :-2])) / (3*dx*2)
    dvdy = ((v[:, 2:, 2:] - v[:, :-2, 2:]) + (v[:, 2:, 1:-1] - v[:, :-2, 1:-1]) + (v[:, 2:, :-2] - v[:, :-2, :-2])) / (3*dy*2)
    
    dvdx = ((v[:, 2:, 2:] - v[:, 2:, :-2]) + (v[:, 1:-1, 2:] - v[:, 1:-1, :-2]) + (v[:, :-2, 2:] - v[:, :-2, :-2])) / (3*dx*2)
    dudy = ((u[:, 2:, 2:] - u[:, :-2, 2:]) + (u[:, 2:, 1:-1] - u[:, :-2, 1:-1]) + (u[:, 2:, :-2] - u[:, :-2, :-2])) / (3*dy*2)
    shr[:, 1:-1, 1:-1] = ((dudx - dvdy)**2 + (dudy + dvdx)**2)**0.5
    return shr

def make_dataset(year, month, w = 1, region = "SH"):
    # ncfile = glob.glob("F:\\2022_Ross\\ERA5\\icemotion_daily_sh_25km_{0}*.nc".format(year))[0]
    ncfile = data_path + f"/{region}/Sea_ice_drift/icemotion_daily_{region.lower()}_25km_{year}0101_{year}1231_v4.1.nc"
    with netCDF4.Dataset(ncfile, 'r') as nc:
    ## Adjust the number of training datasets ===========================
        days = np.array(nc.variables['time']).astype(float)[:]
        row, col = np.shape(np.array(nc.variables['latitude']))

    d1 = int(dt.datetime(year, month, 1).strftime('%j')) - 1
    if month == 12:
        d2 = int(dt.datetime(year, 12, 31).strftime('%j')) - 2
    else:
        d2 = int(dt.datetime(year, month+1, 1).strftime('%j')) - 1

    n_samples = np.arange(d1, d2, 1)

    # ERA5 =================================================
    ds = retrieve_monthly_ERA5(year, month, region)
    print("Load ERA5")
        
    # Initialize grid input ==========================================
    grid_input = np.zeros([1, row, col, 13])

    ## Read ice motion data ===========================================
    sampling_size = 1
    xx, yy, lat, lon, u, v = get_ice_motion(ncfile, n_samples, sampling_size)
    print(u.shape)
    grid_u = np.nanmean(u, axis = 0)
    grid_v = np.nanmean(v, axis = 0)
    # Divergence
    div = calculate_div(u, v, dx = 25, dy = 25)
    div95 = np.nanquantile(div, 0.95, axis = 0)
    div50 = np.nanquantile(div, 0.50, axis = 0)
    div05 = np.nanquantile(div, 0.05, axis = 0)
    # Shear rate
    shr = calculate_shr(u, v, dx = 25, dy = 25)
    shr95 = np.nanquantile(shr, 0.95, axis = 0)
    shr50 = np.nanquantile(shr, 0.50, axis = 0)
    shr05 = np.nanquantile(shr, 0.05, axis = 0)    
    print("Sea ice motion updated")

    ## Read SIC data ==================================================
    grid_sic0 = np.zeros([len(n_samples), row, col])
    for i, idx in enumerate(n_samples):
        t1 = dt.datetime(1970, 1, 1) + dt.timedelta(days = days[idx])
        t2 = dt.datetime(1970, 1, 1) + dt.timedelta(days = days[idx]+1)
        grid_sic0[i] = get_SIC(t1, xx, yy, region = region)
    grid_sic = np.nanmean(grid_sic0, axis = 0)
    print("SIC updated")

    ## Read ERA5 data =================================================
    grid_t2m, grid_u10, grid_v10, _, grid_i10 = get_ERA5(ds, 0, xx, yy, region = region)
    print("ERA5 updated")

    grid_input[0, :, :, 0] = grid_u
    grid_input[0, :, :, 1] = grid_v
    grid_input[0, :, :, 2] = grid_sic
    grid_input[0, :, :, 3] = grid_t2m #grid_t2m - 210)/(310 - 210) #Max temp = 320 K, Min temp = 240 K)
    grid_input[0, :, :, 4] = grid_u10
    grid_input[0, :, :, 5] = grid_v10
    grid_input[0, :, :, 6] = grid_i10
    grid_input[0, :, :, 7] = div95
    grid_input[0, :, :, 8] = div50
    grid_input[0, :, :, 9] = div05
    grid_input[0, :, :, 10] = shr95
    grid_input[0, :, :, 11] = shr50
    grid_input[0, :, :, 12] = shr05
    
    # Masking ======================================
    mask1 = (grid_sic == 0) #(np.isnan(grid_u))   

    var_ip = np.shape(grid_input)[3]

    conv_input = np.copy(grid_input)

    for m in range(0, var_ip):
        subset = grid_input[0, :, :, m]
        subset[mask1] = np.nan
        conv_input[0, :, :, m] = subset
    
    # conv_input = conv_input[np.array(valid), :, :, :]
    
    return xx, yy, conv_input

# READ ICESAT-2 GRID DATA ###################################################

def get_IS2_grid(year, region, filepath = "D:\\IS2_topo_DL"):
    
    ##### Read data ##############################
    try: ncfile.close()  # just to be safe, make sure dataset is not already open.
    except: pass

    ncname = filepath + f'\\Ridges_density_{region}_{year}.nc'
    ds = xr.open_dataset(ncname)

    with netCDF4.Dataset(ncname, 'r') as nc:
        lat = np.array(nc.variables['lat'])
        lon = np.array(nc.variables['lon'])
        x = np.array(nc.variables['x'])
        y = np.array(nc.variables['y'])
        xx, yy = np.meshgrid(x, y)

        times = nc.variables['time']
        times = num2date(times[:], units = times.units)

        hours = np.array(nc.variables['time']).astype(float)
        time_era = []

        for i in range(0, len(hours)):
            time_era.append(dt.datetime(1800, 1, 1) + dt.timedelta(hours = hours[i]))

    date1, date2 = [], []
    for m in range(1, 13):
        date1.append(dt.datetime(year,m,1))
        if m == 12:
            date2.append(dt.datetime(year+1,1,1))
        else:
            date2.append(dt.datetime(year,m+1,1))
    
    fields = ['fb_mode', 'fb_std', 'fr_ridge', 'h_ridge']
    output = np.zeros([len(fields), len(date1), xx.shape[0], xx.shape[1]])
    
    for k, field1 in enumerate(fields):
        for i in range(0, len(date1)):                
    
            tidx = np.where((times >= date1[i]) & (times < date2[i]))[0]
            fb_count = np.nansum(np.array(ds.variables["fb_count"][tidx, :, :]), axis = 0)
            valid_count = (fb_count > 500)
            
            if np.sum(tidx) > 0:
                array = np.array(ds.variables[field1][tidx, :, :])
                # array = np.transpose(np.array(ds.variables[field1][tidx, :, :]), axes = (0, 2, 1))
                data1 = np.nanmedian(array, axis = 0)
                data1[~valid_count] = np.nan
    
            output[k, i] = np.transpose(np.nanmean(array, axis = 0))
            
    return output
############################################################################

def lookupNearest(x0, y0, xx, yy):
    ## xx - original x cooridnate
    ## yy - original y coordinate
    ## x0, y0: the coordinate you want to get index
    xi = np.abs(xx-x0).argmin()
    yi = np.abs(yy-y0).argmin()
    return xi, yi

def make_lstm_input2D(data_input, data_output, days = 7):
    # Input & output should be entire images for CNN
    n_samples, row, col, var_ip = np.shape(data_input)
    _, _, _, var_op = np.shape(data_output)
    row,col = 320, 320;
    lstm_input = np.zeros([n_samples-days, days, row, col, var_ip], dtype="int")
    lstm_output = np.zeros([n_samples-days, row, col, var_op], dtype="int")
    
    for n in range(0, n_samples-days):
        for i in range(0, days):
            for v in range(0, var_ip):
                lstm_input[n, i, :, :, v] = (data_input[n+i, 41:, :-41, v])
            for v in range(0, var_op):
                lstm_output[n, :, :, v] = (data_output[n+days, 41:, :-41, v])
    return lstm_input, lstm_output

def make_cnn_input2D(data_input, data_output, days = 3):
    # Input & output should be entire images for CNN
    n_samples, row, col, var_ip = np.shape(data_input)
    _, _, _, var_op = np.shape(data_output)
    row,col = 320, 320;
    cnn_input = np.zeros([n_samples-days, row, col, var_ip * days])
    cnn_output = np.zeros([n_samples-days, row, col, var_op])
    
    for n in range(0, n_samples-days):
        for v in range(0, var_ip):
            for i in range(0, days):
                cnn_input[n, :, :, v+i] = (data_input[n+i, :, :, v])
        for v in range(0, var_op):
            cnn_output[n, :, :, v] = (data_output[n+days, :, :, v])
    return cnn_input, cnn_output

def MAE(prd, obs):
    return np.nanmean(abs(obs-prd))

def MAE_grid(prd, obs):
    err = abs(obs-prd)
    return np.nanmean(err, axis=0)

def RMSE(prd, obs):
    err = np.square(obs-prd)
    return np.nanmean(err)**0.5

def RMSE_grid(prd, obs):
    err = np.square(obs-prd)
    return np.nanmean(err, axis=0)**0.5

def corr_grid(prd, obs):
    r1 = np.nansum((prd-np.nanmean(prd))*(obs-np.nanmean(obs)),axis=0)
    r2 = np.nansum(np.square(prd-np.nanmean(prd)), axis=0)*np.nansum(np.square(obs-np.nanmean(obs)),axis=0)
    r = r1/r2**0.5
    return r

def skill(prd, obs):
    err = np.nanmean(np.square(prd-obs))**0.5/np.nanmean(np.square(obs-np.nanmean(obs)))**0.5
    return 1-err

def MBE(prd, obs):
    return np.nanmean(prd-obs)

def corr(prd, obs):
    prd = prd.flatten()
    obs = obs.flatten()
    
    r = ma.corrcoef(ma.masked_invalid(prd), ma.masked_invalid(obs))[0, 1]
    return r

def float_to_int(input0, output0):
    offset = [-0.5, -0.5, 0, 0, -0.5, -0.5]
    
    input1 = np.zeros(np.shape(input0), dtype = np.int16)
    output1 = np.zeros(np.shape(output0), dtype = np.int16)
    
    for c in range(0, 6):
        sub_ip = input0[:, :, :, c] + offset[c]
        sub_ip[sub_ip < -1] = -1
        sub_ip[sub_ip > 1] = 1
        
        if c in [0, 1, 4, 5]:
            sub_ip[sub_ip == offset[c]] = 0
        
        sub_ip = (sub_ip * 20000).astype(np.int16)
        input1[:, :, :, c] = sub_ip
        
        if c < 3:
            sub_op = output0[:, :, :, c] + offset[c]
            sub_op[sub_op < -1] = -1
            sub_op[sub_op > 1] = 1
            
            if c in [0, 1, 4, 5]:
                sub_op[sub_op == offset[c]] = 0
            
            sub_op = (sub_op * 20000).astype(np.int16)
            output1[:, :, :, c] = sub_op
    
    return input1, output1

def nanmask(array, mask):
    array[mask] = np.nan
    return array

def advection(u, v, h):
    c = 1
    w_dx = torch.zeros([c, c, 3, 3])
    w_dy = torch.zeros([c, c, 3, 3])
    for i in range(0, c):
        w_dx[i, i] = torch.tensor([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])/3
        w_dy[i, i] = torch.tensor([[-1, -1, -1], [0, 0, 0], [1,1,1]])/3

    dx=nn.Conv2d(c, c, kernel_size=3, stride=1, padding="same", bias=False)
    dx.weight=nn.Parameter(w_dx, requires_grad = False)

    dy=nn.Conv2d(c, c, kernel_size=3, stride=1, padding="same", bias=False)
    dy.weight=nn.Parameter(w_dy, requires_grad = False)
    
    adv = u*dx(h) + v*dy(h)
    return adv