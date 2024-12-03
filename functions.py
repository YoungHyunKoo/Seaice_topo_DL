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

def get_ice_motion(ncfile, i, xx, yy, sampling_size = 1, region = "SH"):
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
        xx1, yy1 = np.meshgrid(xs, ys)
        lat = np.array(nc.variables['latitude'])[::sampling_size, ::sampling_size]
        lon = np.array(nc.variables['longitude'])[::sampling_size, ::sampling_size]
    
        days = np.array(nc.variables['time']).astype(float)
        
        for field in fields:                
    
            # data = np.zeros([len(idxs), xx.shape[0], yy.shape[0]])
    
            data = np.array(nc.variables[field][i][::sampling_size, ::sampling_size])
            # cm/s to km/day
            data[data == -9999] = np.nan
    
            # df[field] = data_mean.flatten()
            if region == "NH":
                inProj = Proj('epsg:3408')
                outProj = Proj('epsg:3411') #EASE:3408
            elif region == "SH":
                inProj = Proj('epsg:3409')
                outProj = Proj('epsg:3412') #EASE:3409
            xx2,yy2 = transform(inProj,outProj,xx1,yy1)
            grid_data = griddata((xx2.flatten(), yy2.flatten()), data.flatten(), (xx, yy), method='linear')

            if field == "u":
                u = np.array([grid_data]) # data_mean
                # u[np.isnan(u)] = 0
            elif field == "v":
                v = np.array([grid_data]) # data_mean
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
                    outProj = Proj('epsg:3411') #EASE:3408
                elif region == "SH":
                    inProj = Proj('epsg:3412')
                    outProj = Proj('epsg:3412') #EASE:3409
                xx1, yy1 = np.meshgrid(xx0, yy0)
                xx2,yy2 = transform(inProj,outProj,xx1,yy1)
                grid_sic = griddata((xx2.flatten(), yy2.flatten()), sic.flatten(), (xx, yy), method='linear')
                grid_sic[np.isnan(grid_sic)] = 0
            return grid_sic

        else:
            print("Filename is NOT correct!")

def a(*args, **kwargs): return ""

def retrieve_hourly_ERA5(year, months, days, region = "SH"):
    c = cdsapi.Client(quiet=True, wait_until_complete=False, delete=True, progress=False, warning_callback = a, sleep_max=10)
    # dataset to read
    dataset = 'reanalysis-era5-single-levels'
    # flag to download data
    # download_flag = False
    variables = [
        '10m_u_component_of_wind', '10m_v_component_of_wind', 'instantaneous_10m_wind_gust',
        '2m_temperature', 'sea_ice_cover', 'surface_pressure', 'skin_temperature'
    ]
    times = ['00:00', '03:00', '06:00', '09:00', '12:00', '15:00', '18:00', '21:00']
    # api parameters 
    if region == "NH":
        params = {
            'format': 'netcdf',
            'product_type': 'reanalysis',
            'variable': variables,
            'year':[str(year)],
            'month': months,
            'day': days,
            'time': times,
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
            'time': times,
            'grid': [1, 0.5],
            'area': [-50, -180, -90, 180]
            }

    # retrieves the path to the file
    # target = 'download.nc'
    fl = c.retrieve(dataset, params).download()
    ds = xr.open_dataset(fl)
    
    # retrieves the path to the file
    # fl = c.retrieve(dataset, params)
    # # load into memory
    # with urlopen(fl.location) as f:
    #     ds = xr.open_dataset(f.read())

    return ds, fl

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

def rotate_vector(u, v, lon, ref_lon = 0, hemi = "NH"):
    if hemi == "NH":
        angle = (lon-ref_lon)*np.pi/180 # rotation angle (radian)
    else:
        angle = -(lon-ref_lon)*np.pi/180
    u2 = u*np.cos(angle) - v*np.sin(angle)
    v2 = u*np.sin(angle) + v*np.cos(angle)
    return u2, v2

def wind_consistency(u):
    u1 = np.nanmean(u, axis = 0) / np.nanstd(u, axis = 0)
    return u1

def get_ERA5(ds, idx_era, xx, yy, region = "NH", ref_lon = 0):
    lat3, lon3 = np.meshgrid(ds.latitude, ds.longitude)
    inProj = Proj('epsg:4326')
    if region == "NH":
        if ref_lon == 0:
            outProj = Proj('epsg:3411')
        elif ref_lon == -45:
            outProj = Proj('proj4: +proj=stere +lon_0=-45 +lat_0=90 +k=1 +R=6378273 +no_defs')
    elif region == "SH":
        outProj = Proj('epsg:3412')
        
    xx3,yy3 = transform(inProj,outProj,lat3,lon3)
    t2m = np.nanmean(np.array(ds.t2m)[idx_era], axis = 0).transpose()
    u10 = wind_consistency(np.array(ds.u10)[idx_era]).transpose()
    v10 = wind_consistency(np.array(ds.v10)[idx_era]).transpose()
    sic = np.nanmean(np.array(ds.siconc)[idx_era], axis = 0).transpose()
    i10 = np.nanmean(np.array(ds.i10fg)[idx_era], axis = 0).transpose()
    
    u10, v10 = rotate_vector(u10, v10, lon3, ref_lon, region)
    
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

def make_dataset(year, sector = "Ross", region = "SH"):
    # Weekly time frame using sea ice drift data ====================
    ncfile = data_path + f"/{region}/Sea_ice_drift/icemotion_weekly_{region.lower()}_25km_{year}0101_{year}1231_v4.1.nc"
    ds1 = xr.open_dataset(ncfile)
    datetimeindex = ds1.time.astype("datetime64[ns]").values #ds1.indexes['time'].to_datetimeindex() #
    d1, d2 = [], []
    for i, d in enumerate(datetimeindex):
        d1.append(pd.to_datetime(d) + dt.timedelta(days = 0))
        d2.append(pd.to_datetime(d) + dt.timedelta(days = 7))
    n_samples = np.arange(0, len(d1), 1)

    # Read IS2 nc file ==================================
    is2_nc = f'D:\\IS2_topo_DL\\Ridges_density_{sector}_{year}.nc'
    is2 = xr.open_dataset(is2_nc)

    with netCDF4.Dataset(is2_nc, 'r') as nc:
        lat = np.array(nc.variables['lat'])
        lon = np.array(nc.variables['lon'])
        x = np.array(nc.variables['x'])
        y = np.array(nc.variables['y'])
        xx, yy = np.meshgrid(x, y)
        times = nc.variables['time']
        times = num2date(times[:], units = times.units)
    
    row, col = np.shape(xx)    

    # Initialize grid input & output ==========================================
    grid_input = np.zeros([len(d1), 10, row, col])
    fields = ['fb_mode', 'fb_std', 'fr_ridge', 'h_ridge']
    output = np.zeros([len(d1), len(fields), row, col])

    for i in n_samples[:]:

        print(i, d1[i], d2[i])

        # ICESat2 ============================================        
        tidx = np.where((times >= d1[i]) & (times < d2[i]))[0]
        fb_count = np.nansum(np.array(is2.variables["fb_count"][tidx, :, :]), axis = 0)
        valid_count = (fb_count > 200)

        for f, field1 in enumerate(fields):
            array = np.array(is2.variables[field1][tidx, :, :])
            # array = np.transpose(np.array(ds.variables[field1][tidx, :, :]), axes = (0, 2, 1))
            data1 = np.nanmean(array, axis = 0)
            data1[~valid_count] = np.nan
            output[i,f] = np.transpose(np.nanmean(array, axis = 0))

        # ERA5 =================================================
        months, days = [], []
        for n in range(0, 7):
            m = str((d1[i] + dt.timedelta(days = n)).month).zfill(2)
            d = str((d1[i] + dt.timedelta(days = n)).day).zfill(2)
            if m not in months:
                months.append(m)
            if d not in days:
                days.append(d)
        ds, fl = retrieve_hourly_ERA5(year, months, days, region)
        # print("Load ERA5")
        
        ## Read ice motion data ===========================================
        sampling_size = 1
        xx, yy, lat, lon, u, v = get_ice_motion(ncfile, i, xx, yy, sampling_size, region)
        grid_u = np.nanmean(u, axis = 0)
        grid_v = np.nanmean(v, axis = 0)
        # Divergence
        div = calculate_div(u, v, dx = 25, dy = 25)
        # Shear rate
        shr = calculate_shr(u, v, dx = 25, dy = 25)
        # print("Sea ice motion updated")
    
        ## Read SIC data ==================================================
        grid_sic0 = np.zeros([7, row, col])
        for n in range(0, 7):
            t1 = d1[i] + dt.timedelta(days = n)
            grid_sic0[n] = get_SIC(t1, xx, yy, region = region)
        grid_sic = np.nanmean(grid_sic0, axis = 0)
        # print("SIC updated")
    
        ## Read ERA5 data =================================================
        times_era = pd.to_datetime(ds.valid_time) # pd.to_datetime(ds.time)
        idx_era = (times_era >= d1[i]) & (times_era < d2[i])
        grid_t2m, grid_u10, grid_v10, _, grid_i10 = get_ERA5(ds, idx_era, xx, yy, region = region)
        # print("ERA5 updated")
    
        grid_input[i, 0, :, :] = grid_u
        grid_input[i, 1, :, :] = grid_v
        grid_input[i, 2, :, :] = grid_sic
        grid_input[i, 3, :, :] = grid_t2m #grid_t2m - 210)/(310 - 210) #Max temp = 320 K, Min temp = 240 K)
        grid_input[i, 4, :, :] = grid_u10
        grid_input[i, 5, :, :] = grid_v10
        grid_input[i, 6, :, :] = grid_i10
        grid_input[i, 7, :, :] = np.nanmean(div, axis = 0)
        grid_input[i, 8, :, :] = np.nanmean(shr, axis = 0)
        # grid_input[i, :, :] = div05
        # grid_input[i, :, :] = shr95
        # grid_input[i, :, :] = shr50
        # grid_input[i, :, :] = shr05

        del ds
        os.remove(fl)
        
    return xx, yy, grid_input, output

# READ ICESAT-2 GRID DATA ###################################################

def get_IS2_grid(year, region, d1, d2, filepath = "D:\\IS2_topo_DL"):
    
    ##### Read data ##############################
    try: ncfile.close()  # just to be safe, make sure dataset is not already open.
    except: pass

    is2_nc = f'D:\\IS2_topo_DL\\Ridges_density_{region}_{year}.nc'
    is2 = xr.open_dataset(is2_nc)

    with netCDF4.Dataset(is2_nc, 'r') as nc:
        lat = np.array(nc.variables['lat'])
        lon = np.array(nc.variables['lon'])
        x = np.array(nc.variables['x'])
        y = np.array(nc.variables['y'])
        xx, yy = np.meshgrid(x, y)

        times = nc.variables['time']
        times = num2date(times[:], units = times.units)
    
    fields = ['fb_mode', 'fb_std', 'fr_ridge', 'h_ridge']
    output = np.zeros([len(d1), len(fields), xx.shape[0], xx.shape[1]])
    
    for k, field1 in enumerate(fields):
        for i in range(0, len(d1)):                
    
            tidx = np.where((times >= 11[i]) & (times < d[i]))[0]
            fb_count = np.nansum(np.array(ds.variables["fb_count"][tidx, :, :]), axis = 0)
            valid_count = (fb_count > 500)
            
            if np.sum(tidx) > 0:
                array = np.array(ds.variables[field1][tidx, :, :])
                # array = np.transpose(np.array(ds.variables[field1][tidx, :, :]), axes = (0, 2, 1))
                data1 = np.nanmedian(array, axis = 0)
                data1[~valid_count] = np.nan
    
            output[k, i] = np.transpose(np.nanmean(array, axis = 0))
            
    return xx, yy, output
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