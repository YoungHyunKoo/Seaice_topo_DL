import warnings
warnings.filterwarnings('ignore')

import os, glob
import csv
import numpy as np
# import icepyx as ipx
from os import listdir
from os.path import isfile, join
import h5py
import matplotlib.pylab as plt
from math import *
import random
# import time
import pandas as pd
from tqdm import tqdm
import pickle

import cartopy.crs as ccrs
import datetime as dt
from shapely.geometry import Point
import geopandas
import scipy.stats as stats
import netCDF4
from netCDF4 import Dataset    # Note: python is case-sensitive!
from netCDF4 import date2num,num2date
from scipy.ndimage import gaussian_filter1d

from pyproj import Proj, transform
from shapely.geometry import Polygon, Point
from sklearn.neighbors import KernelDensity

import geopandas
import time
import shapefile

# Functions 

def dist(lon1,lat1,lon2,lat2):

    lon1_rad = np.radians(lon1)
    lon2_rad = np.radians(lon2)
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    #Assumes degrees input
    #Calculates in metres
    R = 6371000 #Radius of earth in metres (roughly)
    ## Uses Haversine formula
    a1 = (sin((lat2_rad-lat1_rad)/2))**2
    a2 = (cos(lat1_rad))*(cos(lat2_rad))*((sin((lon2_rad-lon1_rad)/2))**2)
    a = a1 + a2
    c = 2*atan2(sqrt(a),sqrt(1-a))
    d = R*c

    return d

# def get_chord_lengths(ice_leads_msk,fb_height,seg_dist_x):
#     floe_chord_lengths = np.zeros(len(ice_leads_msk)) # Making big enough array
#     floe_fb = np.zeros(len(ice_leads_msk)) # Making big enough array
#     ice_cnt_st = 0
#     ice_cnt_en = 0
#     floe_idx = 1
    
#     # ice_lead_msk: 0 (ice), 1 (lead)
#     for i in range(1,len(ice_leads_msk)):
#         if (ice_leads_msk[i] == 1) and (ice_leads_msk[i-1] == 0): # start floe
#             ice_cnt_st = i
#             ice_cnt_en  = i
#         elif (ice_leads_msk[i] == 1) and (ice_leads_msk[i-1] == 1): # grow floe
#             ice_cnt_en += 1
#         elif (ice_leads_msk[i-1] == 1) and (ice_leads_msk[i] == 0): # stop floe
#             floe_chord_lengths[floe_idx] = seg_dist_x[ice_cnt_en] - seg_dist_x[ice_cnt_st]
#             floe_fb[floe_idx] = np.mean(fb_height[ice_cnt_st:ice_cnt_en+1]) 
#             floe_idx += 1
            
#     # Removing spurious floes (< 10m, > 10 km, fb<0.1)
#     remove_idx = np.where(floe_chord_lengths < 10)[0]  
#     remove_idx = np.append(remove_idx,np.where(floe_chord_lengths > 10e3)[0])
#     remove_idx = np.append(remove_idx,np.where(floe_fb < 0.1)[0])
#     floe_fb = np.delete(floe_fb,remove_idx)
#     floe_chord_lengths = np.delete(floe_chord_lengths,remove_idx)
#     #
#     return floe_chord_lengths, floe_fb

# def get_chord_lengths(fb_height, seg_dist_x, stype, lead_values, seq = 2):
    
#     delta_seg_dist_x = np.append(0,np.diff(seg_dist_x)) 
#     spurious_msk = np.ones(np.shape(seg_dist_x))
#     spurious_msk[delta_seg_dist_x > 500] = 0
#     # Removing NaN values
#     spurious_msk[np.isnan(fb_height)] = 0

#     # Creating binary array for: leads or spurious (1) / ice (0)
#     ice_leads_msk = ice_leads_msk*spurious_msk
#     # 1 = lead, 0 = ice
    
#     floe_chord_lengths = np.zeros(len(ice_leads_msk)) # Making big enough array
#     floe_fb = np.zeros(len(ice_leads_msk)) # Making big enough array
#     ice_cnt_st = 0
#     ice_cnt_en = 0
#     floe_idx = 1
#     for i in range(1,len(ice_leads_msk)):
#         if (ice_leads_msk[i] == 0) and (np.prod(ice_leads_msk[max(0,i-seq):i]) == 1): # start floe
#             ice_cnt_st = i
#             ice_cnt_en = i
#         elif (ice_leads_msk[i] == 0) and (ice_leads_msk[i-1] == 0): # grow floe
#             ice_cnt_en += 1
#         elif (ice_leads_msk[i-1] == 0) and (np.prod(ice_leads_msk[i:min(i+seq,len(ice_leads_msk))]) == 1): # stop floe
#             floe_chord_lengths[floe_idx] = seg_dist_x[ice_cnt_en] - seg_dist_x[ice_cnt_st]
#             floe_fb[floe_idx] = np.mean(fb_height[ice_cnt_st:ice_cnt_en+1]) 
#             floe_idx += 1
            
            
# Function to calculate sea ice floe length
def get_floe_length(freeboard, lead_mask, seg_dist, seg_len, lat, lon, nprof = 50):
    # INPUT:
    # freeboard: along-track freeboard measurement of ICESat-2 ATL10 track
    # lead_mask: along-track lead detection result (0: lead; 1: non-lead)
    # seg_dist: along-track distance of ICESat-2 ATL10 track (unit: meters)
    # nprof: number of points in normalized profiles
    
    delta_dist = np.append(0, np.diff(seg_dist))
    
    # Floe parameters
    floe_length = np.array([]) # Floe length (unit: m)    
    floe_fb_mean = np.array([]) # Floe freeboard (unit: m) 
    floe_fb_median = np.array([]) # Floe freeboard (unit: m) 
    floe_fb_std = np.array([]) # Floe freeboard (unit: m)
    floe_lat = np.array([]) # Floe latitude
    floe_lon = np.array([]) # Floe longitude
    floe_loc = np.array([]) # Floe longitude
    
    floe_idx = np.zeros(freeboard.shape) - 1
    floe_cnt = 0
    
    # Lead parameters
    lead_length = np.array([]) # Lead length (unit: m)
    lead_position = np.array([]) # Lead position (unit: m along the track)
    lead_fb = np.array([]) # Lead freeboard (unit: m)
    
    # Floe profile parameter
    rprof = np.linspace(0,1,nprof)
    floe_profiles = []
    
    # Floe starting index
    ice_cnt_st = 0
    # Floe ending index
    ice_cnt_en = 0
    
    # Lead starting index
    lead_cnt_st = 0
    # Lead ending index
    lead_cnt_en = 0    
    
    for i in range(1, len(freeboard)):
        # Remove invalid floe boundary:
        # If two ATL10 points are separated farther than 4 x (their segment length),
        # these two points are considered invalid
        # (because we cannot confirm the real sea ice condition (lead occurence, etc.) between them)
        if delta_dist[i] > 4*(seg_len[i] + seg_len[i-1]):
            freeboard[i] = np.nan
            freeboard[i-1] = np.nan
        
        if (lead_mask[i] == 0) and (lead_mask[i-1] == 1): # start floe & stop lead
            # Initialize floe
            ice_cnt_st = i
            ice_cnt_en = i
            
            # Complete lead
            lead_length = np.append(lead_length, abs(seg_dist[lead_cnt_en] - seg_dist[lead_cnt_st]) + 0.5*seg_len[lead_cnt_st] + 0.5*seg_len[lead_cnt_en])
            # lead_length = np.append(lead_length, np.sum(seg_len[lead_cnt_st:lead_cnt_en+1]))
            lead_fb = np.append(lead_fb, np.mean(freeboard[lead_cnt_st:lead_cnt_en+1]))
            lead_position = np.append(lead_position, (seg_dist[lead_cnt_en] + seg_dist[lead_cnt_st])/2)
                                    
        elif (lead_mask[i] == 0) and (lead_mask[i-1] == 0): # grow floe
            ice_cnt_en += 1
            
        elif (lead_mask[i] == 1) and (lead_mask[i-1] == 1): # grow lead
            lead_cnt_en += 1
            
        elif (lead_mask[i] == 1) and (lead_mask[i-1] == 0): # stop floe & start lead
            # Complete floe
            floe_length = np.append(floe_length, abs(seg_dist[ice_cnt_en] - seg_dist[ice_cnt_st]) + 0.5*seg_len[ice_cnt_st] + 0.5*seg_len[ice_cnt_en])
            # floe_length = np.append(floe_length, np.sum(seg_len[ice_cnt_st:ice_cnt_en+1]))
            floe_fb_mean = np.append(floe_fb_mean, np.mean(freeboard[ice_cnt_st:ice_cnt_en+1]))
            floe_fb_median = np.append(floe_fb_median, np.median(freeboard[ice_cnt_st:ice_cnt_en+1]))
            floe_fb_std = np.append(floe_fb_std, np.std(freeboard[ice_cnt_st:ice_cnt_en+1]))
            floe_lat = np.append(floe_lat, np.mean(lat[ice_cnt_st:ice_cnt_en+1]))
            floe_lon = np.append(floe_lon, np.mean(lon[ice_cnt_st:ice_cnt_en+1]))
            floe_loc = np.append(floe_loc, (seg_dist[ice_cnt_st] + seg_dist[ice_cnt_en+1])/2)
            
            floe_cnt += 1
            floe_idx[ice_cnt_st:ice_cnt_en+1] = floe_cnt
            
            prof = freeboard[ice_cnt_st:ice_cnt_en+1]
            xx_prof = np.linspace(0,1,ice_cnt_en-ice_cnt_st+1)
            prof2 = np.interp(rprof, xx_prof, prof)
            floe_profiles.append(np.interp(rprof, xx_prof, prof))
            # floe_idx += 1
            
            # Initialize lead
            lead_cnt_st = i
            lead_cnt_en = i
        
    # Removing spurious floes (< 50m, > 10 km, fb < 0.1)
    
    # print(floe_length.shape, len(floe_profiles))
    idx = np.where((floe_length >= 10) & (floe_length <= 10000) & (floe_fb_mean >= 0.1))[0]  
    floe_fb_mean = floe_fb_mean[idx] #np.delete(floe_fb_mean, remove_idx)
    floe_fb_median = floe_fb_median[idx] #np.delete(floe_fb_median, remove_idx)
    floe_fb_std = floe_fb_std[idx] #np.delete(floe_fb_std, remove_idx)
    floe_length = floe_length[idx] #np.delete(floe_length, remove_idx)
    floe_lat = floe_lat[idx]
    floe_lon = floe_lon[idx]
    floe_loc = floe_loc[idx]
    if len(floe_profiles) > 0:
        floe_profiles = np.array(floe_profiles)[idx, :].transpose() #np.array(floe_profiles)[~remove_idx, :].transpose()
    else:
        floe_profiles = np.array(floe_profiles)
    # print(floe_length.shape, floe_profiles.shape)
    
    # Removing spurious leads
    idx = np.where((lead_length >= 10) | (lead_fb <= 0.1))[0]
    lead_length = lead_length[idx] #np.delete(lead_length, remove_idx)    
    lead_position = lead_position[idx] #np.delete(lead_position, remove_idx)

    return floe_length, floe_fb_mean, floe_fb_median, floe_fb_std, floe_lat, floe_lon, floe_idx, floe_loc, lead_length, lead_position, floe_profiles

def get_lead_width_spacing_correlation(lead_widths,lead_positions,lead_width_bin_ranges):
    binned_lead_spacings = np.zeros(len(lead_width_bin_ranges)-1)
    binned_lead_count = np.zeros(len(lead_width_bin_ranges)-1)
    for i in range(len(lead_width_bin_ranges)-1):
        idx = np.where( (lead_widths >= lead_width_bin_ranges[i]) * (lead_widths < lead_width_bin_ranges[i+1]) )[0]
        spacings = np.abs(np.diff(lead_positions[idx]))
        if np.any(spacings):
            binned_lead_spacings[i] = np.mean(spacings)
            binned_lead_count[i] = len(idx) - 1 
    return binned_lead_count, binned_lead_spacings

# Calculate modal freeboard based on freeboard distribution
def calculate_mode(data, N = 10, fb_max = 1.5):
    data = data[~np.isnan(data)]
    w = 0.02
    M = 4.0
    m = w    
    
    if len(data) > N: # minimum number of freeboard observations to create distribution
        # instantiate and fit the KDE model
        kde = KernelDensity(bandwidth=w*5, kernel='gaussian')
        kde.fit(data[:, None])
        x_d = np.arange(m, M, w)
        logprob = kde.score_samples(x_d[:, None])
        n_max = np.argmax(np.exp(logprob)[:int(fb_max//w)]) # Set the possible maximum modal freeboard
        mode = x_d[n_max] + w/2
    elif len(data) > 0: # if the number of observation is not enough, just take 0.2 quantile
        mode = np.quantile(data, 0.2)
    else:
        mode = np.nan

    return mode


def smooth_line(data, x, w = 2):
    # Smooth the surface with the defined window size
    output = np.zeros(len(data))
    for n in range(0, len(data)):
#         output[n] = np.mean(data[max(0, n-w):min(len(data), n+w+1)])
        output[n] = np.mean(data[(x <= x[n]+w)&(x >= x[n]-w)])
    return output


def modal_profile(fb, seg_x, refsur_ndx):
    fb_mode = np.zeros(np.shape(fb))
    sample_ndx = np.zeros(np.shape(fb))

    # # Calculate modal freeboard ==============================
    for c, i in enumerate(np.unique(refsur_ndx)):
        part = (refsur_ndx == i)
        x_min = np.min(seg_x[part])
        x_max = np.max(seg_x[part])
        sample_ndx[part] = i*10 + (seg_x[part] - x_min)//(10000/3)
        # sample_ndx[part] = c

    dist_mode = np.zeros(np.shape(np.unique(sample_ndx)))
    val_mode = np.zeros(np.shape(np.unique(sample_ndx)))
    
    for c, i in enumerate(np.unique(sample_ndx)):                
        subndx = (sample_ndx == i)
        subdata = fb[subndx]
        val_mode[c] = calculate_mode(subdata)
        dist_mode[c] = np.median(seg_x[subndx])

    val_mode = smooth_line(val_mode, dist_mode, 12000)

    for c, i in enumerate(np.unique(sample_ndx)):  
        subndx = (sample_ndx == i)
        fb_mode[subndx] = val_mode[c]

    return fb_mode, sample_ndx

# Calculate ridge fraction etc.
def calculate_ridge(df):
    
    mode, ridge_fr, ridge_h = 0,0,0;
    std, mean, med = 0,0,0;
    diff = df['fb'].values - df['fb_mode'].values
    
    if len(df) > 0:
        
        mode = np.nanmean(df['fb_mode'].values)
        std = np.nanstd(df['fb'].values)
        mean = np.nanmean(df['fb'].values)
        med = np.nanmedian(df['fb'].values)
        ridge_h = np.nanmean(diff[df['ridge']==1])
        ridge_fr = len(diff[df['ridge']==1]) / len(diff) * 100
        
#         mode = calculate_mode(freeboard)
#         ridge_fr = len(freeboard[freeboard > mode+0.6])/len(freeboard)*100
#         ridge_h = np.nanmean(freeboard[freeboard > mode+0.6]) - mode
    
    return [mode, ridge_fr, ridge_h, mean, med, std]

def calculate_lead(lead):
    return len(lead[lead == 1])/len(lead)*100

##### Iceberg detection from ATL10 #########################
def determine_iceberg(df, th_fb = 1.0, th_sigma = 0.02, th_std = 0.1):
    # INPUT:
    # freeboard: along-track freeboard measurement of ICESat-2 ATL10 track
    # ib_mask: along-track initial iceberg mask (0: sea ice; 1: iceberg)
    # seg_dist: along-track distance of ICESat-2 ATL10 track (unit: meters)
    # nprof: number of points in normalized profiles

    ib_mask = (df['fb'].values >= th_fb) & (df['sigma'].values <= th_sigma) & (df['fb_std'].values <= th_std)
    
    seg_dist = df['seg_x'].values
    freeboard = df['fb'].values
    seg_len = df['seg_len'].values
    
    delta_dist = np.append(0, np.diff(seg_dist))
    ib_mask2 = ib_mask.copy()

    ib_cnt_st = -1
    ib_cnt_en = -1

    df_ib = pd.DataFrame({})

    c = 0
    for i in range(1, len(freeboard)):
        # print(i)

        if (ib_mask[i] == True) and (ib_mask[i-1] != True):
            ib_cnt_st = i
            ib_cnt_en = i

        elif (ib_mask[i] == True) and (ib_mask[i-1] == True):
            ib_cnt_en = i

        elif (ib_mask[i] != True) and (ib_mask[i-1] == True):
            
            if np.sum(seg_len[ib_cnt_st:ib_cnt_en+1])/2 < 200: #abs(seg_dist[ib_cnt_en] - seg_dist[ib_cnt_st]) < 100:
                # print(seg_dist[ib_cnt_en] - seg_dist[ib_cnt_st], ib_cnt_en, ib_cnt_st)
                ib_mask2[ib_cnt_st:ib_cnt_en+1] = False
            else:
                # buffer around the index
                # buf = 2
                # ib_cnt_st = max(0, ib_cnt_st-buf)
                # ib_cnt_en = min(len(freeboard)-1, ib_cnt_st+buf)
                
                df_ib.loc[c, "lat"] = df.loc[ib_cnt_st:ib_cnt_en+1, 'lat'].median()
                df_ib.loc[c, "lon"] = df.loc[ib_cnt_st:ib_cnt_en+1, 'lon'].median()
                df_ib.loc[c, "seg_x"] = df.loc[ib_cnt_st:ib_cnt_en+1, 'seg_x'].median()
                df_ib.loc[c, "id_st"] = ib_cnt_st
                df_ib.loc[c, "id_en"] = ib_cnt_en
                df_ib.loc[c, "fb_mean"] = df.loc[ib_cnt_st:ib_cnt_en+1, 'fb'].mean()
                df_ib.loc[c, "fb_max"] = df.loc[ib_cnt_st:ib_cnt_en+1, 'fb'].max()
                df_ib.loc[c, "fb_min"] = df.loc[ib_cnt_st:ib_cnt_en+1, 'fb'].min()
                df_ib.loc[c, "fb_std"] = df.loc[ib_cnt_st:ib_cnt_en+1, 'fb'].std()
                df_ib.loc[c, "width"] = df.loc[ib_cnt_st:ib_cnt_en+1, 'seg_len'].sum()/2
                c += 1
        else:
            pass
                
    return ib_mask2, df_ib

def combine_icebergs(df, df_ib, ib_mask, th_fb = 1.0):

    # Buffer zone for landfast ice
    if len(df_ib) > 0:
        df_ib["id_st"] = df_ib["id_st"] - 1
        df_ib["id_en"] = df_ib["id_en"] + 1
    
    for c in range(1, len(df_ib)):
        fb_btw = df.loc[int(df_ib.loc[c-1, "id_en"]): int(df_ib.loc[c, "id_st"])+1, "fb"]
        # print(c, int(df_ib.loc[c-1, "id_en"]), int(df_ib.loc[c, "id_st"])+1, fb_btw > 1.0)
        x1 = df.loc[int(df_ib.loc[c-1, "id_en"]), "seg_x"]
        x2 = df.loc[int(df_ib.loc[c, "id_st"]), "seg_x"]
        if all(fb_btw > th_fb) & (abs(x1-x2) < 1000):
            
            ib_cnt_st = df_ib.loc[c-1, "id_st"]
            ib_cnt_en = df_ib.loc[c, "id_en"]

            # print(c, ib_cnt_st, ib_cnt_en)
            
            df_ib.loc[c, "lat"] = df.loc[ib_cnt_st:ib_cnt_en+1, 'lat'].median()
            df_ib.loc[c, "lon"] = df.loc[ib_cnt_st:ib_cnt_en+1, 'lon'].median()
            df_ib.loc[c, "seg_x"] = df.loc[ib_cnt_st:ib_cnt_en+1, 'seg_x'].median()
            df_ib.loc[c, "id_st"] = ib_cnt_st
            df_ib.loc[c, "id_en"] = ib_cnt_en
            df_ib.loc[c, "fb_mean"] = df.loc[ib_cnt_st:ib_cnt_en+1, 'fb'].mean()
            df_ib.loc[c, "fb_max"] = df.loc[ib_cnt_st:ib_cnt_en+1, 'fb'].max()
            df_ib.loc[c, "fb_min"] = df.loc[ib_cnt_st:ib_cnt_en+1, 'fb'].min()
            df_ib.loc[c, "fb_std"] = df.loc[ib_cnt_st:ib_cnt_en+1, 'fb'].std()
            df_ib.loc[c, "width"] = df.loc[ib_cnt_st:ib_cnt_en+1, 'seg_len'].sum() #abs(df.loc[ib_cnt_st, 'seg_x'] - df.loc[ib_cnt_en, 'seg_x'])

            df_ib = df_ib.drop(c-1)
            
    df_ib = df_ib.reset_index(drop = True)
    
    for c in range(0, len(df_ib)):
        if df_ib.loc[c, "width"] < 200:
            ib_mask[int(df_ib.loc[c, "id_st"]): int(df_ib.loc[c, "id_en"])+1] = 0
            df_ib = df_ib.drop(c)
        else:
            ib_mask[int(df_ib.loc[c, "id_st"]): int(df_ib.loc[c, "id_en"])+1] = 1
            
    df_ib = df_ib.reset_index(drop = True)
    
    return df_ib, ib_mask

def read_ATL10(filename, bbox):
    with h5py.File(filename,'r') as f:
        # print(filename)
        # Check the orbit orientation
        orient = f['orbit_info/sc_orient'][0]
        strong_beams = []

        if orient == 0:
            for i in [1, 2, 3]:
                if f"gt{i}l" in f.keys():
                    strong_beams.append(f"gt{i}l")
        elif orient == 1:
            for i in [1, 2, 3]:
                if f"gt{i}r" in f.keys():
                    strong_beams.append(f"gt{i}r")
                    
        first = True
        
        for beam_num, beam in enumerate(strong_beams):

            lat = f[beam]['freeboard_segment/latitude'][:]
            lon = f[beam]['freeboard_segment/longitude'][:]
            fb = f[beam]['freeboard_segment/beam_fb_height'][:]

            if bbox[0] <= bbox[2]:
                idx = (lat >= bbox[1]) & (lat <= bbox[3]) & (lon >= bbox[0]) & (lon <= bbox[2]) & (fb <= 20)
            else:
                idx = (lat >= bbox[1]) & (lat <= bbox[3]) & ((lon >= bbox[0]) | (lon <= bbox[2])) & (fb <= 20)                    

            if any(idx):
                lat = lat[idx]
                lon = lon[idx]
                fb = fb[idx]
        
                seg_x = f[beam]['freeboard_segment/seg_dist_x'][idx] # (m to km)
                seg_x = seg_x - seg_x.min()
                seg_len = f[beam]['freeboard_segment/heights/height_segment_length_seg'][idx]
                ph_rate = f[beam]['freeboard_segment/heights/photon_rate'][idx]
                sigma = f[beam]['freeboard_segment/heights/height_segment_sigma'][idx]
                # fb[fb > 100] = np.nan
                stype = f[beam]['freeboard_segment/heights/height_segment_type'][idx]
                refsur_ndx = f[beam]['freeboard_segment/beam_refsurf_ndx'][idx]
                fb_std = pd.Series(fb).rolling(3, center = True).std().values
                
                fb_mode, sample_ndx = modal_profile(fb, seg_x, refsur_ndx)

                # Calculate modal freeboard ==============================
                # w = 10000
                # sample_ndx = seg_x // (w/2)
                # for i in np.unique(sample_ndx):
                #     if i-1 < 0:
                #         start_ndx = 0
                #     elif i + 1 > max(sample_ndx):
                #         start_ndx = max(sample_ndx) - 2
                #     else:
                #         start_ndx = i-1
                #     end_ndx = start_ndx + 2
                    
                #     subndx = (sample_ndx == i)
                #     subdata = fb[(sample_ndx >= start_ndx) & (sample_ndx <= end_ndx)]
                #     fb_mode[subndx] = calculate_mode(subdata)
                    

                ridge = np.zeros(np.shape(fb))
                # Ridge or not? (threshold 0.6 m above level (mode) freeboard)
                ridge[fb > fb_mode + 0.6] = 1
        
                df0 = pd.DataFrame({'beam': beam, 'lat': lat, 'lon': lon, 'seg_x': seg_x,
                                    'seg_len': seg_len, 'fb': fb, 'ph_rate': ph_rate, 'sigma': sigma, 'stype': stype,
                                    'refsur_ndx': refsur_ndx, 'sample_ndx': sample_ndx, 'fb_std': fb_std,
                                    'fb_mode': fb_mode, 'ridge': ridge})

                if first:
                    df = df0
                    first = False
                else:
                    df = pd.concat([df, df0], ignore_index=True)

    if first:
        return []
    else:
        return df

def read_shapefile(shp_path):   

    #read file, parse out the records and shapes
    sf = shapefile.Reader(shp_path)
    fields = [x[0] for x in sf.fields][1:]
    records = sf.records()
    # shps = [s.points for s in sf.shapes()]
    shps = [s.points for s in sf.shapes()]
    X = []
    Y = []
    
    for p in shps:
        X.append(p[0][0])
        Y.append(p[0][1])

    #write into a dataframe
    df = pd.DataFrame(columns=fields, data=records)

    return df.reset_index(drop = True)

# Function to convert pandas dataframe to ESRI shapefile
def convertshp(df, outfile):
    '''
    === input
        - df: input pandas dataframe
        - outfile: the nampe of the output shapefile
    '''   

    df=df.reset_index(drop=True)

    if len(df) > 0:
        df['geometry'] = df.apply(lambda x: Point((float(x.lon), float(x.lat))), axis=1)
        collist = list(df.columns)
        if 'time' in collist:
            collist.remove('time')
        df2 = df[collist]

        df2 = geopandas.GeoDataFrame(df2, geometry='geometry')

        # proj WGS84
        df2.crs= "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"

        df2.to_file(outfile, driver='ESRI Shapefile')

    # print('... converted to ' + outfile)



