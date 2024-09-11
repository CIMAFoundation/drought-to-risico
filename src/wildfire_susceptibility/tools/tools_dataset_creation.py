# -*- coding: utf-8 -*-
"""
Created on Sun May 28 21:46:51 2023

@author: Giorg
"""
import rasterio as rio
import geopandas as gpd
import pandas as pd
import numpy as np
import os
import logging
from dataclasses import dataclass
import sys
from dotmap import DotMap
from scipy import signal


os.chdir('/share/home/gruppo4/Bolivia/DATA/BolMLmonthly/test_model/annual_wildfire_susceptibility/tools')
from tools_random_forest import RFForestAlgorithm
from useful_decorators import ram_consumption
from utils import batch_by_rows, create_topographic_vars, reproject_layer, rasterize_numerical_feature


 
@dataclass
class DatasetTools():

    config: dict

    def __post_init__(self):
        self.config = DotMap(self.config)
     
    def initiate_dataset(self, dem_path: str, veg_path: str, other_layer_dict_paths: dict[list], 
                         outdir: str, batch_number = 0): 
        
        '''
        this function will prepare the data for the ML model before splitting in X and Y sets.
        other_layer_dict_paths is a dictionary with the name of the layer and a tuple of the path to the layer and if it is categorical or not
        It provides tools for running the module supranational_model.py wich creates dataset and train model
        '''

        algorithm = RFForestAlgorithm()

        with rio.open(dem_path) as dem:
            dem_arr = dem.read(1)
            dem_nodata = dem.nodata

        # divide in batches if self.config.batches > 1
        dem_arr, _, _ =  batch_by_rows(dem_arr, self.config.batches, batch_number)
        #print(np.unique(dem_arr))
            
        slope_arr, northing_arr, easting_arr =  create_topographic_vars(dem_path, outdir)

        slope_arr, _, _ =  batch_by_rows(slope_arr, self.config.batches, batch_number)
        northing_arr, _, _ =  batch_by_rows(northing_arr, self.config.batches, batch_number)
        easting_arr, _, _ =  batch_by_rows(easting_arr, self.config.batches, batch_number)

        
        # land cover 
        with rio.open(veg_path) as lc:
            veg_arr = lc.read(1)
            veg_arr, _, _ =  batch_by_rows(veg_arr, self.config.batches, batch_number)
            veg_arr =  reproject_layer(dem_arr, veg_arr, dem_path, veg_path, outdir)
        
        # create a dict with optional layer taking care of reprojecting them 
        if other_layer_dict_paths is not None:    
            logging.info('adding optional layer')
            other_layers_dict = {}
            for other_layer_name, other_layer_info in other_layer_dict_paths.items():
                other_layer = other_layer_info[0]
                is_categorical_variable = other_layer_info[1]
                with rio.open(other_layer) as l:
                    layer_arr = l.read(1) 
                    # batch it if needed before adding to dictionary
                    layer_arr, _, _ =  batch_by_rows(layer_arr, self.config.batches, batch_number)
                    layer_arr =  reproject_layer(dem_arr, layer_arr, dem_path, other_layer, outdir)
                    
                # eliminate negative values
                layer_arr = np.where(layer_arr >= 0, layer_arr, np.NaN)
                layer_arr = np.where(np.isnan(layer_arr) == True, np.nanmean(layer_arr), layer_arr)
                
                # update the dict using one hot encoding if needed
                if is_categorical_variable == True:
                    other_layers_dict = algorithm.one_hot_encoding(layer_arr, other_layer_name, other_layers_dict) 
                else:
                    other_layers_dict[other_layer_name] = layer_arr
                           
        # LC codes which are not burnable        
        nb_codes_list = [int(i) for i in self.config.nb_codes_list] 
        nb_codes_list = [str(i) for i in nb_codes_list] 
            
        
        # create a mask (where susceptibility will be valid)  and a dict of all the features with data
        mask, data_dict = algorithm.create_dict_vars(dem_arr, 
                                                  dem_nodata, 
                                                  veg_arr, 
                                                  nb_codes_list, 
                                                  slope_arr, 
                                                  northing_arr, 
                                                  easting_arr, 
                                                  other_layers_dict, 
                                                  )
        
        logging.debug(f'batch idx: {batch_number} shape arrays: {mask.shape}')
       
             
        return dem_arr, mask, data_dict, nb_codes_list



    @ram_consumption
    def create_XY_annual(self, fires_country_path: str, annual_lists_indices: dict[dict[str]], dem_path: str,
                         dem_arr: np.array, mask: np.array, X_country: np.array, country_name: str, fire_points: list,
                         threshold_num_pixels_f: int, threshold_num_pixels_nof: int, batch_number = 0):
                         
        
        '''
        this function will create annual X and Y sets of a country based on annual climate indices and annual fires
        '''
        algorithm = RFForestAlgorithm()

        fires = gpd.read_file(fires_country_path)
        with rio.open(dem_path) as dem:   
            fires = fires.to_crs(dem.crs)

        fires[self.config.name_col_y_fires] = pd.to_datetime(fires[self.config.name_col_y_fires]) 
        years_or_months = lambda x: self.config.wildfire_years if x == False else [f'{str(year)}_{month}' for year in self.config.wildfire_years for month in range(1,13)]


        # here fire points is, for each time step, the number of burned points
        counter = 0
        for year, num_fire_px in zip(years_or_months(self.config.convert_to_month), fire_points):

            logging.info(f'\n time step: {counter}: {year}\n')

            # dont create the dataset if the number of fire points in this here is outside the bounds
            if num_fire_px <= threshold_num_pixels_f and num_fire_px >= threshold_num_pixels_nof:
                logging.info('skip this year')
                pass
            else:

                annual_feature_list_paths = annual_lists_indices[year] #T addressed as YEAR_MONTH if confing.month = true, AND SO ALSO THE DICTIONARY KEY IN THE MAIN INPUT and data folder structure
            
                
                # annual_feature_list_paths.sort() # do not sort the layer names

                # filter shapefile with the current year and rasterie it
                if self.config.convert_to_month == False:
                    fire_annnual_shp = fires[ fires[self.config.name_col_y_fires].dt.year == year]
                else:
                    _year = int(year.split('_')[0])
                    _month = int(year.split('_')[1])
                    
                    fire_monthly_shp = fires[ fires[self.config.name_col_y_fires].dt.month == _month]
                    # select the year I am working on 
                    fire_annnual_shp = fire_monthly_shp[ fire_monthly_shp[self.config.name_col_y_fires].dt.year == _year]

                # rasterize the shapefile for the specific year
                try:                
                    fires_arr =  rasterize_numerical_feature(fire_annnual_shp, dem_path)
                    fires_arr, _, _ =  batch_by_rows(fires_arr, self.config.batches, batch_number)
                except ValueError as e:
                    logging.info(f'{e}: {year} this year is empty')
                    fires_arr = np.zeros(dem_arr.shape)                  

                # praparing a dict of annual feature for the current year
                annual_layers_dict = {}
                temporal_ggr = lambda x: '_y' if x == False else '_m'
                # name_annual_features = [os.path.basename(i).split('.')[0] + temporal_ggr(self.config.convert_to_month) for i in annual_feature_list_paths]
                
                
                # create a dict, annual variable name and data
                for annual_layer_name, annual_layer in annual_feature_list_paths.items():
                    annual_layer_name = annual_layer_name + temporal_ggr(self.config.convert_to_month)
                    with rio.open(annual_layer) as a:
                        
                        annual_arr = a.read(1)
                        annual_arr, _, _ =  batch_by_rows(annual_arr, self.config.batches, batch_number)
                        annual_arr =  reproject_layer(dem_arr, annual_arr, dem_path, annual_layer, os.path.dirname(fires_country_path))
                        
                        # eliminate negative values
                        if self.config.drop_neg_and_na_annual == True:
                            annual_arr = np.where(annual_arr >= 0, annual_arr, np.NaN)
                            annual_arr = np.where(np.isnan(annual_arr) == True, np.nanmean(annual_arr), annual_arr)

                        if os.path.basename(annual_layer).startswith('veg'):
                            logging.info('found an annual layer starting with veg word, I consider it as dynamic vegetation and use one hot encoding')
                            annual_layers_dict = algorithm.one_hot_encoding(annual_arr, annual_layer_name, annual_layers_dict) 
                            # create also monthly perc
                            veg_int = np.where(mask == 1, annual_arr, 0)
                            window_size = 2
                            types = np.unique(veg_int)
                            types_presence = {}
                            count = np.ones((window_size*2+1, window_size*2+1))
                            take_center = 1
                            count[window_size, window_size] = take_center 
                            count = count / np.sum(count)
                            for t in types:
                                if t != 0:
                                    density_entry = 'perc_' + str(int(t)) 
                                    types_presence[density_entry] = 100 * signal.convolve2d(veg_int==t, count, boundary='fill', mode='same')
                            
                            annual_layers_dict.update(types_presence)
                        else:
                            annual_layers_dict[annual_layer_name] = annual_arr
                

                n_pixels = len(dem_arr[mask])
                n_features = len((annual_layers_dict.keys()))
                X_annual = np.zeros((n_pixels, n_features))
                Y_annual = fires_arr[mask]
                
                logging.info('Creating dataset only with annual features\n')
                columns_annual = annual_layers_dict.keys()
                
                # create a dataset of X with annual features 
                for col, k in enumerate(annual_layers_dict):
                    data = annual_layers_dict[k]                   
                    X_annual[:, col] = data[mask] 
                
                # update the X country with the annual values
                num_annual_indices = len(columns_annual)
                logging.debug(f'NUM ANNUAL INDICES INCLUDED VEG AFTER ONEHOT: {num_annual_indices}')

                if counter == 0:
                    # at first loop I add the column for annual indices
                    X_all = np.concatenate([X_country, X_annual], axis = 1)
                else:
                    # update the annual indices 
                    X_all[:, -num_annual_indices::] = X_annual

                # now take the burned points with that annual features 
                fires_annual_rows = Y_annual != 0
                
                # presences for that year                
                X_all_presence = X_all[fires_annual_rows]
                
                # I want to sample the burned points 
                sample_n_fire_points = self.config.reduce_fire_points
                reduction = int((X_all_presence.shape[0]*sample_n_fire_points)/100)  
                

                if num_fire_px >= threshold_num_pixels_f:

                    logging.info(f' found year/month with highest fire occurence (num px fires this period, num px fires threasold): {num_fire_px}, {threshold_num_pixels_f}')

                    # sampling and update presences 
                    X_presence_indexes = np.random.choice(X_all_presence.shape[0], size=reduction, replace=False)
                    X_all_presence = X_all_presence[X_presence_indexes, :]  
                    
                    # initiate the dataset of the country with the presences and update it at each year
                    logging.debug(f' len X annual country: {len(X_all_presence)}')               
                    if counter == 0:
                        X_y_country = X_all_presence
                        Y_country = np.ones(X_all_presence.shape[0],)
                    else:
                        X_y_country = np.concatenate([X_y_country, X_all_presence], axis = 0)  
                        Y_country = np.concatenate([Y_country, np.ones(X_all_presence.shape[0],)], axis = 0)
                    logging.debug(f' len X country updated (burned points): {len(X_y_country)}')
                    logging.debug(f' len Y country updated (burned points): {len(Y_country)}')

                elif num_fire_px < threshold_num_pixels_f and counter == 0:
                    logging.info(f'just initiate the dataset')
                    X_y_country = np.empty( (0, X_all_presence.shape[1]) )
                    Y_country = np.array([],)
                

                # add annual not burned points if year falls under thresold
                
                if num_fire_px <= threshold_num_pixels_nof:

                    logging.info(f' found year/month with lowest fire occurence (num px fires this period, num px fires threasold): {num_fire_px}, {threshold_num_pixels_nof}')
                    X_all_absence = X_all[~fires_annual_rows]
                
                    # sampling and update presences 
                    if country_name in list(self.config.nordic_countries.keys()):
                        size = self.config.nordic_countries[country_name]
                        X_absence_indexes = np.random.choice(X_all_absence.shape[0], size=size, replace=False)
                    else:
                        # all the fires burned point divided by 4 (i pick 4 years7months with few fire for sampling absences), with reduction
                        fire_poins_over_treashold = [i for i in fire_points if i > threshold_num_pixels_f]
                        _sample = int((sum(fire_poins_over_treashold)/len(fire_poins_over_treashold)) * self.config.reduce_fire_points/100)
                        X_absence_indexes = np.random.choice(X_all_absence.shape[0], size=_sample, replace=False)

                    logging.debug(f'absence annual country {len(X_absence_indexes)}')
                    logging.debug(f'presence annual country {len(X_all_presence)}')
                                
                    X_all_absence = X_all_absence[X_absence_indexes, :]  
                
                    # update X and Y with absences 
                    X_y_country = np.concatenate([X_y_country, X_all_absence], axis = 0)
                    Y_country = np.concatenate([Y_country, np.zeros(X_all_absence.shape[0])], axis = 0)
                    
                    logging.debug(f' len X country updated (with absences): {X_y_country.shape}')
                    logging.debug(f' len Y country updated (with absences): {Y_country.shape}')

                
                counter += 1
        
        return X_y_country, Y_country, columns_annual




    def calculate_thresholds_for_sampling(self, fires_country_path, dem_path, batch_number):
        # open the dataset of fires
        fires = gpd.read_file(fires_country_path)
        with rio.open(dem_path) as dem:   
            fires = fires.to_crs(dem.crs)
        
        # convert the date column to datetime
        fires[self.config.name_col_y_fires] = pd.to_datetime(fires[self.config.name_col_y_fires]) 
        
        # create a lambda function to iterate over years or months
        years_or_months = lambda x: self.config.wildfire_years if x == False else [f'{str(year)}_{month}' for year in self.config.wildfire_years for month in range(1,13)]

        # find out in all the years/months, inside my batch, the 4 lowest number of years/month to sample no burned points
        fire_points = list()
        for year in years_or_months(self.config.convert_to_month):
            if self.config.convert_to_month == False:
                fire_annnual_shp = fires[ fires[self.config.name_col_y_fires].dt.year == year]
            else:
                _year = int(year.split('_')[0])
                _month = int(year.split('_')[1])
                
                fire_monthly_shp = fires[ fires[self.config.name_col_y_fires].dt.month == _month]
                # select the year I am working on 
                fire_annnual_shp = fire_monthly_shp[ fire_monthly_shp[self.config.name_col_y_fires].dt.year == _year]

            try:                
                fires_arr =  rasterize_numerical_feature(fire_annnual_shp, dem_path)
                fires_arr, _, _ =  batch_by_rows(fires_arr, self.config.batches, batch_number)
            except:
                fire_points.append(0)

            fire_points.append(len(fires_arr[fires_arr == 1]))
        
        # sort fire points list
        # fire_points.sort()
        # as thresold fire points for defining xxx years/months to sample no burned points I take the minimum required
        # threshold_num_pixels = fire_points[3]
        # use percentile to select the points in low and high fire activity periods (in low I take absences, in high i take presences) 
        sample_abs, sample_pres = self.config.percentiles_absence_presence
        threshold_num_pixels_nof = np.percentile(fire_points, sample_abs)
        threshold_num_pixels_f = np.percentile(fire_points, sample_pres)

        logging.info(f' max of fires in list of years/months: {max(fire_points)}')
        logging.info(f' threasolds: {threshold_num_pixels_nof, threshold_num_pixels_f}')

        return fire_points, threshold_num_pixels_nof, threshold_num_pixels_f






