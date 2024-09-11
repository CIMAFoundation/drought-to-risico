# -*- coding: utf-8 -*-
"""
Created on Mon May  1 06:36:08 2023

@author: Giorg
"""



import rasterio as rio
import numpy as np
import pickle
import os
import pandas as pd
import logging
from dotmap import DotMap
from dataclasses import dataclass, field
import json
from scipy import signal

# file from this repository
os.chdir('/share/home/gruppo4/Bolivia/DATA/BolMLmonthly/test_model/annual_wildfire_susceptibility')
from tools.tools_random_forest import RFForestAlgorithm
from tools.plotting_results import Plots
from tools.tools_dataset_creation import DatasetTools
from tools.useful_decorators import timer, ram_consumption
from tools.utils import remove_features, batch_by_rows, reproject_layer, save_raster_as, create_folder, set_logging, raster_classes, get_lat_lon_arrays_from_raster

from input_config import DEFAULT

@dataclass
class Susceptibility():

    '''
    Class for:
    1) running a ML model at any scale (static - no temporal resolution) to get a single outcome
    2) run a ML model at any scale (dynamic - annula/monthly temporal resolution) to get multiple outcomes
    3) train and run a model on a single dataset,national or sub national level (static - no temporal resolution) to get a single outcome
    '''

    dem_path: str
    veg_path: str
    working_dir: str
    optional_input_dict: dict
    config: dict = field(default_factory = lambda: DEFAULT, init=True)

    
    def __post_init__(self):
        
        self.config = DotMap(self.config)
        
        self.plots = Plots()
        self.algorithm = RFForestAlgorithm()

        set_logging(self.working_dir, level = logging.INFO)

        
    @timer   
    @ram_consumption 
    def run_existed_model_annual(self, model_path: str, annual_features_paths: dict[dict], training_df_path: str, start_year: int = 0):
        
        '''
        annual_features_paths: {year1: {feature1: path1, feature2: path2}, year2: {feature1: path1, feature2: path2}}, etc}
        '''


        outdir = create_folder(self.working_dir)
        # load a model to create susc maps
        model = pickle.load(open(model_path, "rb"))


        # update the dataset each year with annual clim indices
        # counter = 0 # start_year
        for map_number, year in enumerate(annual_features_paths.keys()):

            
            annual_feature_list = annual_features_paths[year]

            # initiate susc array to ahve to full size in case of batching procedure
            with rio.open(self.dem_path) as dem:
                susc_map_complete_annual = np.zeros(dem.read(1).shape).astype(np.float32())

            batch_number = 0
            while batch_number < self.config.batches:

                # initiate a dataset with the basic featueres 
                dem_arr, mask, data_dict, nb_codes_list = DatasetTools(self.config).initiate_dataset(self.dem_path, self.veg_path,
                                                                                        self.optional_input_dict,
                                                                                        outdir, batch_number)


                
                # create a dict with annual indices with the possibility to reproject them
                temporal_aggr = lambda x: '_y' if x == False else '_m'
                annual_layers_dict = {}              
                for annual_layer_name, annual_layer in annual_feature_list.items():
                    annual_layer_name = annual_layer_name + temporal_aggr(self.config.temporal_aggregation)
                    with rio.open(annual_layer) as a:
                        annual_arr = a.read(1)
                        annual_arr, _, _ =  batch_by_rows(annual_arr, self.config.batches, batch_number)
                        annual_arr =  reproject_layer(dem_arr, annual_arr, self.dem_path, annual_layer, outdir)
                        
                        if self.config.drop_neg_and_na_annual == True:
                            annual_arr = np.where(annual_arr >= 0, annual_arr, np.NaN)
                            annual_arr = np.where(np.isnan(annual_arr) == True, np.nanmean(annual_arr), annual_arr)

                        if os.path.basename(annual_layer).startswith('veg'):
                            logging.info('found an annual layer starting with veg word, I consider it as dynamic vegetation and use one hot encoding')
                            annual_layers_dict = self.algorithm.one_hot_encoding(annual_arr, annual_layer_name, annual_layers_dict) 
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
                            logging.info(f'found annnual layer with max val {np.nanmax(annual_arr)}')
                            annual_layers_dict[annual_layer_name] = annual_arr
                            
        
                # update the dataset dict with the annual layers 
                data_dict.update(annual_layers_dict)
                
                data_dict =  remove_features(data_dict, self.config.list_features_to_remove)
                
                # use the dict to create a dataset of X 
                X_all, _ , mask, columns = self.algorithm.create_XY_from_dict(dem_arr, mask, data_dict, None)
                  
                # make current dataset structure as the training one
                X_all = pd.DataFrame(X_all, columns = columns)

                training_df = pd.read_csv(training_df_path) 
                X_all = self.algorithm.adjust_dataset(X_all, training_df)

                # save dataset of this year if wanted
                if self.config.save_dataset == True:
                    outdir_dataset =  create_folder(outdir, name = 'dataset')
                    out_dataset = os.path.join(outdir_dataset, f'dataset{batch_number}_{str(year)}.npy')
                    np.save(out_dataset, X_all)
                    np.save(os.path.join(outdir_dataset, f'mask.npy'), mask)

                # create a ssuc map with the static + annual features
                print('running model...')
                annual_susc_map = self.algorithm.get_results(model, X_all, dem_arr, mask)
                print(f'finittto (batch {batch_number})')

                # reconstructing annual map
                # make sure dimention is the same as susc_map_complete
                if len(annual_susc_map.shape) == 3:
                    annual_susc_map = annual_susc_map[0, :, :]

                # build the susc map bacth by batch
                _, batch_row_start, batch_row_end =  batch_by_rows(susc_map_complete_annual, self.config.batches, batch_number)
                susc_map_complete_annual[batch_row_start : batch_row_end, :] = annual_susc_map

                batch_number += 1

            
            # create new folder for saving annual maps             
            outdir_annual =  create_folder(outdir, name = 'annual_maps')

            # paths of annual maps
            out_annual = os.path.join(outdir_annual, f'Annual_susc_{str(year)}.tif')

            # save the map    
            save_raster_as(susc_map_complete_annual,
                                    out_annual,
                                    self.dem_path,
                                    dtype = np.float32(),
                                    nodata = -1)    
            
            # update a new array containing the sum of all the annual maps in order to evalaute the average
            if year == start_year:
                susc_map_sum = susc_map_complete_annual
            else:
                susc_map_sum += susc_map_complete_annual

            # evaluate AUC and MSE for the second test set ("validation?" set) making sure to consider only the fires (and not) of the current year
            if self.config.make_plots == True:
                fires_shp_custom_test = self.config.validation_ba_dict['fires10']
                fires_all = self.config.validation_ba_dict['fires90']

                self.algorithm.print_stasts_on_custom_dataset(fires_shp_custom_test, fires_all, susc_map_complete_annual, outdir_annual, self.dem_path, year = year,
                                                              config = self.config)

                # plot histogram of distribution of susc in custom test data of burned areas (this works just yor years)
                annual_susc_5_classes =  raster_classes(susc_map_complete_annual, [0.3, 0.5, 0.8, 0.95], -1)
                self.plots.plot_BA_over_susc(annual_susc_5_classes, self.config.validation_ba_dict, 
                                            outdir_annual, self.dem_path, 
                                            year_of_susc = str(year), 
                                            country_name = self.config.country_name,
                                            pixel_size = self.config.pixel_size, 
                                            single_year = year,
                                            colname = self.config.name_col_y_fires,
                                            )   
                                
            # counter += 1
                        
    
        # average of all the annual maps
        susc_map = susc_map_sum/(map_number + 1) 
                                
        # SAVE the map
        out = os.path.join(outdir, 'SUSCEPTIBILITY.tif')  
        save_raster_as(susc_map,
                             out,
                             self.dem_path,
                             dtype = np.float32(),
                             nodata = -1)
        
        # save maps classified with quantiles
        susc_5_classes =  raster_classes(susc_map, [0.3, 0.5, 0.8, 0.95], -1)
        out5 = os.path.join(outdir, 'SUSCEPTIBILITY_5classes.tif')  
        save_raster_as(susc_5_classes,
                             out5,
                             self.dem_path,
                             dtype = np.int8(),
                             nodata = 0)
        
        susc_3_classes =  raster_classes(susc_map, [0.25, 0.75], -1)
        out3 = os.path.join(outdir, 'SUSCEPTIBILITY_3classes.tif')  
        save_raster_as(susc_3_classes,
                             out3,
                             self.dem_path,
                             dtype = np.int8(),
                             nodata = 0)
        
        if self.config.make_plots == True:
            # evaluate AUC and MSE for the second test set ("validation?" set)
            fires_shp_custom_test = self.config.validation_ba_dict['fires10']
            fires_all = self.config.validation_ba_dict['fires90']
            self.algorithm.print_stasts_on_custom_dataset(fires_shp_custom_test, fires_all, susc_map, outdir, self.dem_path,
                                                          config = self.config)

            # plot histogram of distribution of susc in custom test data of burned areas
            year_of_susc = f'{start_year}_{year}'
            self.plots.plot_BA_over_susc(susc_5_classes, self.config.validation_ba_dict, outdir, self.dem_path,
                                        pixel_size = self.config.pixel_size,
                                        year_of_susc = year_of_susc, 
                                        country_name = self.config.country_name, 
                                        colname = self.config.name_col_y_fires) 
                                          
                                     

    @timer      
    @ram_consumption  
    def run_existed_model(self, model_path: str, training_df_path: str):

        outdir = create_folder(self.working_dir)
        
        # initiate susc array to ahve to full size in case of batching procedure
        with rio.open(self.dem_path) as dem:
            susc_map_complete = np.zeros(dem.read(1).shape).astype(np.float32())

        batch_number = 0
        while batch_number < self.config.batches:

            # initiate a dataset with the basic featueres
            dem_arr, mask, data_dict, nb_codes_list =  DatasetTools(self.config).initiate_dataset(self.dem_path, self.veg_path,
                                                                                        self.optional_input_dict,
                                                                                        outdir, batch_number)
            
            data_dict = remove_features(data_dict, self.config.list_features_to_remove)

            # create a dataset of X
            X_all, _ , mask, columns = self.algorithm.create_XY_from_dict(dem_arr, mask, data_dict, None)   
            # print(X_all)
            # here i am ordering the features as the training dataset of the model that will be used
            X_all = pd.DataFrame(X_all, columns = columns)
            training_df = pd.read_csv(training_df_path)
            X_all = self.algorithm.adjust_dataset(X_all, training_df)
            
            # load an already trained ML model to create susc maps
            model = pickle.load(open(model_path, "rb"))
            susc_map = self.algorithm.get_results(model, X_all, dem_arr, mask)
            
            # make sure dimention is the same as susc_map_complete
            if len(susc_map.shape) == 3:
                susc_map = susc_map[0, :, :]

            # build the susc map bacth by batch
            _, batch_row_start, batch_row_end =  batch_by_rows(susc_map_complete, self.config.batches, batch_number)
            susc_map_complete[batch_row_start : batch_row_end, :] = susc_map

            batch_number += 1
        

        # SAVE the map
        out = os.path.join(outdir, 'SUSCEPTIBILITY.tif')  
        save_raster_as(susc_map_complete,
                             out,
                             self.dem_path,
                             dtype = np.float32(),
                             nodata = -1)
    
        # save quantile maps
        susc_5_classes =  raster_classes(susc_map_complete, [0.3, 0.5, 0.8, 0.95], -1)
        out5 = os.path.join(outdir, 'SUSCEPTIBILITY_5classes.tif')  
        save_raster_as(susc_5_classes,
                             out5,
                             self.dem_path,
                             dtype = np.int8(),
                             nodata = -1)
        
        if self.config.make_plots == True:
            # evaluate AUC and MSE for the second test set ("validation?" set)
            fires_shp_custom_test = self.config.validation_ba_dict['fires10']
            fires_all = self.config.validation_ba_dict['fires90']
            self.algorithm.print_stasts_on_custom_dataset(fires_shp_custom_test, fires_all, susc_map_complete, outdir, self.dem_path,
                                                          config = self.config)
            
            # plot histogram of distribution of susc in custom test data of burned areas
            self.plots.plot_BA_over_susc(susc_5_classes, self.config.validation_ba_dict, outdir, self.dem_path,
                                    pixel_size = self.config.pixel_size,
                                    year_of_susc = 'static', 
                                    country_name = self.config.country_name, 
                                    colname = self.config.name_col_y_fires) 

        
        
    @timer    
    def create_and_run_model(self, fires_raster_path: str, model_path: str):
        
        outdir = create_folder(self.working_dir) 

        # add lat lon to dataset 
        lat_arr, lon_arr = get_lat_lon_arrays_from_raster(self.dem_path)
        lat_path = save_raster_as(lat_arr, os.path.join(self.working_dir, 'lat.tif'), self.dem_path)
        lon_path = save_raster_as(lon_arr, os.path.join(self.working_dir, 'lon.tif'), self.dem_path)
        self.optional_input_dict.update({'lat': [lat_path, False], 'lon': [lon_path, False]})


        # initiate a dataset with the basic featueres     
        dem_arr, mask, data_dict, nb_codes_list = DatasetTools(self.config).initiate_dataset(self.dem_path, self.veg_path,
                                                                                        self.optional_input_dict,
                                                                                        outdir)
            
        # open dataset of fires
        with rio.open(fires_raster_path) as fire:
            fires_arr = fire.read(1)
            
        list_features_to_remove = [i for i in self.config.list_features_to_remove if i != 'lat' and i != 'lon']
        data_dict =  remove_features(data_dict, list_features_to_remove)

        # create a set of X and Y
        X_all, Y_all, mask, columns = self.algorithm.create_XY_from_dict(dem_arr, mask, data_dict, fires_arr)   
        print(f'dataset features:\n{columns}')

        X_all = pd.DataFrame(X_all, columns = columns)
        # save X and Y country 
        X_all.to_csv(os.path.join(self.working_dir, f"X.csv"))
        Y_csv = pd.DataFrame(Y_all, columns = ['label'])
        Y_csv.to_csv(os.path.join(self.working_dir, f"Y.csv"))

        # drop lat and lon columns if they are present in list to remove
        if 'lat' in self.config.list_features_to_remove:
            X_all.drop('lat', axis = 1, inplace = True)
            columns = list(columns)
            columns.remove('lat')
        if 'lon' in self.config.list_features_to_remove:
            X_all.drop('lon', axis = 1, inplace = True)
            columns.remove('lon')

        # return all the X values without lat and lon
        X_all = X_all.values
    
        # create and train directly a model returning the model and the train and test sets
        model, X_train, X_test, y_train, y_test = self.algorithm.train(X_all, 
                                                                 Y_all, 
                                                                 self.config.reduce_fire_points, 
                                                                 self.config.ntree, 
                                                                 self.config.max_depth,
                                                                 )
        
        # cross validating
        if self.config.make_CV == True:
            
            # _scores = self.algorithm.cross_validation(model, X_all, Y_all,
            #                                             cv = 5, scoring = 'roc_auc')
            
            # make the prediction after each fold and average them to produce a susc map
            preds, auc, mse = self.algorithm.cross_validation_predictions(model, X_all, Y_all,
                                                                            cv = 5)
            
            _scores = auc
            # use the predictions from the cross validation for producing a susc map 
            susc_pred_cv = np.zeros_like(dem_arr).astype(np.float32())
            susc_pred_cv[mask] = preds
            
            # clip susc where dem exsits
            susc_pred_cv[~mask] = -1

            # save this map 

            out_cv = os.path.join(outdir, 'SUSCEPTIBILITY_CV.tif')
            save_raster_as(susc_pred_cv,
                                out_cv,
                                self.dem_path,
                                dtype = np.float32(),
                                nodata = -1)

            susc_cv_5_classes =  raster_classes(susc_pred_cv, [0.3, 0.5, 0.8, 0.95], -1)
            out_cv_5 = os.path.join(outdir, 'SUSCEPTIBILITY_CV_5classes.tif')  
            save_raster_as(susc_cv_5_classes,
                                out_cv_5,
                                self.dem_path,
                                dtype = np.int8(),
                                nodata = -1)



        else:
            logging.info('no cv selected')
            _scores = None
        
        # performance of the model already created
        self.algorithm.print_stats(model, X_train, y_train, X_test, y_test, _scores, outdir)    
        out_path_importances = os.path.join(outdir, 'importances.csv')
        df, df_perc, df_veg = self.algorithm.RF_feature_importances(model, columns, out_path_importances)
        
        # save the model
        pickle.dump(model, open(model_path, 'wb'))
        
        # saving arryas
        arrs = [X_train, X_test, y_train, y_test]
        arr_names = ['X_train.npy', 'X_test.npy', 'y_train.npy', 'y_test.npy']
        
        for arr, arr_name in zip(arrs, arr_names):
            np.save(os.path.join(outdir, arr_name), arr) # to load: np.load(file.npy)
            
        # plotting results
        out_plot_importances = os.path.join(outdir, 'importances.png')
        self.plots.plot_main_importances(df, out_plot_importances)
        out_imp_perc = os.path.join(outdir, 'imp_perc.png' )
        self.plots.plot_main_importances(df_perc, out_imp_perc)
        out_imp_veg = os.path.join(outdir, 'imp_veg.png' )
        self.plots.plot_main_importances(df_veg, out_imp_veg)

        # produce susc map
        susc_map = self.algorithm.get_results(model, X_all, dem_arr, mask)
    
        # SAVE the map
        out = os.path.join(outdir, 'SUSCEPTIBILITY.tif')  
        save_raster_as(susc_map,
                             out,
                             self.dem_path,
                             dtype = np.float32(),
                             nodata = -1)
        
        # save the map classified with quantiles
        susc_5_classes =  raster_classes(susc_map, [0.3, 0.5, 0.8, 0.95], -1)
        out5 = os.path.join(outdir, 'SUSCEPTIBILITY_5classes.tif')  
        save_raster_as(susc_5_classes,
                             out5,
                             self.dem_path,
                             dtype = np.int8(),
                             nodata = 0)
        
        if self.config.make_plots == True:
            # evaluate AUC and MSE for the second test set ("validation?" set)
            fires_shp_custom_test = self.config.validation_ba_dict['fires10']
            fires_all = self.config.validation_ba_dict['fires90']
            self.algorithm.print_stasts_on_custom_dataset(fires_shp_custom_test, fires_all, susc_map, outdir, self.dem_path,
                                                        config = self.config)

            # plot histogram of distribution of susc in custom test data of burned areas
            self.plots.plot_BA_over_susc(susc_5_classes, self.config.validation_ba_dict, outdir, self.dem_path,
                                    pixel_size = self.config.pixel_size,
                                    year_of_susc = 'static', 
                                    country_name = self.config.country_name, 
                                    colname = self.config.name_col_y_fires) 


                

#%%






