import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
# xg boost
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
import rasterio as rio
import geopandas as gpd
import logging
from dataclasses import dataclass, field
from dotmap import DotMap
import json

# files from the same repository
# from input_config import InputConfiguration

from tools.tools_dataset_creation import DatasetTools
from tools.tools_random_forest import RFForestAlgorithm
from tools.plotting_results import Plots
from tools.useful_decorators import timer, ram_consumption
from tools.utils import set_logging, send_email, remove_features, rasterize_numerical_feature, batch_by_rows, get_lat_lon_arrays_from_raster, save_raster_as

from input_config import DEFAULT


@dataclass
class SupranationalModel():

    '''
    This class is used to prapere X and Y dataset composed by multiple country sources and then to build the model.
    the datasets can have annual structure or static structure.
    multiple countries can be passed and their datasets will be sampled in order to create an unified dataset
    '''

    working_dir: str
    config: dict = field(default_factory = lambda: DEFAULT, init = True)
    
    def __post_init__(self):

        self.config = DotMap(self.config)
        self.algorithm = RFForestAlgorithm()
        self.plots = Plots()

        set_logging(self.working_dir, level = logging.INFO)


    @timer
    @ram_consumption
    def creation_dataset_annual(self, annual_features_paths: dict[dict[dict]], 
                                mandatory_input_dict: dict[list], optional_input_dict: dict[dict[list]]):
                              
        '''
        this method takes dictionaries of mandatory, optional and annual layers
        and build the X and Y datasets.
        annual_features_paths is structured as following: 
        {country1: {year1: {feature1: path1, feature2: path2}, year2: {feature1: path1, feature2: path2}}, etc}
        mandatory_input_dict and optional_input_dict are structured as following:
        {country1: [dem_path, veg_path, fires_path], etc  }
        {country1: {var1_name: [var1_path, is_categorical]}, etc }
        '''

        logging.info('starting looping the countries')   
        countries = list(mandatory_input_dict.keys())

        for idx, country in enumerate(countries):
            dem_path, veg_path, fires_country_path = mandatory_input_dict[country]
            annual_lists_indices = annual_features_paths[country]

            if optional_input_dict is not None:     
                other_layer_dict_paths = optional_input_dict[country]     
            else:
                other_layer_dict_paths = {}
            
            # add lat and lon to the dataset
            lat_arr, lon_arr = get_lat_lon_arrays_from_raster(dem_path)
            lat_path = save_raster_as(lat_arr, os.path.join(self.working_dir, 'lat.tif'), dem_path)
            lon_path = save_raster_as(lon_arr, os.path.join(self.working_dir, 'lon.tif'), dem_path)
            other_layer_dict_paths.update({'lat': [lat_path, False], 'lon': [lon_path, False]})


            print(f'\nCOUNTRY: {country}\n') 
            logging.info(f'\nCOUNTRY: {country}\n')   
            
            batch_number = 0
            while batch_number < self.config.batches:
            
                # initiate a dataset gathering feature names and data
                dem_arr, mask, data_dict, nb_codes_list = DatasetTools(self.config).initiate_dataset(dem_path, veg_path,
                                                                                                        other_layer_dict_paths,
                                                                                                        self.working_dir, batch_number)
                
                list_features_to_remove = [i for i in self.config.list_features_to_remove if i != 'lat' and i != 'lon']
                data_dict = remove_features(data_dict, list_features_to_remove)

                # create the country dataset with just static variables (excluding then the annual indices)
                X_country, _ , mask, columns = self.algorithm.create_XY_from_dict(dem_arr, mask, data_dict, None)        
                logging.info(f'country dataset static features:\n{columns}')

                
                # start the association yeaerly fires with yearly climate indices  
                fire_points, threshold_num_pixels_nof, threshold_num_pixels_f = DatasetTools(self.config).calculate_thresholds_for_sampling(fires_country_path, dem_path,
                                                                                                                                            batch_number = batch_number)

                X_y_country, Y_country, columns_annual = DatasetTools(self.config).create_XY_annual(fires_country_path, 
                                                                                            annual_lists_indices,
                                                                                            dem_path,
                                                                                            dem_arr,
                                                                                            mask, 
                                                                                            X_country,
                                                                                            country,
                                                                                            fire_points, threshold_num_pixels_f, threshold_num_pixels_nof,
                                                                                            batch_number = batch_number,
                                                                                            ) 
                
                # transform dataset in dataframe, this is for adding veg columns not available for certain countries (but available for others)
                all_cols_y = list(columns) + list(columns_annual)
                X_y_country = pd.DataFrame(X_y_country, columns = all_cols_y)

                # save X and Y country 
                X_y_country.to_csv(os.path.join(self.working_dir, f"X_{country}_batchnum{batch_number}.csv"))
                Y_csv = pd.DataFrame(Y_country, columns = ['label'])
                Y_csv.to_csv(os.path.join(self.working_dir, f"Y_{country}__batchnum{batch_number}.csv"))

                            
                # initiate EU Y which already include 0 1 labels
                if idx == 0 and batch_number == 0:
                    Y_eu = Y_country 
                    X_eu = X_y_country
                else:
                    # update datasets 
                    Y_eu = np.concatenate([Y_eu, Y_country], axis = 0)
                    X_eu = pd.concat([X_eu, X_y_country], axis = 0)
                    X_eu = X_eu.fillna(0)

                for i,j in zip(range(0, len(X_eu.iloc[0])), X_eu.columns):
                    logging.debug(f'MAX of {j} --> {np.max(X_eu[j])}')
                
                logging.debug(f' updated len Y eu: {Y_eu.shape}')
                logging.debug(f' updated len X eu: {X_eu.shape}')
                batch_number += 1
            
            
        # removing all zero features
        for col in X_eu.columns:
            if len(np.unique(X_eu[col])) == 1:
                X_eu.drop(col, axis = 1, inplace = True)

        
        # save all the features
        all_cols = X_eu.columns
        logging.info(f'final features of your model: {all_cols}')
        
        # print max val of each feature
        for i,j in zip(range(0, len(X_eu.iloc[0])), all_cols):
            logging.info(f'min, mean, max of {j} --> {np.nanmin(X_eu[j])}, {np.nanmean(X_eu[j])}, {np.nanmax(X_eu[j])}')
            
        # saving the datasets    
        X_eu_path = os.path.join(self.working_dir, "X.csv")
        X_eu.to_csv(X_eu_path)
        Y_eu_csv = pd.DataFrame(Y_eu, columns = ['label'])
        Y_eu_path = os.path.join(self.working_dir, "Y.csv")
        Y_eu_csv.to_csv(Y_eu_path)
        # save also without coords
        X_eu_no_coords = X_eu.drop(['lat', 'lon'], axis = 1)
        X_eu_no_coords.to_csv(os.path.join(self.working_dir, "X_no_coords.csv"))
    
        return X_eu_path, Y_eu_path


    @timer
    @ram_consumption
    def creation_dataset_static(self, mandatory_input_dict: dict, optional_input_dict: dict):
                            
        
        '''
        this function will loop between different countries and create the X and Y datasets whitout including annual features.
        first input dict contains mandatory layers: it is a dictionary is structured as following : {country1: [dem_path, veg_path, fires_path], etc  }
        second input dict contains optional layers: it is a dictionary is structured as following : {country1: {var1_name: [var1_path, is_categorical]}, etc  }
         '''

        logging.info('starting looping the countries')   
        countries = list(mandatory_input_dict.keys())

        for idx, country in enumerate(countries):
            dem_path, veg_path, fires_country_path = mandatory_input_dict[country]
            # list of optional layers
            if optional_input_dict is not None:     
                other_layer_dict_paths = optional_input_dict[country]     
            else:
                other_layer_dict_paths = {}
            
            # add lat and lon to the dataset
            lat_arr, lon_arr = get_lat_lon_arrays_from_raster(dem_path)
            lat_path = save_raster_as(lat_arr, os.path.join(self.working_dir, 'lat.tif'), dem_path)
            lon_path = save_raster_as(lon_arr, os.path.join(self.working_dir, 'lon.tif'), dem_path)
            other_layer_dict_paths.update({'lat': [lat_path, False], 'lon': [lon_path, False]})
            

            print(f'\nCOUNTRY: {country}\n') 
            logging.info(f'\nCOUNTRY: {country}\n')   
            
            batch_number = 0
            while batch_number < self.config.batches:
                # initiate a dataset gathering feature names and data
                dem_arr, mask, data_dict, nb_codes_list = DatasetTools(self.config).initiate_dataset(dem_path, veg_path,
                                                                                            other_layer_dict_paths,
                                                                                            self.working_dir, batch_number)
                
                # not remove lat lon for now
                list_features_to_remove = [i for i in self.config.list_features_to_remove if i != 'lat' and i != 'lon']
                data_dict = remove_features(data_dict, list_features_to_remove)

                # open fires and rasterize it 
                fires = gpd.read_file(fires_country_path)
                with rio.open(dem_path) as dem:   
                    fires = fires.to_crs(dem.crs)
                            
                fires_arr =  rasterize_numerical_feature(fires, dem_path)
                fires_arr, _, _ =  batch_by_rows(fires_arr, self.config.batches, batch_number)
                
                # create the country dataset with just static variables 
                X_country, Y_country , mask, columns = self.algorithm.create_XY_from_dict(dem_arr, mask, data_dict, fires_arr)  

                # increase the number of zeros in the dataset of specifi countries
                if country in list(self.config.nordic_countries.keys()):
                    print('It is nordic country name: ', country)
                    unbalance_zeros = self.config.nordic_countries[country]
                else:
                    unbalance_zeros = None

                # sample points of X ad Y sets
                X_country, Y_country = self.algorithm.reduce_X_Y_sets(X_country, Y_country, self.config.reduce_fire_points,
                                                                    unbalance_zeros = unbalance_zeros)    
                logging.info(f'dataset features:\n{columns}')

                # convert to dataframe
                X_country = pd.DataFrame(X_country, columns = columns)

                # save X and Y country 
                X_country.to_csv(os.path.join(self.working_dir, f"X_{country}_{batch_number}.csv"))
                Y_csv = pd.DataFrame(Y_country, columns = ['label'])
                Y_csv.to_csv(os.path.join(self.working_dir, f"Y_{country}_batchnum{batch_number}.csv"))

                # initiate EU Y which already include 0 1 labels
                if idx == 0 and batch_number == 0:
                    Y_eu = Y_country 
                    X_eu = X_country
                else:
                    # update datasets 
                    Y_eu = np.concatenate([Y_eu, Y_country], axis = 0)
                    # X_eu = np.concatenate([X_eu, X_y_country], axis = 0)
                    X_eu = pd.concat([X_eu, X_country], axis = 0)
                    X_eu = X_eu.fillna(0)

                # print max of variables
                for i,j in zip(range(0, len(X_eu.iloc[0])), X_eu.columns):
                    logging.debug(f'min max of {j} --> {np.min(X_eu[j])} , {np.max(X_eu[j])}')
                
                logging.info(f'updated len Y eu: {Y_eu.shape}')
                logging.info(f'updated len X eu: {X_eu.shape}')
                batch_number += 1
                    
            
        # removing all zero features
        for col in X_eu.columns:
            if len(np.unique(X_eu[col])) == 1:
                X_eu.drop(col, axis = 1, inplace = True)
            else:
                pass
        
        # features
        all_cols = X_eu.columns
        logging.info(f'final features of your model: {all_cols}')
        
        # print max val of each feature
        for i,j in zip(range(0, len(X_eu.iloc[0])), X_eu.columns):
            logging.info(f'min max of {j} --> {np.min(X_eu[j])} , {np.max(X_eu[j])}')
            
        # saving the datasets    
        X_eu_path = os.path.join(self.working_dir, "X.csv")
        X_eu.to_csv(X_eu_path)
        Y_eu_csv = pd.DataFrame(Y_eu, columns = ['label'])
        Y_eu_path = os.path.join(self.working_dir, "Y.csv")
        Y_eu_csv.to_csv(Y_eu_path)
    
        return X_eu_path, Y_eu_path



    @timer
    def creation_model(self, X_eu_path: str, Y_eu_path: str, model_path: str):
        
        '''
        this method takes X and Y paths and build the model
        '''

        # open X and Y sets
        X_eu = pd.read_csv(X_eu_path, index_col=0)
        all_cols = X_eu.columns
        Y_eu_csv = pd.read_csv(Y_eu_path, index_col=0)
        Y_eu = Y_eu_csv.values
            
        X_eu = X_eu.values
        
        # now I have the EU dataset: let's build the model
        logging.debug('uniques Y europe:\n', np.unique(Y_eu, return_counts = True))
        
        X_train, X_test, y_train, y_test = train_test_split(X_eu, Y_eu, test_size=0.25, random_state=42)

        # saving arryas
        arrs = [X_train, X_test, y_train, y_test]
        arr_names = ['X_train_coords.npy', 'X_test_coords.npy', 'y_train.npy', 'y_test.npy']
        for arr, arr_name in zip(arrs, arr_names):
            np.save(os.path.join(self.working_dir, arr_name) , arr) # to load: np.load(file.npy)

        # drop lat and lon columns if they are present in list to remove 
        X_train = pd.DataFrame(X_train, columns = all_cols)
        X_test = pd.DataFrame(X_test, columns = all_cols)

        if 'lat' in self.config.list_features_to_remove:
            X_train.drop('lat', axis = 1, inplace = True)
            X_test.drop('lat', axis = 1, inplace = True)
            all_cols = list(all_cols)
            all_cols.remove('lat')
        if 'lon' in self.config.list_features_to_remove:
            X_train.drop('lon', axis = 1, inplace = True)
            X_test.drop('lon', axis = 1, inplace = True)
            all_cols = list(all_cols)
            all_cols.remove('lon')

        X_train = X_train.values
        X_test = X_test.values

        # save again daaset excluding lat and lon
        arrs = [X_train, X_test]
        arr_names = ['X_train.npy', 'X_test.npy']
        for arr, arr_name in zip(arrs, arr_names):
            np.save(os.path.join(self.working_dir, arr_name) , arr) # to load: np.load(file.npy)
            
        
        try:
            y_train = y_train.ravel()   
        except:
            logging.debug('couldnt ravel the Y array')

        logging.info(f'Running RF on data sample: {X_train.shape}')
       
        if self.config.gridsearch == 1:
            
            base_model = RandomForestClassifier()
            random_grid = [{'n_estimators': self.config.ntree,
                            'max_depth': self.config.max_depth,
                            'min_samples_split': [2, 5, 10],
                            'min_samples_leaf': [1, 2, 4],
                }]
            rf_random = GridSearchCV(estimator = base_model, param_grid = random_grid,  
                                        cv = 3, scoring="accuracy",return_train_score=True,
                                        n_jobs = -1, verbose = 1)   
            rf_random.fit(X_train, y_train) 
            model = rf_random.best_estimator_
            logging.info(f' best params founded:\n {rf_random.best_params_}')

        else:
            model = RandomForestClassifier(n_estimators = self.config.ntree, max_depth = self.config.max_depth, verbose = 1, n_jobs = -1)
            # logging.info('using XG boost')
            # model = GradientBoostingClassifier(n_estimators = self.config.ntree, max_depth = self.config.max_depth, verbose = 1)

        # performances of the model    
        self.algorithm.print_stats(model, X_train, y_train, X_test, y_test, None, self.working_dir)    
        out_path_importances = os.path.join(self.working_dir, 'importances.csv')
        df, df_perc, df_veg = self.algorithm.RF_feature_importances(model, all_cols, out_path_importances)
        
        # SAVE MODEl
        pickle.dump(model, open(model_path, 'wb'))
        
            
        # plotting results
        out_plot_importances = os.path.join(self.working_dir, 'importances.png')
        self.plots.plot_main_importances(df, out_plot_importances)
        out_imp_perc = os.path.join(self.working_dir, 'imp_perc.png' )
        self.plots.plot_main_importances(df_perc, out_imp_perc)
        out_imp_veg = os.path.join(self.working_dir, 'imp_veg.png' )
        self.plots.plot_main_importances(df_veg, out_imp_veg)

        send_email(mail_user = self.config.user_email, 
                            mail_password = self.config.email_pwd, 
                            message = 'ML model has been saved!!!')
        
        return X_train, X_test, y_train, y_test
                    





