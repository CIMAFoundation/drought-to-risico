# -*- coding: utf-8 -*-InputConfiguration
"""
Created on Mon May  1 06:41:13 2023

@author: Giorg
"""

 
from scipy import signal
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import roc_auc_score, mean_squared_error, accuracy_score
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import roc_curve, accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import rasterio as rio
import pandas as pd
import geopandas as gpd
import os
from dataclasses import dataclass
import logging
from dotmap import DotMap


# files from the same repository
os.chdir('/share/home/gruppo4/Bolivia/DATA/BolMLmonthly/test_model/annual_wildfire_susceptibility/tools')
from plotting_results import Plots
from utils import rasterize_numerical_feature, aggregate_vars_importances



@dataclass
class RFForestAlgorithm():
    
    '''
    This class contains basic functions for training and running a ML model.
    It prints and saves performance stats and get the results of prediction in tiff format.
    '''   
        
    def create_dict_vars(self, dem_arr, dem_nodata, veg_arr, 
                      nb_codes_list, slope_arr, northing_arr, easting_arr, 
                      other_layers_dict):
        
        '''
        takes mandatory layers for Ml and other layers and return a dictionary with {feature name : array}
        '''
        # mask the vegetation
        veg_arr = veg_arr.astype(int)
        veg_arr_str = veg_arr.astype(str)
        del veg_arr       
        for i in nb_codes_list:
            veg_arr_str = np.where(veg_arr_str == i, '0', veg_arr_str)
        veg_mask = np.where(veg_arr_str == '0', 0, 1)
                
        # now a complete the mask adding dem existence
        mask = (veg_mask == 1) & (dem_arr != dem_nodata)
        
        # evaluation of perc just in vegetated area, non vegetated are grouped in code 0
        veg_int = veg_arr_str.astype(int)
        del veg_arr_str
        veg_int = np.where(veg_mask == 1, veg_int, 0)
        window_size = 2
        types = np.unique(veg_int)
        types_presence = {}
        
        counter = np.ones((window_size*2+1, window_size*2+1))
        take_center = 1
        counter[window_size, window_size] = take_center 
        counter = counter / np.sum(counter)

        veg_onehot_dict = {}
        # perc --> neightboring vegetation generation 
        # add also veg one hot encoding               
        for t in types:
            density_entry = 'perc_' + str(int(t)) 
            types_presence[density_entry] = 100 * signal.convolve2d(veg_int==t, counter, boundary='fill', mode='same')
            veg_code_entry = 'veg_' + str(int(t))
            veg_onehot_dict[veg_code_entry] = np.where(veg_int == t, 1, 0)
        
        
        # dict of layers for ML dataset generation            
        data_dict = {
            'dem': dem_arr,
            'slope': slope_arr,
            'north': northing_arr,
            'east': easting_arr,
        }

        for layer_name, layer_arr in other_layers_dict.items():
            data_dict[layer_name] = layer_arr
        
        # add perc to dictionary
        data_dict.update(types_presence)
        
        # add veg binary to dict
        data_dict.update(veg_onehot_dict)
        
        return mask, data_dict
    
    def one_hot_encoding(self, layer_arr, name_var, layer_dict):
                
        layer_arr = layer_arr.astype(int)
        types = np.unique(layer_arr)

        for t in types:
            veg_code_entry = name_var + '_' + str(int(t))
            layer_dict[veg_code_entry] = np.where(layer_arr == t, 1, 0)
        
        return layer_dict

    
    
    def create_XY_from_dict(self, dem_arr, mask, data_dict, fires_arr):

        # creaate X and Y datasets
        n_pixels = len(dem_arr[mask])
        n_features = len((data_dict.keys()))
        X_all = np.zeros((n_pixels, n_features))
        if fires_arr is not None:
            Y_all = fires_arr[mask]
        else:
            Y_all = None

        logging.info('Creating dataset for the ML model')
        columns = data_dict.keys()
        for col, k in enumerate(data_dict):
            data = data_dict[k]
            X_all[:, col] = data[mask]

        return X_all, Y_all, mask, columns 
    
    def reduce_X_Y_sets(self, X_all, Y_all, percentage, unbalance_zeros = None):

        # filter df taking info in the burned points
        fires_rows = Y_all != 0
        X_presence = X_all[fires_rows]
        
        # reduction of burned points --> reduction of training points       
        reduction = int((X_presence.shape[0]*percentage)/100)
        logging.info(f"reduced df points: {reduction} of {X_presence.shape[0]}")
        
        # sampling and update presences 
        X_presence_indexes = np.random.choice(X_presence.shape[0], size=reduction, replace=False)
        X_presence = X_presence[X_presence_indexes, :]         
        # select not burned points
        X_absence = X_all[~fires_rows]
        if unbalance_zeros != None:
            X_absence_choices_indexes = np.random.choice(X_absence.shape[0], size = unbalance_zeros, replace=False)
        else:
            X_absence_choices_indexes = np.random.choice(X_absence.shape[0], size=X_presence.shape[0], replace=False)
        X_pseudo_absence = X_absence[X_absence_choices_indexes, :]
        # create X and Y with same number of burned and not burned points
        X = np.concatenate([X_presence, X_pseudo_absence], axis=0)
        Y = np.concatenate([np.ones((X_presence.shape[0],)), np.zeros((X_pseudo_absence.shape[0],))])

        return X, Y


    def train(self, X_all: np.array, Y_all: np.array, percentage: int, ntrees: int, max_depth: int): 
        

        X, Y = RFForestAlgorithm().reduce_X_Y_sets(X_all, Y_all, percentage)

        # create training and testing df with random sampling
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
        
        logging.info(f'Running RF on data sample: {X_train.shape}')
        model  = RandomForestClassifier(n_estimators = ntrees, max_depth = max_depth, verbose = 2, n_jobs=-1)
        
        return model, X_train, X_test, y_train, y_test


    def cross_validation(self, model, X: np.array, Y: np.array, cv = 5, scoring = 'roc_auc'):
        
        print('starting the cross validation')
        auc_scores_cv = cross_val_score(model, X, Y, cv = cv, scoring = scoring)
        print('auc score on folds:\n')
        for i in auc_scores_cv:
            print(f'{i:.2f}')
            
        return auc_scores_cv
            
    def cross_validation_predictions(self, model, X: np.array, Y: np.array, cv = 5):

        preds = cross_val_predict(model, X, y = Y, cv = cv,  method = 'predict_proba')
        logging.debug(preds)
        preds = preds[:,1]
        roc_auc = roc_auc_score(Y, preds)
        mse = mean_squared_error(Y, preds)

        return preds, roc_auc, mse

        

    def print_stats(self, model, X_train, y_train, X_test, y_test, auc_scores_cv: list[str], working_dir: str):
        
        '''
        fit the ML model and return scores saved in a txt file
        '''

        # fit model 
        logging.info('fitting...')
        model.fit(X_train, y_train)
        # stats on training df
        p_train = model.predict_proba(X_train)[:,1]
        auc_train = roc_auc_score(y_train, p_train)
        logging.info(f'AUC score on train: {auc_train:.2f}')
        
        # stats on test df
        p_test = model.predict_proba(X_test)[:,1]
        auc_test = roc_auc_score(y_test, p_test)
        logging.info(f'AUC score on test: {auc_test:.2f}')
        mse = mean_squared_error(y_test, p_test)
        logging.info(f'MSE: {mse:.2f}')
        
        p_test_binary = model.predict(X_test)
        accuracy = accuracy_score(y_test, p_test_binary)
        logging.info(f'accuracy: {accuracy:.2f}')
        
        # no skill auc for plotting roc curve
        ns_probs = [0 for _ in range(len(y_test))]
        # calculate roc curves
        ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
        fpr, tpr, _ = roc_curve(y_test, p_test)
        
        # import the plot from the other class
        plots = Plots()
        
        # plot roc curve and save it
        roc_outfile = os.path.join(working_dir, 'ROC_curve.png')
        plots.plot_roc_curve(ns_fpr, ns_tpr, fpr, tpr, roc_outfile)

        cm = confusion_matrix(y_test, p_test_binary, normalize = 'true')
        report = classification_report(y_test, p_test_binary, target_names = ['no fire', 'fire']) 
        logging.info(f'REPORT\n{report}')
        
        # plot cm and save it
        cm_outfile = os.path.join(working_dir, 'confusion_matrix.png')
        plots.plot_confusion_matrix(cm, cm_outfile)
        
        # save scores in a txt file
        with open(os.path.join(working_dir, 'results.txt') , 'a') as f:
            
            metrics = [auc_test, auc_train, auc_scores_cv, mse, accuracy, cm, report]
            metrics_name = ['auc_test', 'auc_train', 'auc_folds', 'mse', 'accuracy', 'cm', 'report']
            
            for m, n in zip(metrics, metrics_name): 
                f.write('\n')
                f.write(f'\n{n}\n{m}\n')

    def print_stasts_on_custom_dataset(self, fires_shp_custom_test: str, fires_shp_all: str, susceptibility_arr: np.array, working_dir: str, dem_path: str, config: dict, year = None ):

        # config = InputConfiguration()
        config = DotMap(config)

        fires_shp_custom_test = gpd.read_file(fires_shp_custom_test)
        fires_shp_all = gpd.read_file(fires_shp_all)

        if year is not None:
            fires_shp_custom_test[config.name_col_y_fires] = pd.to_datetime(fires_shp_custom_test[config.name_col_y_fires])
            fires_shp_custom_test = fires_shp_custom_test[ fires_shp_custom_test[config.name_col_y_fires].dt.year == year]
            fires_shp_all[config.name_col_y_fires] = pd.to_datetime(fires_shp_all[config.name_col_y_fires])
            fires_shp_all = fires_shp_all[ fires_shp_all[config.name_col_y_fires].dt.year == year]

        # be sure to exclude no data tu the susc map
        with rio.open(dem_path) as src:
            dem_arr = src.read(1)
            dem_nodata = src.nodata
            susceptibility_arr = np.where(dem_arr == dem_nodata, -9999, susceptibility_arr)
            susceptibility_arr = np.where(susceptibility_arr < 0, -9999, susceptibility_arr) # make sure all nodata are -9999


        
        try: # if fires for this year are not available the scores will not be evaluated

            #rasterize the fires
            fires_arr_test = rasterize_numerical_feature(fires_shp_custom_test, dem_path)
            fires_arr_all =  rasterize_numerical_feature(fires_shp_all, dem_path)


            # fires test labels
            fires_test_flatten = fires_arr_test[fires_arr_test == 1]
            logging.debug(f'fires_test_flatten: {len(fires_test_flatten)}')

            # fires test predictions
            predictions_1labelled = np.where(fires_arr_test == 1, susceptibility_arr, -0.5) # here I put -0.5 instead of -9999 becasue some fires can fall in nodata
            fire_predictions_flatten = predictions_1labelled[predictions_1labelled != -0.5] 
            # if some fire points fall in nodata put susc value = 0
            fire_predictions_flatten = np.where(fire_predictions_flatten == -9999, 0, fire_predictions_flatten) # where some nodata remain put 0
            logging.debug(f'fire_predictions_flatten: {len(fire_predictions_flatten)}')

            # I want to add 0labelled pixels considering the areas that never burned in the past
            fires_arr_all = np.where(fires_arr_test == 1, 1, fires_arr_all) 
            # put nodata different from 0 to avoid to consider them as 0
            fires_arr_all = np.where(susceptibility_arr == -9999, -9999, fires_arr_all)
            not_fires_flatten = fires_arr_all[fires_arr_all == 0][0:len(fires_test_flatten)]
            logging.debug(f'not_fires_flatten: {len(not_fires_flatten)}')

            # final test labels
            labels = np.concatenate([fires_test_flatten, not_fires_flatten])
            logging.debug(f'labels: {len(labels)}')

            # now I update the predictions in 0 labelled points 
            predictions_0label = np.where(fires_arr_test == 0, susceptibility_arr, -9999)
            not_fires_prediction_flatten = predictions_0label[predictions_0label != -9999][0:len(fires_test_flatten)]

            logging.debug(f'not_fires_prediction_flatten: {len(not_fires_prediction_flatten)}')
            logging.debug(f'not fire predictions: {not_fires_prediction_flatten}')

            # final predictions
            predictions = np.concatenate([fire_predictions_flatten, not_fires_prediction_flatten])
            logging.debug(f'predictions: {len(predictions)}')

            # evalaute performances 
            auc_custom_test = roc_auc_score(labels, predictions)
            mse_custom_test = mean_squared_error(labels, predictions)

        except ValueError as e:

            logging.info(f'{e}: {year} this year is empty')
            auc_custom_test = 0
            mse_custom_test = 0

        # update results file
        with open(os.path.join(working_dir, 'results.txt') , 'a') as f:
            
            metrics = [round(auc_custom_test, 3), round(mse_custom_test, 3)]
            metrics_name = ['auc_custom_test', 'mse_custom_test']
            
            for m, n in zip(metrics, metrics_name): 
                if year is not None:
                    f.write(f'{year}')
                f.write('\n')
                f.write(f'\n{n}\n{m}\n')


        return auc_custom_test, mse_custom_test

         
        
    def RF_feature_importances(self, model, columns: list[str], out_path_csv: str):    
        
        # features impotance
        logging.info('I am evaluating features importance')       
        imp = model.feature_importances_


        # aggregating some vars
        cols, list_imp_noVar, dict_perc =  aggregate_vars_importances(columns, imp, variable_to_aggregate = 'perc')
        cols, list_imp_noVar, dict_veg =  aggregate_vars_importances(cols, list_imp_noVar, variable_to_aggregate = 'veg')
        
        logging.info('importances')
        dict_imp = dict(zip(cols, list_imp_noVar))

        
        dict_imp_sorted = {k: v for k, v in sorted(dict_imp.items(), 
                                                   key=lambda item: item[1], 
                                                   reverse=True)}
        
            
            
        dict_imp_sorted = {k: round(v,2) for k, v in dict_imp_sorted.items()}
        df = pd.DataFrame.from_dict(dict_imp_sorted, orient ='index')  
    
        df_perc = pd.DataFrame.from_dict(dict_perc, orient ='index')  
        
        df_veg = pd.DataFrame.from_dict(dict_veg, orient ='index')
        
        df_merged = pd.concat([df, df_perc, df_veg], axis = 0)
        df_merged.drop(['veg', 'perc'], axis = 0, inplace = True)
        df_merged.to_csv(out_path_csv)

        
        return df, df_perc, df_veg
        
    
    def get_results(self, model, X_all: np.array, dem_arr: np.array, mask: np.array):
        
        
        # prediction over all the points
        Y_out = model.predict_proba(X_all)
        # array of predictions over the valid pixels 
        Y_raster = np.zeros_like(dem_arr).astype(np.float32())
        Y_raster[mask] = Y_out[:,1]
        
        # clip susc where dem exsits
        Y_raster[~mask] = -1
                
        
        return Y_raster


    def adjust_dataset(self, X_all: pd.DataFrame, training_df: pd.DataFrame):
        '''
        In case the model is already trained,the dataset of the country in which we want to run the model
        has to have all the columns with the same order of the training dataset. this function adjust the 
        dataset with this scope. 
        '''
        
        # adjust size of the 2 datasets
        points = len(X_all)
        # training_df = training_df.iloc[0:points]
        
        # concat in order that X all has the same columns of training df 
        training_df_1row = training_df.iloc[0:2]

        complete_df = pd.concat([X_all, training_df_1row], axis = 0)
        complete_df = complete_df.fillna(0)
        complete_df = complete_df.iloc[0:points]
            
        for col in training_df.columns:
            if col == 'Unnamed: 0': # or len(np.unique(training_df[col])) == 1
                training_df.drop(col, axis = 1, inplace = True)
            else:
                pass

        # ordering the columns
        cols_training = training_df.columns
        
        complete_df = complete_df[cols_training]
        
        logging.debug(f'training dataset features:\n{cols_training}')
        logging.debug(f'current dataset features:\n{complete_df.columns}')
        
        complete_df = complete_df.values
        
        return complete_df
        
        
        
        
        









