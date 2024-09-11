# -*- coding: utf-8 -*-
"""
Created on Mon May  1 07:06:21 2023

@author: Giorg
"""

import rasterio as rio
import matplotlib.pyplot as plt
import numpy as np
from rasterio import features
from osgeo import gdal
import os
import time
import smtplib
import pandas as pd
import logging
 
    

def set_logging( working_dir, level = logging.INFO):

    log_path = os.path.join(working_dir, 'loggings')
    if not os.path.exists(log_path):
        os.mkdir(log_path)

    logging_date = time.time()
    logging_date = time.strftime("%Y-%m-%d", time.localtime(logging_date))

    p =  os.path.join(log_path, f'log_{logging_date}.log')
    print(f'find the logging here \n{p}')

    logging.basicConfig(format = '[%(asctime)s] %(filename)s: {%(lineno)d} %(levelname)s - %(message)s',
                        datefmt ='%H:%M:%S',
                        filename = p,
                        level = level)  
    
    logging.info(f'\n\n NEW SESSION \n\n')

            
def create_folder(path, name = 'susceptibility'):
    
    # create out folder 
    outdir = os.path.join(path, name)
    if not os.path.exists(outdir):
        os.mkdir(outdir)
        
    return outdir

def remove_features(data_dict, features_to_remove):
    
    for name in features_to_remove:
        del data_dict[name]
        
    return data_dict
        

    
def aggregate_vars_importances(variable_names: list[str], importances: list, variable_to_aggregate: str):
    '''
    variable names: list of keys or var names
    importnaces. sklearn feature_importances_ output
    variable_to_aggregate: string of name of vars which start with
    '''
    
    var_imp_list = list()
    list_imp_noVar = list()
    names_excluded_vars = list()
    
    # separate the perc featuers with the others 
    for i,j in zip(variable_names, importances):
        if i.startswith(variable_to_aggregate + '_'):
            var_imp_list.append(j)
            names_excluded_vars.append(i)
        else:
            list_imp_noVar.append(j)
    
            
    dict_imp_vartoaggr = dict(zip(names_excluded_vars, var_imp_list))
    
    dict_imp_sorted = {k: v for k, v in sorted(dict_imp_vartoaggr.items(), 
                                                key=lambda item: item[1], 
                                                reverse=True)}            
        
    dict_imp_sorted = {k: round(v,2) for k, v in dict_imp_sorted.items()}

    
    
    # aggregate perc importances
    var_imp = sum(var_imp_list)
    # add the aggregated result
    list_imp_noVar.append(var_imp)
    
    # list of columns of interest
    cols = [col for col in variable_names if not col.startswith(variable_to_aggregate + '_')]
    cols.append(variable_to_aggregate)
    
    return cols, list_imp_noVar, dict_imp_sorted

def update_dict_annual_data(annual_feature_list, name_annual_features):

    annual_layers_dict = {}
    for annual_layer, annual_layer_name in zip(annual_feature_list, name_annual_features):
        with rio.open(annual_layer) as a:
            annual_arr = a.read(1)
            
        annual_layers_dict[annual_layer_name] = annual_arr
    
    return annual_layers_dict


    
def save_raster_as(array, output_file, reference_file, clip_extent = False, **kwargs):
    
    with rio.open(reference_file) as f:
        
        profile = f.profile
        print(f'input profile\n{profile}')
        
        profile['compress'] = 'lzw'
        profile['tiled'] =  'True'

        
        profile.update(**kwargs)
        print(f'output profile\n{profile}')
                
        if len(array.shape) == 3:
            array = array[0,:,:]
        else:
            pass

        if clip_extent == True:
            f_arr= f.read(1)
            noodata = f.nodata
            array = np.where(f_arr == noodata, profile['nodata'], array)

        with rio.open(output_file, 'w', **profile) as dst:
            dst.write(array.astype(profile['dtype']), 1)
        
    return output_file
            
def plot_raster(self, array):
    
    if len(array.shape) == 3:
        array = array[0,:,:]
    else:
        pass

    
    fig, ax = plt.subplots(figsize=(5, 5))
    
    # use imshow so that we have something to map the colorbar to
    image = ax.imshow(array, 
                        cmap='seismic')
    
    fig.colorbar(image, ax=ax)     
    
def rasterize_numerical_feature(gdf, reference_file: str, column=None):
    with rio.open(reference_file) as f:
        out = f.read(1,   masked = True)
        myshape = out.shape
        mytransform = f.transform #f. ...
    del out       
    out_array = np.zeros(myshape)#   out.shape)
    # this is where we create a generator of geom, value pairs to use in rasterizing
    if column is not None:
        shapes = ((geom, value) for geom, value in zip(gdf.geometry, gdf[column]))
    else:
        shapes = ((geom, 1) for geom in gdf.geometry)
        
    burned = features.rasterize(shapes=shapes, fill=np.NaN, out=out_array, transform=mytransform, all_touched=True)
    #    out.write_band(1, burned)
    # print("rasterization completed")

    return burned
    
def create_topographic_vars(dem_path: str, out_dir: str):
    
    temp_slope = os.path.join(out_dir, 'slope.tif')
    temp_aspect = os.path.join(out_dir, 'aspect.tif')
    
    logging.info('Creating slope file')
    ds1 = gdal.DEMProcessing(temp_slope, dem_path, 'slope')
    ds1 = gdal.DEMProcessing(temp_slope, dem_path, 'slope')   
    logging.info('Creating aspect file')
    ds2 = gdal.DEMProcessing(temp_aspect, dem_path, 'aspect')
    ds2 = gdal.DEMProcessing(temp_aspect, dem_path, 'aspect')
    with rio.open(temp_slope) as slope:
        slope_arr = slope.read(1)
        
    with rio.open(temp_aspect) as aspect:
        aspect_arr = aspect.read(1)
        northing_arr = np.cos(aspect_arr * np.pi/180.0)
        easting_arr = np.sin(aspect_arr * np.pi/180.0)
        
    ds1 = None
    ds2 = None
    
    # give the time to file to close
    time.sleep(4)
    
    try:
        os.remove(temp_slope)
        os.remove(temp_aspect)
    except:
        logging.warning('BUG: I cannot delete this file')
        pass
        
    return slope_arr, northing_arr, easting_arr

    
def reproject_layer(ref_arr, arr, reference_file, raster_input_path, out_dir):
    
    if ref_arr.shape != arr.shape:
        logging.warning(f'WARNING: {raster_input_path} has different size from DEM, resampling')

        # create a folder and filename to store reporjected file
        reproject_folder = os.path.join(out_dir, 'reprojected')
        os.makedirs(reproject_folder, exist_ok = True)
        filename = os.path.basename(raster_input_path)
        filename = filename.split('.')[0] + '_reprojected.tif'
        
        out_resampled = os.path.join(reproject_folder, filename)
        
        ref = rio.open(reference_file)
        in_ras = rio.open(raster_input_path)
            
        Res = ref.transform[0]
        # target bounds: (minX, minY, maxX, maxY)
        # rasterio bounds: left, bottom, right, top
        ref_bounds = ref.bounds

        # reproject before then clip over extent 
        temporary_name = os.path.join(out_dir, 'temprepr.tif')
        gdal.Warp(temporary_name, raster_input_path, dstSRS = ref.crs, srcSRS=in_ras.crs)
        time.sleep(6)
        gdal.Warp(out_resampled, temporary_name,
                        outputBounds = ref_bounds, xRes=Res, yRes=Res,
                        srcSRS = ref.crs, dstSRS = ref.crs, dstNodata = -9999,
                        creationOptions=["COMPRESS=LZW", "PREDICTOR=2", "ZLEVEL=3", "BLOCKXSIZE=512", "BLOCKYSIZE=512"])    
        
        ref.close()
        in_ras.close()
        with rio.open(out_resampled) as f:
            resampled_arr = f.read(1)
            
        time.sleep(4)
        
        # os.remove(out_resampled)
        os.remove(temporary_name)

        
        if resampled_arr.shape == ref_arr.shape:
            logging.info('resampled correctly')
        else:
            logging.critical(f'ERROR: perhaps input arr was not resampled correctly: shapes are {resampled_arr.shape} (resampled) and {arr.shape}')
            raise ValueError('resampling error')
    else:
        resampled_arr = arr
    
    return resampled_arr
    
def get_lat_lon_arrays_from_raster(raster_path: str) -> tuple[np.array, np.array]:
        
        with rio.open(raster_path) as src:

            # Get the geographic coordinate transform (affine transform)
            transform = src.transform
            # Generate arrays of row and column indices
            rows, cols = np.indices((src.height, src.width))
            # Transform pixel coordinates to geographic coordinates
            lon, lat = transform * (cols, rows)

        return lat, lon

def raster_classes(data, quantile, nodata, norm = False, quantiles = True):
    '''
    save a raster with classes following input quantiles
    '''

    # quantile = [0.25, 0.75] # metedologia paolo
    
    data3 = np.where(data == nodata, np.NaN, data)
    del data
    if norm == True:
        data3 = data3/np.nanmax(data3)
    else:
        pass
    
    if quantiles == True:
        q = np.nanquantile(data3, quantile)
        
        logging.info(f'quantile values {q}')
        
        # force the 0 values to be 0.001 for palette purposes
        if q[0] < 0.001:
            q[0] = 0.001
        
        if q[1] < 0.001:
            q[1] = 0.002
            
        bounds = [0] + list(q) + [1]
    else:
        bounds = [0] + quantile + [1]
    
    # convert the raster map into a categorical map based on quantile values
    conditions = list()
    for i in range(0, len(quantile)+1 ):
        # first position take also ssuc = 0, the dosnt take the low limit
        if i == 0:
            conditions.append( ((data3 >= bounds[i]) & (data3 <= bounds[i+1])) )
        else:
            conditions.append(((data3 > bounds[i]) & (data3 <= bounds[i+1])))
    

    classes = [i for i in range(1, len(bounds))]
    logging.debug(classes)
    
    out_arr = np.select(conditions, classes, default=0)
    out_arr = out_arr.astype(np.int8())
    
    return out_arr



def send_email(mail_user, mail_password, message):
    
    
    sent_from = mail_user
    to = [mail_user, 
            'giorgio.meschi@cimafoundation.org', 
            'farzad.ghasemiazma@cimafoundation.org',
            'andrea.trucchia@cimafoundation.org'
            ]
    subject = '[BOLIVIA ML MODEL] - RUN UPDATES '
    body = message
    
    # spazi e dove andare a capo deve seguire formattazione corretta standard
    email_text = 'From: {}\nTo: {}\nSubject: {}\n\n{}'.format(sent_from, ", ".join(to), subject, body)

    try:
        server = smtplib.SMTP('smtp.cimafoundation.org', 25)
        #server = smtplib.SMTP('smtp.gmail.com', 587)
        server.ehlo()
        server.login(mail_user, mail_password)
        server.sendmail(sent_from, to, email_text)
        server.close()
    
        print('Email sent!')
    except:
        print('no mail is sent')
        

def __veg_burned_area(ff_gdf, arr, reference_file):
    '''
    this function clip the raster of veg or susc over defined BA
    '''
    
    #clipped_img, clip_transform = rio.mask.mask(arr, inc.geometry)
    # rasterize ff_gdf
    try:
        burned = rasterize_numerical_feature(ff_gdf, reference_file)
    # except ValueError as e:
    #     print(e)
    #     burned = np.zeros(arr.shape)
    # mask arr with burned
        clipped_img = np.where(burned == 1, arr, 0)
        
        clipped_img = clipped_img.astype(int)

        # burned pixels
        cli = clipped_img[clipped_img > 0]

        # classes
        data = np.unique(cli, return_counts = True)

        classes = data[0]
        numbers = data[1]
        
        class_ = pd.Series(classes)
        num = pd.Series(numbers)

        df = pd.DataFrame( columns = ['class', 'num_of_burned_pixels'])
        
        df['class'] = class_
        df['num_of_burned_pixels'] = num
    except ValueError as e:
        print(e)
        df = pd.DataFrame( columns = ['class', 'num_of_burned_pixels'])

    return df

def df_statistic(arr: np.array, ff_gdf, col_years: str, reference_file: str, pixel_size: int, single_year: int = None):

    ''' 
    prepare a dataset of fires considering every year of BA associated with veg or susc classes
    '''

    df1 = pd.DataFrame( columns = ['class', 'num_of_burned_pixels'])

    if single_year is not None:
    
        inc1 = ff_gdf[ff_gdf[col_years] == single_year]
        logging.debug(len(inc1))
        df = __veg_burned_area(inc1, arr, reference_file)
        df1 = pd.concat([df1,df], axis=0)
        logging.debug('len dataframe: ', len(df1))
        
    else:
        for i in np.unique(ff_gdf[col_years]):
            
            logging.debug('year: ', i)
            
            inc1 = ff_gdf[ff_gdf[col_years] == i]
            logging.debug(f'lenght polygons for this year {len(inc1)}')
            df = __veg_burned_area(inc1, arr, reference_file)
            df1 = pd.concat([df1, df], axis=0)
            logging.debug('len updated dataframe: ', len(df1))
            
    try:    
        stats = df1.groupby('class').agg('sum')

        pixel_area_ha = (pixel_size*pixel_size/1000000)*100

        stats['burned_area(ha)'] = stats['num_of_burned_pixels']*pixel_area_ha

        
        cl, num_overall = np.unique(arr, return_counts = True)

        num_overall_s = pd.Series(num_overall, name='tot_num_of_pixels_in_class', index = cl)

        df3 = pd.merge(stats, num_overall_s, left_index=True, right_index=True)

        # this shows the burned area over total burned area
        df3['total_area'] = df3['tot_num_of_pixels_in_class']*pixel_area_ha
        total_burned_area = df3['burned_area(ha)'].sum()
        df3['percentage_of_class_burned_area_to_total_burned_area'] = df3["burned_area(ha)"]*100/total_burned_area
        df3['percentage_of_class_burned_area_to_total_burned_area_of_class'] = df3["burned_area(ha)"]*100/df3["total_area"]
    except KeyError:
        df3 = None

    return df3

def check_size_on_memory(self, variable):

    import sys
    size = sys.getsizeof(variable)

    power = 2**10
    n = 0
    power_labels = {0: '', 1: 'KB', 2: 'MB', 3: 'GB', 4: 'TB'}
    while size > power:
        size /= power
        n += 1

    print(f"{size:.2f} {power_labels[n]}")

def batch_by_rows(arr_to_batch: np.array, total_batch_num: int, batch_number: int):
    '''
    arr_to_batch is the input array that will be sliced  depending on batch numer 
    tot num of batches is from 1 to x 
    batch number identify the index of the batch, starts from 0
    '''
    rows = arr_to_batch.shape[0]
    interval = int(rows/total_batch_num)
    intervals = [i for i in range(0, rows+1, interval)]
    batch_row_start = intervals[batch_number]
    batch_row_finish = intervals[batch_number + 1]

    # include the missed rows due to rounding operation:
    if total_batch_num - batch_number == 1: # it means last batch
        last_row_wrong = intervals[-1]
        last_row_right = last_row_wrong + (rows - last_row_wrong)
        batch_row_finish = last_row_right
        
    batched_array = arr_to_batch[batch_row_start : batch_row_finish, :]

    return batched_array, batch_row_start, batch_row_finish








