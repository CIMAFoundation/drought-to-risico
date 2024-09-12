

#%%

from datetime import datetime as dt
from datetime import timedelta
import os
import sys
import logging
import numpy as np

from settings import (INDICES, OUTPUT_DROUGHT_PATH, 
                      RAW_DROUGHT_PATH,
                      DEM_PATH, VEG_PATH, OUTPUT_DIR, MODEL_PATH, X_PATH, MODEL_CONFIG, 
                      DEM_WGS_PATH, SLOPE_WGS_PATH,
                      ASPECT_WGS_PATH, FUEL12_WGS_PATH, RISICO_OUTPUT_PATH,
                      VEG_MAPPING_PATH)

from reproject import reproject_raster_as
from fuel12cl import save_raster_as, hazard_12cl_assesment
from get_risico_file import write_risico_files
from wildfire_susceptibility.susceptibility import Susceptibility

try: 
    from osgeo import gdal
except Exception as e:
    logging.info(e)
    logging.info('trying importing gdal direclty')
    import gdal




logging.basicConfig(
    format = '[%(asctime)s] %(filename)s: {%(lineno)d} %(levelname)s - %(message)s',
    datefmt ='%H:%M:%S',
    handlers=[
        logging.FileHandler('logging.log'),
        logging.StreamHandler()
    ],
    level=logging.INFO
)  




def get_ssmi_rawfile(date): 
    year = date.strftime('%Y')
    month = date.strftime('%m')
    return f'{RAW_DROUGHT_PATH}/SSMI/{year}/{month}/SSMI01-HSAF_{year}{month}.tif'

def get_edi_rawfile(date): 
    year = date.strftime('%Y')
    month = date.strftime('%m')
    return f'{RAW_DROUGHT_PATH}/EDI/{year}/{month}/EDI03-PERSIANN_{year}{month}.tif'

def get_combined_rawfile(date): 
    year = date.strftime('%Y')
    month = date.strftime('%m')
    return f'{RAW_DROUGHT_PATH}/Combined/{year}/{month}/Combined-SPEI03-SWDIHSAF02-VHI01-FAPAR01_{year}{month}.tif'

def find_latest(path_fn, date):
    oldest_date = date - timedelta(days=90)
    current_date = date
    found = False
    while current_date > oldest_date:
        rawpath = path_fn(current_date)
        
        if os.path.isfile(rawpath):
            found = True
            break
        
        current_date = date - timedelta(days=15)
    if not found:
        raise ValueError('Could not find data')

    return rawpath, current_date



def run():
    date = dt.now()
    year = dt.now().year
    month = dt.now().month
    montlhy_folder_name = f'{year}_{month}'
    susceptibility_out_path = f'{OUTPUT_DIR}/susceptibility/annual_maps/Annual_susc_{montlhy_folder_name}.tif'

    try:
        ssmi_rawpath, found_date = find_latest(get_ssmi_rawfile, date)
        ssmi_actualmonth = found_date.strftime('%m')
    except ValueError:
        raise Exception("Could not find data for SSMI in the latest 90 days")
    
    try:
        edi_rawpath, found_date = find_latest(get_edi_rawfile, date)
        edi_actuamonth = found_date.strftime('%m')
    except ValueError:        
        raise Exception("Could not find data for EDI in the latest 90 days")
    
    try:
        combined_rawpath, found_date = find_latest(get_combined_rawfile, date)
        combined_actualmonth = found_date.strftime('%m')
    except ValueError:        
        raise Exception("Could not find data for Combined in the latest 90 days")

    logging.info(f'''
    found 
        COMBINED: {combined_rawpath}
        EDI: {edi_rawpath} 
        SSMI{ssmi_rawpath}
    ''')

    # keep track if a new file has to be created
    overwrite = False

    # create the name of the final fuel12cl, if already exist dont do anything
    fuel12cl_path = f'{OUTPUT_DIR}/fuel12cl_{year}_{month}_ssmi{ssmi_actualmonth}edi{edi_actuamonth}comb{combined_actualmonth}.tif'
    if not os.path.isfile(fuel12cl_path):
        logging.info(f'Creating {os.path.basename(fuel12cl_path)}')

        raster_files = [ssmi_rawpath, edi_rawpath, combined_rawpath]
        out_files = [f'{OUTPUT_DROUGHT_PATH}/{idx}.tif' for idx in INDICES]
        for raw_file, repr_file in zip(raster_files, out_files):
            reproject_raster_as(raw_file, repr_file, DEM_PATH)

        # dictionary of input dymi files with structure coherent with Suseptibility module
        monthly_files = {montlhy_folder_name: {tiffile : f'{OUTPUT_DROUGHT_PATH}/{tiffile}.tif'
                            for tiffile in INDICES}
                        }


        susceptibility = Susceptibility(DEM_PATH, VEG_PATH, # mandatory vars
                                        working_dir = OUTPUT_DIR,
                                        optional_input_dict = {}, # optional layers
                                        config = MODEL_CONFIG # configuration file
                                        ) 


        # create a model and save it
        susceptibility.run_existed_model_annual(MODEL_PATH, 
                                                annual_features_paths = monthly_files,
                                                training_df_path = X_PATH,
                                                start_year = list(monthly_files.keys())[0])


        # create the fuel12cl
        hazard, _, _ = hazard_12cl_assesment(susceptibility_out_path, 
                                            thresholds=[0.33, 0.66], 
                                            veg_path=VEG_PATH, 
                                            mapping_path=VEG_MAPPING_PATH, 
                                            out_hazard_file=fuel12cl_path)


        overwrite = True

    else:
        logging.info(f'{os.path.basename(fuel12cl_path)} already exists')

    if not os.path.exists(RISICO_OUTPUT_PATH) or overwrite:
        # convert in risico file
        if not os.path.exists(SLOPE_WGS_PATH): # if slope and aspect already exist dont do it
            # use gdal to create slope and aspect
            # create file using dem (not wgs)
            gdal.DEMProcessing(SLOPE_WGS_PATH.replace('.tif', '0.tif'), DEM_PATH, 'slope')
            gdal.DEMProcessing(ASPECT_WGS_PATH.replace('.tif', '0.tif'), DEM_PATH, 'aspect')
            # reproject
            reproject_raster_as(SLOPE_WGS_PATH.replace('.tif', '0.tif'), SLOPE_WGS_PATH, DEM_WGS_PATH)
            reproject_raster_as(ASPECT_WGS_PATH.replace('.tif', '0.tif'), ASPECT_WGS_PATH, DEM_WGS_PATH)
            os.remove(SLOPE_WGS_PATH.replace('.tif', '0.tif'))
            os.remove(ASPECT_WGS_PATH.replace('.tif', '0.tif'))

        # create a txt file in which each row has x and y coordinates and the value of the hazard
        reproject_raster_as(fuel12cl_path, FUEL12_WGS_PATH, DEM_WGS_PATH)
        write_risico_files(FUEL12_WGS_PATH, SLOPE_WGS_PATH, ASPECT_WGS_PATH, RISICO_OUTPUT_PATH)
        logging.info(f'{RISICO_OUTPUT_PATH} created')


if __name__ == '__main__':
    run()
        

