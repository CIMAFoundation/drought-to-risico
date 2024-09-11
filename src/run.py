

#%%

from datetime import datetime as dt
import os
import sys
import logging
import numpy as np

from settings import BASEP

from settings import (indices, operational_drought_datapath, 
                      year, month, monthname, ssmi_rawfile, edi_rawfile, combined_rawfile,
                      dem_path, veg_path, output_dir, model_path, X_path, model_config, 
                      montlhy_folder_name, susceptibility_out_path, dem_wgs_path, slope_wgs_path,
                      aspect_wgs_path, fuel12_wgs_path, risico_output_path)

from reproject import reproject_raster_as
from fuel12cl import save_raster_as, hazard_12cl_assesment
from get_risico_file import get_risico_static_file
from wildfire_susceptibility.susceptibility import Susceptibility


logging.basicConfig(format = '[%(asctime)s] %(filename)s: {%(lineno)d} %(levelname)s - %(message)s',
                    datefmt ='%H:%M:%S',
                    filename = f'logging.log',
                    level = logging.INFO)  



#%%

def run():

    # check if drought datapath is valid (isfile) otherwise take the one of month before
    # ssmi
    ssmi_rawpath = ssmi_rawfile(year, monthname)
    ssmi_actualmonth = month
    _year = year
    while not os.path.isfile(f'{ssmi_rawpath}'):
        ssmi_actualmonth = ssmi_actualmonth - 1
        _year = _year -1 if ssmi_actualmonth == 0 else _year
        if ssmi_actualmonth == 0:
            ssmi_actualmonth = 12
        _monthname = f'0{ssmi_actualmonth}' if ssmi_actualmonth < 10 else ssmi_actualmonth
        ssmi_rawpath = ssmi_rawfile(_year, _monthname)
        logging.info(f'No SSMI file for {year}_{month} taking {_year}_{ssmi_actualmonth}')
        logging.info(ssmi_rawpath)

    # edi
    edi_rawpath = edi_rawfile(year, monthname)
    edi_actuamonth = month
    _year = year
    while not os.path.isfile(f'{edi_rawpath}'):
        edi_actuamonth = edi_actuamonth - 1
        _year = _year -1 if edi_actuamonth == 0 else _year
        if edi_actuamonth == 0:
            edi_actuamonth = 12
        _monthname = f'0{edi_actuamonth}' if edi_actuamonth < 10 else edi_actuamonth
        edi_rawpath = edi_rawfile(_year, _monthname)
        logging.info(f'No EDI file for {year}_{month} taking {year}_{edi_actuamonth}')
        logging.info(edi_rawpath)

    # combined
    combined_rawpath = combined_rawfile(year, monthname)
    combined_actualmonth = month
    _year = year
    while not os.path.isfile(f'{combined_rawpath}'):
        combined_actualmonth = combined_actualmonth - 1
        _year = _year -1 if combined_actualmonth == 0 else _year
        if combined_actualmonth == 0:
            combined_actualmonth = 12 
        _monthname = f'0{combined_actualmonth}' if combined_actualmonth < 10 else combined_actualmonth
        combined_rawpath = combined_rawfile(_year, _monthname)
        logging.info(f'No combined file for {year}_{month} taking {year}_{combined_actualmonth}')
        logging.info(combined_rawpath)

    # create the name of the final fuel12cl, if already exist dont do anything
    fuel12cl_path = f'{BASEP}/script_fuel12cl_risico/fuel12_risico/fuel12cl_{year}_{month}_ssmi{ssmi_actualmonth}edi{edi_actuamonth}comb{combined_actualmonth}.tif'

    if not os.path.isfile(fuel12cl_path):
        logging.info(f'Creating {os.path.basename(fuel12cl_path)}')

        raster_files = [ssmi_rawpath, edi_rawpath, combined_rawpath]
        out_files = [f'{operational_drought_datapath}/{idx}.tif' for idx in indices]
        for raw_file, repr_file in zip(raster_files, out_files):
            reproject_raster_as(raw_file, repr_file, dem_path)

        # dictionary of input dymi files with structure coherent with Suseptibility module
        monthly_files = {montlhy_folder_name: {tiffile : f'{operational_drought_datapath}/{tiffile}.tif'
                            for tiffile in indices}
                        }


        susceptibility = Susceptibility(dem_path, veg_path, # mandatory vars
                                        working_dir = output_dir,
                                        optional_input_dict = {}, # optional layers
                                        config = model_config # configuration file
                                        ) 


        # create a model and save it
        susceptibility.run_existed_model_annual(model_path, 
                                                annual_features_paths = monthly_files,
                                                training_df_path = X_path,
                                                start_year = list(monthly_files.keys())[0])


        # create the fuel12cl
        hazard, _, _ = hazard_12cl_assesment(susceptibility_out_path, 
                                            thresholds = [0.33, 0.66], 
                                            veg_path = veg_path, 
                                            mapping_path = f'{BASEP}/script_fuel12cl_risico/script/veg_mapping.json', 
                                            out_hazard_file = fuel12cl_path)

        # convert in risico file
        if not os.path.exists(slope_wgs_path): # if slope and aspect already exist dont do it
            # use gdal to create slope and aspect
            try: 
                from osgeo import gdal
            except Exception as e:
                logging.info(e)
                logging.info('trying importing gdal direclty')
                import gdal
            # create file using dem (not wgs)
            gdal.DEMProcessing(slope_wgs_path.replace('.tif', '0.tif'), dem_path, 'slope')
            gdal.DEMProcessing(aspect_wgs_path.replace('.tif', '0.tif'), dem_path, 'aspect')
            # reproject
            reproject_raster_as(slope_wgs_path.replace('.tif', '0.tif'), slope_wgs_path, dem_wgs_path)
            reproject_raster_as(aspect_wgs_path.replace('.tif', '0.tif'), aspect_wgs_path, dem_wgs_path)
            os.remove(slope_wgs_path.replace('.tif', '0.tif'))
            os.remove(aspect_wgs_path.replace('.tif', '0.tif'))

        # create a txt file in which each row has x and y coordinates and the value of the hazard
        reproject_raster_as(fuel12cl_path, fuel12_wgs_path, dem_wgs_path)
        get_risico_static_file(fuel12_wgs_path, slope_wgs_path, aspect_wgs_path, risico_output_path)


        logging.info(f'{risico_output_path} created')

    else:
        logging.info(f'{os.path.basename(fuel12cl_path)} already exist')


if __name__ == '__main__':
    run()
        

