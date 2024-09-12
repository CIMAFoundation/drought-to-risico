# settings 
from datetime import datetime as dt
import os

RAW_DROUGHT_PATH = '/home/sequia/drought/products'
BASE_PATH = '.' # server path


DATA_PATH = f'{BASE_PATH}/data'
MODEL_PATH = f'{BASE_PATH}/model/RF.sav'
X_PATH = f'{BASE_PATH}/model/X.csv'

INDICES = ['SSMI01-HSAF', 'EDI03-PERSIANN', 'Combined-SPEI03-SWDIHSAF02-VHI01-FAPAR01']

OUTPUT_DROUGHT_PATH = f'{DATA_PATH}/drought'
DEM_PATH = f"{DATA_PATH}/static_data/DEM_SC_100_32721.tif"
VEG_PATH = f"{DATA_PATH}/static_data/veg_SantaCruz_ESACCI-LC-L4-LCCS-Map-300m-P1Y-2015-v2.0.7.tif" 
OUTPUT_DIR = f'{BASE_PATH}/fuel12_risico'
VEG_MAPPING_PATH = f'{BASE_PATH}/src/veg_mapping.json'

MODEL_CONFIG = {    
    "batches" : 5, 
    "nb_codes_list" : [255, 0, 180, 190, 200, 201, 202, 210],
    "list_features_to_remove" : [ "perc_0"],
    "convert_to_month" : 1, 
    "wildfire_years" : [],
    "nordic_countries" : {}, 
    "save_dataset" : 0,
    "reduce_fire_points" : 8,
    "gridsearch" : 0,
    "ntree" : 750,
    "max_depth" : 15,
    "percentiles_absence_presence": [10, 90],
    "drop_neg_and_na_annual": 0,
    "name_col_y_fires" : "ig_date",
    "make_CV" : 0,
    "make_plots" : 0,    
    "validation_ba_dict" :   {"fires10" : "",  "fires90" : "" },
    "country_name" : "", 
    "pixel_size" : 100, 
    "user_email" : "",
    "email_pwd" : ""

}  

# input for risico file conversion
DEM_WGS_PATH = f"{DATA_PATH}/static_data/DEM_SC_500_4326.tif"
SLOPE_WGS_PATH = f"{DATA_PATH}/static_data/slope_SC_500_4326.tif"
ASPECT_WGS_PATH = f"{DATA_PATH}/static_data/aspect_SC_500_4326.tif"
FUEL12_WGS_PATH = f'{OUTPUT_DIR}/fuel12cl_wgs.tif'
RISICO_OUTPUT_PATH = f"{BASE_PATH}/bolivia.txt"
