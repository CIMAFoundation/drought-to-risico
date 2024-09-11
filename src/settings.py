# settings 
from datetime import datetime as dt
import os

BASEP = './' # server path
# BASEP = '/share/home/gruppo4/Bolivia/DATA'
DATAPATH = f'{BASEP}/script_fuel12cl_risico/data'
MODELPATH = f'{BASEP}/script_fuel12cl_risico/model'


indices = ['SSMI01-HSAF', 'EDI03-PERSIANN', 'Combined-SPEI03-SWDIHSAF02-VHI01-FAPAR01']

raw_drough_datapath = f'{BASEP}/products'
operational_drought_datapath = f'{DATAPATH}/drought'

year = dt.now().year
month = dt.now().month

montlhy_folder_name = f'{year}_{month}'

monthname = f'0{month}' if month < 10 else month

ssmi_rawfile = lambda year, monthname: f'{raw_drough_datapath}/SSMI/{year}/{monthname}/SSMI01-HSAF_{year}{monthname}.tif'
edi_rawfile = lambda year, monthname: f'{raw_drough_datapath}/EDI/{year}/{monthname}/EDI03-PERSIANN_{year}{monthname}.tif'
combined_rawfile = lambda year, monthname: f'{raw_drough_datapath}/combined/{year}/{monthname}/Combined-SPEI03-SWDIHSAF02-VHI01-FAPAR01_{year}{monthname}.tif'


dem_path = f"{DATAPATH}/static_data/DEM_SC_100_32721.tif"
veg_path = f"{DATAPATH}/static_data/veg_SantaCruz_ESACCI-LC-L4-LCCS-Map-300m-P1Y-2015-v2.0.7.tif" 
output_dir = f'{BASEP}/script_fuel12cl_risico/fuel12_risico'

model_path = f'{MODELPATH}/RF.sav'
X_path = f'{MODELPATH}/X.csv'

susceptibility_out_path = f'{output_dir}/susceptibility/annual_maps/Annual_susc_{year}_{month}.tif'

model_config = {    
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
dem_wgs_path = f"{DATAPATH}/static_data/DEM_SC_500_4326.tif"
slope_wgs_path = f"{DATAPATH}/static_data/slope_SC_500_4326.tif"
aspect_wgs_path = f"{DATAPATH}/static_data/aspect_SC_500_4326.tif"
fuel12_wgs_path = f'{output_dir}/fuel12cl_wgs.tif'
risico_output_path = f"{BASEP}/script_fuel12cl_risico/bolivia.txt"
