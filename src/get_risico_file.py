
import rasterio as rio
import numpy as np
import os

import logging

def write_risico_files(fuel12cl_path, slope_path, aspect_path, outfile):
    logging.info(f'read {fuel12cl_path}')
    with rio.open(fuel12cl_path, 'r') as src:
        values = src.read(1)

        # Get the geographic coordinate transform (affine transform)
        transform = src.transform
        # Generate arrays of row and column indices

    logging.info('Generate arrays of row and column indices')
    rows, cols = np.indices((src.height, src.width))        
    # mask rows and cols to get only the valid pixels (where hazard not 0)
    rows = rows[values != 0]
    cols = cols[values != 0]
    # Transform pixel coordinates to geographic coordinates
    lon, lat = transform * (cols, rows)
    logging.info('Extract values')
    hazard = values[rows, cols]
    

    # get value of raster in those coordinates
    with rio.open(slope_path) as src:
        values = src.read(1)
        slope = values[rows, cols]
    
    with rio.open(aspect_path) as src:
        values = src.read(1)
        aspect = values[rows, cols] 
        
                
    with open(outfile, 'w') as ff:
        # write header
        ff.write('# lon lat slope aspect veg_id \n')
        for i in range(len(hazard)):
            veg_id = hazard[i]
            if veg_id < 0:
                veg_id = 0
            new_line = f'{lon[i]:.5f} {lat[i]:.5f} {slope[i]:.2f} {aspect[i]:.2f} {veg_id}'
            ff.write(f'{new_line}\n')


