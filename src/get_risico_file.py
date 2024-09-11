
import rasterio as rio
import numpy as np
import os

def get_risico_static_file(fuel12cl_path, slope_path, aspect_path, outfile):
    with rio.open(fuel12cl_path, 'r') as src:

        # Get the geographic coordinate transform (affine transform)
        transform = src.transform
        # Generate arrays of row and column indices
        rows, cols = np.indices((src.height, src.width))
        # mask rows and cols to get only the valid pixels (where hazard not 0)
        rows = rows[src.read(1) != 0]
        cols = cols[src.read(1) != 0]
        # Transform pixel coordinates to geographic coordinates
        lon, lat = transform * (cols, rows)
            
        hazard = list(rio.sample.sample_gen(src, list(zip(lon, lat)) ))
    

    static_file = outfile.replace('.txt', '_no_hazard.txt')
    if not os.path.exists(static_file):
        # get value of raster in those coordinates
        with rio.open(slope_path) as src:
            slope = list(rio.sample.sample_gen(src, list(zip(lon, lat)) ))
        
        with rio.open(aspect_path) as src:
            aspect = list(rio.sample.sample_gen(src, list(zip(lon, lat)) ))
        
                
        # create before a file without hazard info, then open it and add hazard
        with open(static_file, 'w') as f:
            for i in range(len(lat)):
                f.write(f'{lon[i]} {lat[i]} {slope[i][0]} {aspect[i][0]}\n')
    
    # now open this last file and add the hazard
    with open(static_file, 'r') as f:
        lines = f.readlines()
        # now open the file to write
        with open(outfile, 'w') as ff:
            for i, line in enumerate(lines):
                ff.write(f'{line.strip()} {hazard[i][0]}\n')


