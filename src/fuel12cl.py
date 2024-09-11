

import numpy as np
import json
import rasterio as rio


def read_1band(path: str, band = 1) -> np.array:

    with rio.open(path) as f:
        arr = f.read(band)
        
    return arr

def categorize_raster(array: np.ndarray, thresholds: list[int], nodata: float) -> np.array:
    '''
    return an array categorized in calsses on thresholds with out nodata = 0  
    '''
    if np.isnan(np.array(nodata)) == True:
        array = np.where(np.isnan(array)==True, -9E6, array)
        nodata = -9E6
    
    mask = np.where(array == nodata, 0, 1)

    # Convert the raster map into a categorical map based on quantile values
    out_arr = np.digitize(array, thresholds, right=True)
    # make values starting from 1
    out_arr = out_arr + 1 
    out_arr = out_arr.astype(np.int8())

    # mask out no data
    categorized_arr = np.where(mask == 0, 0, out_arr)

    return categorized_arr


def remap_raster(array: np.array, mapping: dict, nodata = 0) -> np.array:
    '''
    reclassify an array with numpy, make sure all input classes are defined in the mapping table, othewise assertion error will be raised
    passed nodata will be mapped with 0
    '''
    
    input_codes = [ int(i) for i in mapping.keys() ]
    output_codes = [ int(i) for i in mapping.values() ]

    # add codification for no data:0
    output_codes.extend([nodata])
    input_codes.extend([0])

    # convert numpy array
    input_codes_arr = np.array(input_codes)
    output_codes_arr = np.array(output_codes)
            
    # check all values in the raster are present in the array of classes
    assert np.isin(array, input_codes_arr).all()
    
    # do the mapping with fancy indexing 
    # find indeces of input codes in input array
    sort_idx = np.argsort(input_codes_arr)
    idx = np.searchsorted(input_codes_arr, array, sorter = sort_idx) # put indeces of input codes in input array
    
    # create an array with indices of new classes linked to indices of input codes, 
    # then in the previous array of indeces put such new classes
    mapped_array = output_codes_arr[sort_idx][idx] 
    
    print(f'List of classes of mapped array: {np.unique(mapped_array)}')
    
    return mapped_array

def contigency_matrix_on_array(xarr, yarr, xymatrix, nodatax, nodatay) -> np.array:
    '''
    xarr: 2D array, rows entry of contingency matrix (min class = 1, nodata = nodatax)
    yarr: 2D array, cols entry of contingency matrix (min class = 1, nodata = nodatax)
    xymatrix: 2D array, contingency matrix
    nodatax1: value for no data in xarr
    nodatax2: value for no data in yarr
    '''


    if np.isnan(np.array(nodatax)) == True:
        xarr = np.where(np.isnan(xarr)==True, 999999, xarr)
        nodatax = 999999
    
    if np.isnan(np.array(nodatay)) == True:
        yarr = np.where(np.isnan(yarr)==True, 999999, yarr)
        nodatay = 999999

    # if arr have nan 8differenct from passed nodata), mask it with lowest class
    xarr = np.where(np.isnan(xarr)==True, 1, xarr)
    yarr = np.where(np.isnan(yarr)==True, 1, yarr)

    # convert to int
    xarr = xarr.astype(int)
    yarr = yarr.astype(int)

    mask = np.where(((xarr == nodatax) | (yarr == nodatay)), 0, 1) # mask nodata   

    # put lowest class in place of no data
    yarr[mask == 0] = 1
    xarr[mask == 0] = 1

    # apply contingency matrix
    output = xymatrix[ xarr - 1, yarr - 1]

    # mask out no data
    output[mask == 0] = 0

    return output

def save_raster_as(array: np.array, output_file: str, reference_file: str, clip_extent = False, **kwargs) -> str:
    '''
    save raster based on reference file, set clip_extent = True to automatically set nodata value as the reference file
    '''
    
    with rio.open(reference_file) as f:
        
        profile = f.profile
        
        profile['compress'] = 'lzw'
        profile['tiled'] =  'True'

        profile.update(**kwargs)
                
        if len(array.shape) == 3:
            array = array[0,:,:]

        if clip_extent == True:
            f_arr= f.read(1)
            noodata = f.nodata
            array = np.where(f_arr == noodata, profile['nodata'], array)

        with rio.open(output_file, 'w', **profile) as dst:
            dst.write(array.astype(profile['dtype']), 1)
        
    return output_file


def hazard_12cl_assesment(susc_path: str, thresholds: list, veg_path: str, mapping_path: str, out_hazard_file: str) -> tuple:
    '''
    susc path is the susceptibility file, contineous values, no data -1
    threasholds are the values to categorize the susceptibility (3classes)
    veg_path is the input vegetation file
    mapping_path is the json file with the mapping of vegetation classes (input veg: output FT class from 1 to 4)
    where FT class are 1: grasslands, 2: broadleaves, 3: shrubs, 4: conifers.
    out_hazard_file is the output hazard file
    
    Return: wildfire hazard, susc classes and fuel type array
    '''

    matrix = np.array([ [1, 4, 7, 10],
                        [2, 5, 8, 11],
                        [3, 6, 9, 12]])
            
    susc = read_1band(susc_path)
    susc_cl = categorize_raster(susc, thresholds, nodata = -1)
    veg = read_1band(veg_path)
    mapping = json.load(open(mapping_path))
    ft = remap_raster(veg, mapping)
    hazard = contigency_matrix_on_array(susc_cl, ft, matrix, nodatax = 0, nodatay = 0)
    save_raster_as(hazard, out_hazard_file, reference_file = susc_path, dtype = np.int8(), nodata = 0)
    
    return hazard, susc_cl, ft

