import rasterio as rio
try:
    from osgeo import gdal
except Exception as e:
    print(e)
    print('trying importing gdal direclty')
    import gdal

def reproject_raster_as(input_file: str, output_file: str, reference_file: str) -> str:
    '''
    reproj and clip raster based on reference file
    '''

    with rio.open(input_file) as file_i:
        input_crs = file_i.crs
        #bounds = haz.bounds
    with rio.open(reference_file) as ref:
        bounds = ref.bounds
        res = ref.transform[0]
        output_crs = ref.crs

    gdal.Warp(output_file, input_file,
                    outputBounds = bounds, xRes=res, yRes=res,
                    srcSRS = input_crs, dstSRS = output_crs, dstNodata = -9999,
                    creationOptions=["COMPRESS=LZW", "PREDICTOR=2", "ZLEVEL=3", "BLOCKXSIZE=512", "BLOCKYSIZE=512"])    
    
    return output_file
    
