import rasterio as rio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import rasterio.crs

def reproject_raster_as(input_file, output_file, reference_file):
    with rio.open(reference_file) as ref:
        bounds = ref.bounds
        res = ref.transform[0]
        crs = ref.crs


    with rio.open(input_file) as src:       
        src_crs = src.crs
        transform, width, height = calculate_default_transform(src_crs, crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
       
        kwargs.update({
            'crs': crs,
            'transform': transform,
            'width': width,
            'height': height})
       
        with rio.open(output_file, 'w', **kwargs) as dst:
            for ii in range(1, src.count + 1):
                reproject(
                    source=rio.band(src, ii),
                    destination=rio.band(dst, ii),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=crs,
                    resampling=Resampling.nearest)


