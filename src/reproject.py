import rasterio as rio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import rasterio.crs

def reproject_raster_as(input_file, output_file, reference_file):
    with rio.open(reference_file) as ref:
        with rio.open(input_file) as src:       
            kwargs = src.meta.copy()     
            kwargs.update({
                'crs': ref.crs,
                'transform': ref.transform,
                'width': ref.width,
                'height': ref.height})
        
            with rio.open(output_file, 'w', **kwargs) as dst:
                for ii in range(1, src.count + 1):
                    reproject(
                        source=rio.band(src, ii),
                        destination=rio.band(dst, ii),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=ref.transform,
                        dst_crs=ref.crs,
                        resampling=Resampling.nearest
                    )
