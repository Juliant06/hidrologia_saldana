from pysheds.grid import Grid
from pysheds.view import Raster, ViewFinder
import numpy as np
import xarray as xr

def delimitacion_cuenca(
        dem:xr.DataArray, 
        outlet:tuple, 
        name:str,
        save:bool
    ):

        
    data = dem.data
    affine = dem.rio.transform()
    crs = dem.rio.crs
    # Keep same data type as DEM values for nodata
    nodata = data.dtype.type(np.nan)

    viewfinder = ViewFinder(
        affine=affine, 
        shape=data.shape, 
        crs=crs,
        nodata=nodata
    )
    dem = Raster(data, viewfinder=viewfinder)

            # Create grid
    grid = Grid.from_raster(dem)

    # Detect number of pits in DEM
    pits = grid.detect_pits(dem)
    npits = np.count_nonzero(pits)

    # Fill pits and check they have been filled
    pit_filled_dem = grid.fill_pits(dem)
    pits = grid.detect_pits(pit_filled_dem)
    assert not pits.any()

    depressions = grid.detect_depressions(pit_filled_dem)

    # Fill depressions - Slow, takes 5-10 minutes...
    flooded_dem = grid.fill_depressions(pit_filled_dem)
    depressions = grid.detect_depressions(flooded_dem)
    assert not depressions.any()

    inflated_dem = grid.resolve_flats(
        flooded_dem, 
        max_iter=1e9, 
        eps=1e-12
    )

    # Compute flow direction based on corrected DEM (D8)
    fdir = grid.flowdir(inflated_dem)

    # Compute flow accumulation based on computed flow direction
    acc = grid.accumulation(fdir)

    # Snap outlet point to nearby cell with high flow accumulation
    x_snap, y_snap = grid.snap_to_mask(acc > 20000, outlet)

    # Delineate the watershed
    catch = grid.catchment(
        x=x_snap, 
        y=y_snap, 
        fdir=fdir,
        xytype="coordinate"
    )

    # catch = grid.catchment(x=x_out, y=y_out, fdir=fdir, xytype='coordinate')

    # 1. Recortar el grid a la extensión de la cuenca
    grid.clip_to(catch)

    # 2. Vista de la máscara de la cuenca (ya recortada)
    catch_view = grid.view(catch, nodata=0)


    # 3. Guardar como GeoTIFF 
    # (la máscara es booleana, conviene pasarla a uint8)
    grid.to_raster(
        catch_view.astype('uint8'),
        name,
        target_view=grid.viewfinder,
        blockxsize=256,
        blockysize=256,
        nodata=0,
    )

    return catch_view