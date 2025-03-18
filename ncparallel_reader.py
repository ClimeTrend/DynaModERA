import h5py
import numpy as np
import pyLOM
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import xarray as xr

from mpi4py import MPI

MPI_COMM = MPI.COMM_WORLD
MPI_RANK = MPI_COMM.Get_rank()
MPI_SIZE = MPI_COMM.Get_size()

pyLOM.style_plots()

def array_to_dataarray(
        array,
        lat,
        lon,
        time,
        level,
        name="",
        units="",
):
    """
    Convert numpy array to xarray DataArray.
    """
    if array.ndim != 2:
        raise ValueError("Array must be 2D.")
    if len(lat) != array.shape[0] or len(lon) != array.shape[0]:
        raise ValueError(
            "Latitude and longitude must have the same length as the number of array rows."
        )
    if len(time) != array.shape[1]:
        raise ValueError(
            "Time must have the same length as the number of array columns."
        )

    unique_lat = np.unique(lat)
    unique_lon = np.unique(lon)

    lat_indices = np.searchsorted(unique_lat, lat)  # Find indices of latitudes
    lon_indices = np.searchsorted(unique_lon, lon)  # Find indices of longitudes

    reshaped_array = np.empty((len(unique_lat), len(unique_lon), len(time)))

    for i in range(array.shape[0]):
        reshaped_array[lat_indices[i], lon_indices[i], :] = array[i, :]

    reshaped_array = reshaped_array.transpose(2, 0, 1)  # Transpose to time, lat, lon

    return xr.DataArray(
        data=reshaped_array,
        dims=["time", "latitude", "longitude"],
        coords={
            "time": time,
            "latitude": unique_lat,
            "longitude": unique_lon,
            "level": level,
        },
        attrs={"units": units, "name": name},
    )


def multi_dimension_mapping(ID:int, dims:np.ndarray, start=True):
    '''
    Function that converts an ID it to ndimensional coordinates

    Args:
        ID (int): 1D position to convert
        dims (np.ndarray): ndarray of the sizes of each dimension. Its shape is (ndims,)

    Returns:
        coordinates in the dims space
    '''
    ndims = len(dims)
    frac  = -1*np.ones((ndims,), dtype=int)
    if start: 
        for idim in range(ndims-1):
            prod = np.prod(dims[idim+1:])
            frac[idim] = int(np.floor(ID/prod))
            ID = ID%prod
        frac[-1] = ID
        return frac
    else:
        for idim in range(ndims-1):
            prod = np.prod(dims[idim+1:])
            div  = int(np.floor(ID/prod))
            frac[idim] = div #if div < dims[idim] and div != 0 else dims[idim]
            ID = ID%prod
        frac[-1] = ID #if ID < dims[-1] and ID != 0 else dims[-1]
        return frac

## Inputs for the parser
var   = 'temperature'
fname = 'data/era5_download/2019-01-01T00_2019-01-01T12_1h.nc'
plotL = 0

## Read file
file  = h5py.File(fname,'r',driver='mpio',comm=MPI_COMM)
time  = np.array(file['time'], dtype=np.float32)
lev   = np.array(file['level'], dtype=np.float32)
lat   = np.array(file['latitude'], dtype=np.float32)
lon   = np.array(file['longitude'], dtype=np.float32)
mesh  = np.meshgrid(lon,lat)

nlev, nlat, nlon = lev.shape[0], lat.shape[0], lon.shape[0]
npts   = nlev*nlat*nlon
ntime  = len(time)
points = np.array([nlev, nlat, nlon])

start, end = pyLOM.utils.worksplit(0, npts, MPI_RANK, nWorkers=MPI_SIZE)
start3D    = multi_dimension_mapping(start, points, start=True)
end3D      = multi_dimension_mapping(end, points, start=False)
pyLOM.pprint(-1, start3D, end3D, (end3D[0]-start3D[0])*nlat*nlon + (end3D[1]-start3D[1])*nlon + (end3D[2]-start3D[2]), flush=True)
mynptsG    = (end3D[0]-start3D[0])*nlat*nlon
var        = np.array(file[var][:,start3D[0]:end3D[0],:,:], dtype=np.float32).reshape(ntime,mynptsG).T
startslice = start3D[1]*nlon+start3D[2]
endslice   = (end3D[1])*nlon+(nlon-end3D[2])
var        = var[startslice:endslice,:]

## Gather all parts of the snapshot matrix for plotting
pyLOM.pprint(-1, var.shape, flush=True)
varG = pyLOM.utils.mpi_gather(var,0,all=True)
varG = varG.reshape(nlat*nlon, ntime)
if pyLOM.utils.is_rank_or_serial(0):
    fig, axs = plt.subplots(1, 1, figsize=(30, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    level2plot = array_to_dataarray(varG, mesh[1].flatten(), mesh[0].flatten(), time, 0)
    # First subplot
    level2plot.sel(time=time[0]).plot.contourf(ax=axs, transform=ccrs.PlateCarree(), cmap='jet', levels=range(230, 320, 1))
    axs.coastlines(color='black')
    axs.gridlines(draw_labels=True)

    plt.savefig('validation_figure_%i.png' % MPI_SIZE)

pyLOM.cr_info()