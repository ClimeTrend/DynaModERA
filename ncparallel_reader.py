import h5py
import numpy as np
import pyLOM

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
            frac[idim] = div if div < dims[idim] and div != 0 else dims[idim]
            ID = ID%prod
        frac[-1] = ID if ID < dims[-1] and ID != 0 else dims[-1]
        return frac

## Inputs for the parser
var   = 'temperature'
fname = 'data/era5_download/2019-01-01T00_2019-01-01T12_1h.nc'

## Read file
file  = h5py.File(fname,'r')
time  = np.array(file['time'], dtype=np.float32)
lev   = np.array(file['level'], dtype=np.float32)
lat   = np.array(file['latitude'], dtype=np.float32)
lon   = np.array(file['longitude'], dtype=np.float32)

nlev, nlat, nlon = lev.shape[0], lat.shape[0], lon.shape[0]
npts   = nlev*nlat*nlon
points = np.array([nlev, nlat, nlon])
print(points)

start, end = pyLOM.utils.worksplit(0, npts, 0, nWorkers=1)
start3D    = multi_dimension_mapping(start, points, start=True)
end3D      = multi_dimension_mapping(end, points, start=False)
print(start3D, end3D)
var        = np.array(file[var][:,start3D[0]:end3D[0],start3D[1]:end3D[1],start3D[2]:end3D[2]], dtype=np.float32)
print(var[0,:,:,:].shape)
