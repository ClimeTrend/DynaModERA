import logging
import sys
from datetime import datetime, timedelta

import numpy as np
import xarray as xr
from numpy.lib.stride_tricks import sliding_window_view

from dmd_era5 import log_and_print, setup_logger

# Set up logger
logger = setup_logger("ERA5Processing", "era5_processing.log")
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(console_handler)


def slice_era5_dataset(
    ds: xr.Dataset,
    start_datetime: datetime | str | None = None,
    end_datetime: datetime | str | None = None,
    levels: list | None = None,
) -> xr.Dataset:
    """
    Slice an ERA5 dataset by time range and pressure levels.

    Parameters
    ----------
    ds : xr.Dataset
        The input ERA5 dataset to slice.
    start_datetime : datetime or str, optional
        The start datetime for slicing. If str, must be in isoformat
        (e.g. "2020-01-01T06"). If None, uses first datetime in dataset.
    end_datetime : datetime or str, optional
        The end datetime for slicing. If str, must be in isoformat
        (e.g. "2020-01-05"). If None, uses last datetime in dataset.
    levels : list of int, optional
        The pressure levels to select. If None, selects all levels.

    Returns
    -------
    xr.Dataset
        The sliced ERA5 dataset.

    Raises
    ------
    ValueError
        If requested time range is outside dataset bounds or levels not found.
    """

    # Convert string datetimes to datetime objects if needed
    start_dt = (
        datetime.fromisoformat(start_datetime)
        if isinstance(start_datetime, str)
        else start_datetime
    )
    end_dt = (
        datetime.fromisoformat(end_datetime)
        if isinstance(end_datetime, str)
        else end_datetime
    )

    # Get dataset time bounds
    time_bounds = _get_dataset_time_bounds(ds)

    # Use dataset bounds if no times specified
    start_dt = start_dt or time_bounds["first"]
    end_dt = end_dt or time_bounds["last"]

    # Validate time range is within dataset bounds
    if start_dt < time_bounds["first"] or end_dt > time_bounds["last"]:
        msg = f"Time range ({start_dt} to {end_dt}) is outside dataset"
        msg += f"bounds ({time_bounds['first']} to {time_bounds['last']})."
        log_and_print(logger, msg, "error")
        raise ValueError(msg)

    # Validate the start is before the end datetime
    if start_dt >= end_dt:
        msg = "Start datetime must be before end datetime."
        log_and_print(logger, msg, "error")
        raise ValueError(msg)

    # Use all levels if none specified
    levels = levels or list(ds.level.values)

    # Slice the dataset
    try:
        sliced_ds = ds.sel(time=slice(start_dt, end_dt), level=levels)
        log_and_print(
            logger,
            f"Dataset slicing completed successfully using {start_dt}"
            f"to {end_dt} and levels {levels}",
        )
        return sliced_ds

    except KeyError as e:
        available_levels = list(ds.level.values)
        msg = "Requested level is not available in the dataset."
        msg += f"Available levels: {available_levels}"
        log_and_print(logger, msg, "error")
        raise ValueError(msg) from e


def _get_dataset_time_bounds(ds: xr.Dataset) -> dict:
    """
    Get the first and last timestamps from an ERA5 dataset.

    Parameters
    ----------
    ds : xr.Dataset
        The ERA5 dataset.

    Returns
    -------
    dict
        Dictionary with 'first' and 'last' datetime objects.
    """
    return {
        "first": datetime.fromtimestamp(ds.time.values[0].astype(int) * 1e-9),
        "last": datetime.fromtimestamp(ds.time.values[-1].astype(int) * 1e-9),
    }


def resample_era5_dataset(ds: xr.Dataset, delta_time: timedelta) -> xr.Dataset:
    """
    Resample an ERA5 dataset along the time dimension by a
    specified time delta, using nearest neighbor.

    Args:
        ds (xr.Dataset): The input ERA5 dataset.
        delta_time (timedelta): The time delta for resampling.

    Returns:
        xr.Dataset: The resampled ERA5 dataset.
    """

    resampled_ds = ds.resample(time=delta_time).nearest()
    log_and_print(logger, f"Resampled the dataset with time delta: {delta_time}")
    return resampled_ds


def standardize_data(
    data: xr.DataArray,
    dim: str = "time",
    scale: bool = True,
) -> xr.DataArray:
    """
    Standardize the input DataArray by applying mean centering and (optionally)
    scaling to unit variance along the specified dimension.

    Args:
        data (xr.DataArray): The input data to standardize.
        dim (str): The dimension along which to standardize. Default is "time".
        scale (bool): Whether to scale the data. Default is True.

    Returns:
        xr.DataArray: The standardized data.
    """
    log_and_print(logger, f"Standardizing data along {dim} dimension...")

    # Mean center the data
    log_and_print(
        logger,
        f"Removing mean along {dim} dimension...",
    )
    data = data - data.mean(dim=dim)
    if scale:
        # Scale the data by the standard deviation
        log_and_print(logger, f"Scaling to unit variance along {dim} dimension...")
        data = data / data.std(dim=dim)
    return data


def apply_delay_embedding(X, d):
    """
    Apply delay embedding to temporal snapshots.

    Parameters
    ----------
    X : np.ndarray
        The input data array of shape (n_space * n_variables, n_time).
    d : int
        The number of snapshots from X to include in each snapshot of the output.

    Returns
    -------
    np.ndarray
        The delay-embedded data array of shape
        (n_space * n_variables * d, n_time - d + 1).
    """

    if X.ndim != 2:
        msg = "Input array must be 2D."
        raise ValueError(msg)
    if not isinstance(d, int) or d <= 0:
        msg = "Delay must be an integer greater than 0."
        raise ValueError(msg)

    return (
        sliding_window_view(X.T, (d, X.shape[0]))[:, 0]
        .reshape(X.shape[1] - d + 1, -1)
        .T
    )


def flatten_era5_variables(era5_ds: xr.Dataset) -> tuple[np.ndarray, dict, list]:
    """
    Flatten the variables in an ERA5 dataset to a single two-dimensional NumPy array.

    Parameters
    ----------
    era5_ds : xr.Dataset
        The input ERA5 dataset.

    Returns
    -------
    np.ndarray
        The flattened array of variables, with shape (n_space * n_variables, n_time),
        where n_space = n_level * n_lat * n_lon. Variables are concatenated along the
        first dimension (space).
    dict
        A dictionary containing the coordinates of the flattened array, with keys
        "level", "latitude", "longitude", and "time". "level", "latitude", and
        "longitude" are 1D arrays with n_space elements, and "time" is a 1D array
        with n_time elements.
    list
        A list of length n_variables, containing the names of the variables
        in the order they appear in the flattened array.
    """

    variables: list[str] = list(map(str, era5_ds.data_vars.keys()))
    coords: list[str] = list(map(str, era5_ds.coords.keys()))
    must_have_coords = ["latitude", "longitude", "time", "level"]
    spatial_stack_order = ["level", "latitude", "longitude"]

    if sorted(coords) != sorted(must_have_coords):
        msg = f"Missing required coordinates: {must_have_coords}."
        raise ValueError(msg)

    # stack the spatial dimensions
    stacked = era5_ds.stack(space=spatial_stack_order)

    # create a list of variable arrays with shape (n_space, n_time)
    data_list = [stacked[var].transpose("space", "time").values for var in variables]
    # concatenate the variable arrays along the space dimension
    data_combined = np.concatenate(data_list, axis=0)

    # create a dictionary of the coordinates of the flattened array
    flattened_coords = dict.fromkeys([*spatial_stack_order, "time"])
    for coord in flattened_coords:
        flattened_coords[coord] = stacked[coord].values

    return data_combined, flattened_coords, variables
