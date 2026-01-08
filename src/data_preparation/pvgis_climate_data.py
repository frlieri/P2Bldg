"""
src/data_preparation/pvgis_climate_data.py

Helpers to fetch, cache and process PVGIS photovoltaic and meteorological data
for a given geographic location and PV plane configuration.

Main responsibilities:

- get_pvgis_hourly_data(...):
    - Download or load cached hourly PV generation and meteorological data
      from the EU-JRC PVGIS service.
    - Return a :class:`Timeseries` with PV power converted to kW and columns
      defined in `PvgisDataCols`.
    - Supports custom year ranges, plane tilt/azimuth, horizon shading and
      caching to disk.

- get_pvgis_tmy_data(...):
    - Fetch or load cached PVGIS Typical Meteorological Year (TMY) data and
      return it as a :class:`Timeseries`.
    - Provides common TMY columns referenced by `PvgisTmyDataCols`.

- get_pvgis_monthly_data(...):
    - Query PVGIS monthly summaries via HTTP, produce simple aggregations
      (average temperature, yearly irradiation) and call plotting helpers.

Notes and behavior:

- Depends on `pvlib.iotools.pvgis`, `pandas`, `requests`, project `Timeseries`
  and project constants (e.g. `const.LATITUDE`, `const.LONGITUDE`, `PATH_TO_WD`).
- Functions cache CSV files in the specified `pvgis_folder`; initial calls
  require network access.
- PV power returned by PVGIS (W) is converted to kW before constructing the
  `Timeseries`.
- Column name enums `PvgisDataCols` and `PvgisTmyDataCols` expose the expected
  output fields for downstream processing.
- Some helpers produce plots or write files as side effects; callers in
  headless or restricted environments may need to disable plotting or adjust
  file paths.
"""

import os
from enum import Enum

import pandas as pd
import requests
import pvlib.iotools.pvgis as pvgis

from src import const

from src.config import PATH_TO_WD
from src.data_preparation.data_formats import Timeseries
from src.export.analysis_plots import scatterplot


class PvgisDataCols(str, Enum):
    # Time = "time"
    P_el_t = "P"
    Direct_rad = "poa_direct"
    Diffuse_rad_sky = "poa_sky_diffuse"
    Diffuse_rad_ground = "poa_ground_diffuse"
    Solar_elevation = "solar_elevation"
    Air_temp = "temp_air"
    Wind_speed = "wind_speed"
    # Int = "Int" # unknown


class PvgisTmyDataCols(str, Enum):
    # T2m [°C] - Dry bulb (air) temperature.
    # RH [%] - Relative Humidity.
    # G(h) [W/m2] - Global horizontal irradiance.
    # Gb(n) [W/m2] - Direct (beam) irradiance.
    # Gd(h) [W/m2] - Diffuse horizontal irradiance.
    # IR(h) [W/m2] - Infrared radiation downwards.
    # WS10m [m/s] - Windspeed.
    # WD10m [°] - Wind direction.
    # SP [Pa] - Surface (air) pressure.
    Air_temp = 'T2m'
    Humidity = 'RH'
    GHI = 'G(h)'
    Direct_rad = 'Gb(n)'
    Diffuse_rad = 'Gd(h)'
    Infrared = 'IR(h)'
    Windspeed = 'WS10m'
    Winddir = 'WD10m'
    AirPressure = 'SP'


def get_pvgis_hourly_data(latitude=const.LATITUDE, longitude=const.LONGITUDE, surface_tilt=35, surface_azimuth=0,
                          peak_power_kw=1, pvgis_folder=PATH_TO_WD+"/data/resources/PVGIS_hourly_data/",
                          start_year:int=None, end_year:int=None, horizon:str = None,
                          ) -> Timeseries:
    """
        Retrieve hourly PV generation and meteorological time series from the EU-JRC PVGIS service.

        The function uses `pvlib.iotools.pvgis.get_pvgis_hourly` to download an hourly profile for a given
        location and PV plane configuration. Results are cached on disk: if a matching CSV file already
        exists in `pvgis_folder`, it is loaded instead of calling the PVGIS API again.

        **What you get**
        - A :class:`~core.data_preparation.data_formats.Timeseries` instance containing hourly data for the
          selected period (`start_year`..`end_year`, inclusive).
        - The returned table includes the columns defined by :class:`PvgisDataCols` (e.g. PV power and
          weather/radiation quantities). PV power is converted from **W** (PVGIS default) to **kW**
          before building the Timeseries object.

        **Caching / file naming**
        The cache key is derived from latitude/longitude, tilt, azimuth, horizon setting, peak power and
        the requested year range. Changing any of these inputs results in a different CSV file name.

        :param latitude: Latitude of the location in decimal degrees (-90..90), north is positive.
        :type latitude: float
        :param longitude: Longitude of the location in decimal degrees (-180..180), east is positive.
        :type longitude: float
        :param surface_tilt: Tilt angle of the fixed plane in degrees (0 = horizontal, 90 = vertical).
        :type surface_tilt: int | float
        :param surface_azimuth: Azimuth/orientation of the plane in degrees.
            Convention used by PVGIS/pvlib: 0 = south, 90 = west, -90 = east.
        :type surface_azimuth: int | float
        :param peak_power_kw: Installed PV capacity in kWp used for the PVGIS PV calculation.
        :type peak_power_kw: int | float
        :param pvgis_folder: Folder used to read/write cached PVGIS hourly CSV files.
        :type pvgis_folder: str
        :param start_year: First year of the requested time range. Must be provided together with `end_year`.
            If both are omitted, defaults to `const.MODELED_YEAR`.
        :type start_year: int | None
        :param end_year: Last year of the requested time range. Must be provided together with `start_year`.
            If both are omitted, defaults to `const.MODELED_YEAR`.
        :type end_year: int | None
        :param horizon: Optional user-defined horizon profile to account for shading by surrounding terrain/objects.
            Provide as a comma-separated string of elevation angles (degrees) around the horizon, as expected by PVGIS
            (e.g. ``"0,0,0,0,50,0,0,0"``). If ``None``, PVGIS' default horizon handling is used.
        :type horizon: str | None
        :return: Hourly PVGIS time series (PV power and selected weather/radiation variables).
        :rtype: Timeseries
        """

    if start_year is not None or end_year is not None:
        assert start_year is not None and end_year is not None, "start_year and end_year always need to be given both"
    else:
        start_year = const.MODELED_YEAR
        end_year = const.MODELED_YEAR

    hourlydata_fname = (f"Timeseries_{latitude:.3f}_{longitude:.3f}_{surface_tilt:d}deg_{surface_azimuth:d}"
                        f"_{horizon.replace(',','') + '_' if horizon else '_'}deg_{peak_power_kw:d}kWp_"
                        f"{start_year:d}_{end_year:d}.csv")
    fpath = pvgis_folder + hourlydata_fname
    if os.path.exists(fpath):
        print("getting stored PVGIS hourly data...")
        df = pd.read_csv(fpath, sep=',', decimal='.')

    else:
        print("getting hourly data from PVGIS API...")
        query_args = dict(
            latitude=latitude,
            longitude=longitude,
            surface_tilt=surface_tilt,
            surface_azimuth=surface_azimuth,
            peakpower=peak_power_kw,
            loss=14,
            pvcalculation=True,
            outputformat="csv",
            start=start_year,
            end=end_year,
            components=1,
            usehorizon=1 if horizon is not None else 0,
            userhorizon=[float(e) for e in horizon.split(',')] if horizon is not None else None,
            # url="https://re.jrc.ec.europa.eu/api/v5_3/"
        )
        df, meta = pvgis.get_pvgis_hourly(**query_args)
        df.to_csv(fpath, sep=',', decimal='.')

    # Deprecated: time column is not used anymore
    # df[PVDataCols.Time.value] = pd.to_datetime(df.index)

    # P in kW
    df[PvgisDataCols.P_el_t] = df[PvgisDataCols.P_el_t] / 1000
    # return Timeseries(df, freq='1h')
    ts = Timeseries(df[[e.value for e in PvgisDataCols]], start_year=start_year, end_year=end_year)

    # plot_seasonal_boxplots(ts.df, PvgisDataCols.Air_temp)
    # ts.plot_typeday(PvgisDataCols.P_el_t, title=f"tilt: {surface_tilt}, azimuth: {surface_azimuth}")

    return ts


def get_pvgis_tmy_data(latitude=const.LATITUDE, longitude=const.LONGITUDE,
                       pvgis_folder=PATH_TO_WD+"/data/resources/PVGIS_hourly_data/"):
    """
    Fetches PVGIS Typical Meteorological Year (TMY) data based on the provided latitude 
    and longitude. If the data already exists in the specified folder, it loads the 
    stored CSV file. Otherwise, it fetches the data from PVGIS, saves it to the CSV file, 
    and then loads it. 

    This function helps in assessing renewable energy potential by providing solar 
    radiation and meteorological data for simulations.

    :param latitude: The latitude of the location for which TMY data is requested.
    :type latitude: float
    :param longitude: The longitude of the location for which TMY data is requested.
    :type longitude: float
    :param pvgis_folder: The folder to check or store the resulting TMY data file.
    :type pvgis_folder: str
    :return: Timeseries object containing TMY data.
    :rtype: Timeseries
    """

    fname = f"TMY_{const.LATITUDE:.3f}_{const.LONGITUDE:.3f}.csv"
    fpath = pvgis_folder + fname
    if os.path.exists(fpath):
        print("getting stored PVGIS TMY data...")
        tmy_data = pd.read_csv(fpath, sep=',', decimal='.')
    else:
        tmy_data, months_selected, inputs, metadata = pvgis.get_pvgis_tmy(latitude, longitude)
        tmy_data.to_csv(fpath, sep=',', decimal='.')

    # # hourly data for comparison
    # hourly_data = get_pvgis_hourly_data(surface_tilt=0, surface_azimuth=0)
    # compare_typical_day_quantiles([
    #     (hourly_data[PVDataCols.Direct_rad] + hourly_data[PVDataCols.Diffuse_rad_sky]).astype(float),
    #     tmy_data['G(h)']], subplot_titles=(str(START_YEAR), 'TMY'),
    #     title='Solar radiation'
    # )
    # compare_typical_day_quantiles([
    #     hourly_data[PVDataCols.Air_temp].astype(float),
    #     tmy_data[TMYDataCols.Air_temp]], subplot_titles=(str(START_YEAR), 'TMY'),
    #     title='Air temperature'
    # )

    return Timeseries(tmy_data)


def get_pvgis_monthly_data(latitude=const.LATITUDE, longitude=const.LONGITUDE):
    """
    Fetches and processes PVGIS monthly data for a specified latitude and longitude.

    This function retrieves photovoltaic data such as average temperature and
    global irradiation for a specified location using the PVGIS API. It processes
    the data to calculate average temperature and yearly global irradiation, then
    creates a scatter plot to visualize these relationships. 

    :param latitude: Latitude of the location for which PVGIS data is retrieved.
    :type latitude: float
    :param longitude: Longitude of the location for which PVGIS data is retrieved.
    :type longitude: float
    :return: None
    """
    "https://re.jrc.ec.europa.eu/api/MRcalc?lat=45&lon=8&horirrad=1"

    api_url = "https://re.jrc.ec.europa.eu/api/v5_3/MRcalc"
    query = f"?lat={const.LATITUDE}&lon={const.LONGITUDE}&horirrad=1&outputformat=json&global=1&avtemp=1"

    ret_val = requests.get(api_url + query).json()
    monthly_df = pd.DataFrame(ret_val['outputs']['monthly'])

    avg_temp = monthly_df.groupby('year')['T2m'].mean()
    yearly_glob_irr = monthly_df.groupby('year')['H(h)_m'].sum()

    scatterplot(pd.DataFrame(
        {
            'Average temperature': avg_temp,
            'Yearly global irradiation': yearly_glob_irr,
         }
        ), y1=['Average temperature'], y2=['Yearly global irradiation']
    )

    # print("Average temperature:\n", avg_temp, "\nYearly global irradiation:\n", yearly_glob_irr)



if __name__ == "__main__":

    climate_ts = get_pvgis_hourly_data(horizon="0,0,0,0,50,0,0,0")
    # climate_ts['HDD'] = climate_ts.df.apply(lambda x: max(20 - x[PvgisDataCols.Air_temp],0), axis=1)
    # climate_ts['year_week'] = climate_ts.df.apply(lambda x: f"{x[TimeseriesCols.Year]}_{x[TimeseriesCols.Week]}", axis=1)
    # # df = get_pvgis_tmy_data()
    # get_pvgis_monthly_data()

