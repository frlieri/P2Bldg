# PVGIS API - hourly data

- hourly profiles for PV generation for a given location and area orientation / tilt are retrieved using the 
[PVGIS API](https://joint-research-centre.ec.europa.eu/pvgis-online-tool/getting-started-pvgis/api-non-interactive-service_en)
using the pvlib python package
- the retrieved profiles are stored in this folder for later usage
- Naming convention for the stored profiles is 
``Timeseries_*latitude*_*longitude*_*surface_tilt*deg_*surface_azimuth*_*horizonprofile*deg_*peak_power_kw*kWp*start_year*_*end_year*.csv``