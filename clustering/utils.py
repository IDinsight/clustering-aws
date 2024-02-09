import geopandas as gpd


def df_w_latlons_to_gdf(
    df,
    lat_name: str = "Lat",
    lon_name: str = "Lon",
    crs: str = "EPSG:4326",
):
    """
    Convert DataFrame to GeoDataFrame by creating points from lat-lon columns.

    Parameters
    ----------
    df : The DataFrame to convert.
    lat_name, lon_name : The names of the columns containing the latitude and longitude values.
        Default is 'Lat' and 'Lon'.
    crs : The coordinate reference system of the lat-lon columns.
    """
    latlon_point_geoms = gpd.points_from_xy(x=df[lon_name], y=df[lat_name])
    gdf = gpd.GeoDataFrame(df.copy(), geometry=latlon_point_geoms, crs=crs)
    return gdf


def create_ids(df_length: int, prefix: str = "GRID_") -> list[str]:
    """
    Create a list of string IDs in the format "{prefix}001".

    Parameters
    ----------
    df : dataframe containing grids

    Returns
    -------
    list : list of string_id_list.
    """

    # make list of ids [1, 2, 3, ...]
    ids = list(range(1, df_length + 1))

    # make list of ids with leading zeros ["001", "002", "003", ...]
    max_digits = len(str(df_length))
    string_ids = [prefix + str(id).zfill(max_digits) for id in ids]

    return string_ids
