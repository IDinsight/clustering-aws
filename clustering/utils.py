import math
import os
from typing import Tuple
import geopandas as gpd


def pivot_by_cluster(
    gdf_w_clusters: gpd.GeoDataFrame,
    cluster_id_col: str,
    weight_col: str,
    cols_to_keep: list[str] = [],
) -> gpd.GeoDataFrame:
    """
    Returns a pivot table of the gdf_w_clusters with the cluster_id.

    Parameters
    ----------
    gdf_w_clusters : dataframe containing GPS coordinates, point IDs and weights.
    cluster_id_col : name of the column containing the cluster IDs for grouping
    weight_col : name of the column containing the weights each point
        (e.g. population).
    cols_to_keep : list of other columns to keep in the pivot table.

    Returns
    -------
    cluster_pivot_gdf : pivoted dataframe with cluters as rows and aggregated
        cluster_weight, n_points, convex hull geometry and other optional columns.
    """

    original_crs = gdf_w_clusters.crs

    gdf_w_clusters.loc[:, "n_points"] = 1
    # groupby cluster_id and aggregate
    agg_func_dict = {
        weight_col: "sum",
        "n_points": "size",
        "geometry": lambda x: x.unary_union.convex_hull,
    }
    if len(cols_to_keep) != 0:
        agg_func_dict.update({col: "first" for col in cols_to_keep})
    cluster_pivot_gdf = gdf_w_clusters.groupby(cluster_id_col).agg(agg_func_dict)

    # process
    cluster_pivot_gdf = cluster_pivot_gdf.rename(columns={weight_col: "cluster_weight"})
    cluster_pivot_gdf = gpd.GeoDataFrame(
        cluster_pivot_gdf, geometry="geometry", crs=original_crs
    )

    cluster_pivot_gdf = cluster_pivot_gdf.reset_index()

    return cluster_pivot_gdf


def pivot_by_cluster_w_stats(
    gdf_w_clusters: gpd.GeoDataFrame,
    cluster_id_col: str,
    weight_col: str,
    epsg: int,
    cols_to_keep: list[str] = [],
) -> gpd.GeoDataFrame:
    """
    Adds additional statistics to the cluster_pivot_gdf.

    Parameters
    ----------
    gdf_w_clusters : dataframe containing GPS coordinates, point IDs and weights.
    cluster_id_col : name of the column containing the cluster IDs for grouping
    weight_col : name of the column containing the weights each point
        (e.g. population).
    cols_to_keep : list of other columns to keep in the pivot table.
    epsg : EPSG code for the projected coordinate reference system to use for
        calculating the radius of the clusters. Find the appropriate EPSG code
        for your region from https://epsg.io/.

    Returns
    -------
    cluster_pivot_gdf : geodataframe containing GPS coordinates, grid IDs, weights,
        and additional statistics, pivoted by cluster_id.
    """

    cluster_pivot_gdf = pivot_by_cluster(
        gdf_w_clusters=gdf_w_clusters,
        cluster_id_col=cluster_id_col,
        weight_col=weight_col,
        cols_to_keep=cols_to_keep,
    )

    # get latlon of cluster centroids
    cluster_pivot_gdf["Lat_cluster_centroid"] = cluster_pivot_gdf.geometry.apply(
        lambda row: row.centroid.y
    )
    cluster_pivot_gdf["Lon_cluster_centroid"] = cluster_pivot_gdf.geometry.apply(
        lambda row: row.centroid.x
    )
    # switch to projected CRS for radius and area
    cluster_pivot_gdf_projected = cluster_pivot_gdf.to_crs(epsg=epsg)
    cluster_pivot_gdf["minimum_bounding_radius"] = cluster_pivot_gdf_projected[
        "geometry"
    ].minimum_bounding_radius()
    cluster_pivot_gdf["area_km^2"] = (
        cluster_pivot_gdf_projected["geometry"].area / 10**6
    )

    return cluster_pivot_gdf


def create_ids(size: int, prefix: str) -> list[str]:
    """
    Create a list of string IDs in the format "ID_001".
    Note - the numbering goes from 1 to size, not 0 to size-1.

    Parameters
    ----------
    size : Number to make the IDs up to.
    prefix : String to use as prefix for the IDs

    Returns
    -------
    list : list of string ids.
    """

    # make list of ids [1, 2, 3, ...]
    ids = list(range(1, size + 1))

    # make list of ids with as many leading zeros as necessary
    # ["001", "002", "003", ...]
    max_digits = len(str(size))
    string_ids = [prefix + str(id).zfill(max_digits) for id in ids]

    return string_ids


def split_n_jobs(n_jobs: int) -> Tuple[int, int]:
    """Split n_jobs into two factors that are as close as possible to each other."""
    if n_jobs == 1:
        n_parallel_clusters = 1
        optuna_n_jobs = 1
    elif n_jobs == -1:
        n_jobs = os.cpu_count() or 1  # in case os.cpu_count() returns None

    # Split n_jobs into two factors that are as close as possible to each other.
    root = math.sqrt(n_jobs)
    factor = int(root)
    while n_jobs % factor != 0:
        factor -= 1
    optuna_n_jobs = factor  # the smaller number
    n_parallel_clusters = n_jobs // factor  # the larger number

    return n_parallel_clusters, optuna_n_jobs
