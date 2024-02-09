from typing import Optional

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd


def get_cluster_pivot_gdf(
    gdf_w_clusters: gpd.GeoDataFrame,
    weight_col: str,
    cols_to_keep: Optional[list[str]] = None,
    with_stats: bool = True,
    epsg: int = 26191,  # for morocco
) -> pd.DataFrame:
    """
    Returns a pivot table of the gdf_w_clusters with the cluster_id.

    Parameters
    ----------
    gdf_w_clusters : dataframe containing GPS coordinates, point IDs and weights.
    weight_col : name of the column containing the weights each point
        (e.g. population).
    with_stats : whether to include additional statistics in the pivot table.
        Default is True.

    Returns
    -------
    cluster_pivot_gdf : dataframe containing GPS coordinates, grid IDs and weights,
        pivoted by cluster_id.
    """

    original_crs = gdf_w_clusters.crs

    # groupby cluster_id and aggregate
    agg_func_dict = {weight_col: "sum", "geometry": lambda x: x.unary_union.convex_hull}
    if cols_to_keep is not None:
        agg_func_dict.update({col: "first" for col in cols_to_keep})
    cluster_pivot_gdf = gdf_w_clusters.groupby("cluster_id").agg(agg_func_dict)

    # process
    cluster_pivot_gdf = cluster_pivot_gdf.rename(columns={weight_col: "cluster_weight"})
    cluster_pivot_gdf = gpd.GeoDataFrame(
        cluster_pivot_gdf, geometry="geometry", crs=original_crs
    )

    # add extra stats if required
    if with_stats:
        cluster_pivot_gdf["num_points"] = gdf_w_clusters.groupby("cluster_id").size()
        # get latlon of cluster centroids
        cluster_pivot_gdf["Lat_cluster_centroid"] = cluster_pivot_gdf.geometry.apply(
            lambda row: row.centroid.y
        )
        cluster_pivot_gdf["Lon_cluster_centroid"] = cluster_pivot_gdf.geometry.apply(
            lambda row: row.centroid.x
        )
        # switch to projected CRS for radius and area
        cluster_pivot_gdf = cluster_pivot_gdf.to_crs(epsg=epsg)
        cluster_pivot_gdf["minimum_bounding_radius"] = cluster_pivot_gdf[
            "geometry"
        ].minimum_bounding_radius()
        cluster_pivot_gdf["area_km^2"] = cluster_pivot_gdf["geometry"].area / 10**6

    return cluster_pivot_gdf


def plot_weights_vs_radii(
    cluster_pivot_gdf=None,
    df_w_clusters=None,
    weight_col="weight",
    epsg=26191,
    y_human_readable="Rooftops",
):
    """
    Plots the number of grids per cluster and the cluster radius.

    Parameters
    ----------
    cluster_pivot_gdf : pivoted dataframe containing cluster geometries and weights
        (as per get_cluster_pivot_gdf). Either provide this or the unpivoted gdf_w_clusters
    gdf_w_clusters : dataframe containing GPS coordinates, grid IDs and weights.
    weight_col : name of the column containing the weights each point
        (e.g. population).
    epsg : EPSG code for the coordinate reference system to use. Default is 26191 (morocco).
    y_human_readable : human-readable name of the weight column. Default is "Rooftops".

    Returns
    -------
    ax : matplotlib axis object.
    """

    if cluster_pivot_gdf is None:
        # stats that depend on cluster geometries need pivot
        cluster_pivot_gdf = get_cluster_pivot_gdf(
            gdf_w_clusters=df_w_clusters,
            weight_col=weight_col,
            with_stats=True,
            epsg=epsg,
        )

    # plot
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(
        cluster_pivot_gdf["minimum_bounding_radius"],
        cluster_pivot_gdf["cluster_weight"],
        alpha=0.5,
    )
    ax.set_xlabel("Cluster radius (m)")
    ax.set_ylabel(f"Cluster weight ({y_human_readable})")
    ax.set_title(f"Cluster weight ({y_human_readable}) vs. cluster radius")
    plt.tight_layout()
    return ax
