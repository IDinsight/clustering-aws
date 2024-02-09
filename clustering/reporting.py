from typing import Optional

import geopandas as gpd


def get_cluster_pivot_gdf(
    gdf_w_clusters: gpd.GeoDataFrame,
    weight_col: str,
    cols_to_keep: Optional[list[str]] = None,
    with_stats: bool = True,
    epsg: int = 26191,  # for morocco
):
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
