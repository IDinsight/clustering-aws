import geopandas as gpd


def get_cluster_pivot_gdf(
    gdf_w_clusters: gpd.GeoDataFrame,
    cluster_id_col: str,
    weight_col: str,
    epsg: int,
    cols_to_keep: list[str] = [],
    with_stats: bool = True,
) -> gpd.GeoDataFrame:
    """
    Returns a pivot table of the gdf_w_clusters with the cluster_id.

    Parameters
    ----------
    gdf_w_clusters : dataframe containing GPS coordinates, point IDs and weights.
    cluster_id_col : name of the column containing the cluster IDs for grouping
    weight_col : name of the column containing the weights each point
        (e.g. population).
    epsg : EPSG code for the projected coordinate reference system to use for
        calculating the radius of the clusters. Find the appropriate EPSG code
        for your region from https://epsg.io/.
    cols_to_keep : list of other columns to keep in the pivot table.
    with_stats : whether to include additional statistics in the pivot table.
        Default is True.

    Returns
    -------
    cluster_pivot_gdf : geodataframe containing GPS coordinates, grid IDs and weights,
        pivoted by cluster_id.
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

    # add extra stats if required
    if with_stats:
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

        cluster_pivot_gdf = cluster_pivot_gdf[
            cols_to_keep
            + [
                "cluster_weight",
                "n_points",
                "Lat_cluster_centroid",
                "Lon_cluster_centroid",
                "minimum_bounding_radius",
                "area_km^2",
                "geometry",
            ]
        ]

    cluster_pivot_gdf = cluster_pivot_gdf.reset_index()

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
