from pathlib import Path
from typing import Optional, Union

import geopandas as gpd
import seaborn as sns
import matplotlib.pyplot as plt

from clustering.utils import get_cluster_pivot_gdf


def plot_weights_vs_radii(
    cluster_df: Optional[gpd.GeoDataFrame] = None,
    point_gdf_w_cluster: Optional[gpd.GeoDataFrame] = None,
    point_gdf_cluster_id_col: str = "cluster_id",
    point_weight_col: str = "weight",
    point_projected_epsg: int = 26191,
    y_human_readable: str = "Rooftops",
    output_filepath: Optional[Union[Path, str]] = None,
) -> sns.JointGrid:
    """
    Plots the number of grids per cluster and the cluster radius.

    Parameters
    ----------
    cluster_df : pivoted dataframe containing cluster geometries and cluster_weights
        (as per get_cluster_pivot_gdf).
        Either provide this or the unpivoted point_df_w_cluster
    point_gdf_w_cluster : geodataframe containing point geometries and weights.
        Either provide this or the pivoted cluster_df.
    point_gdf_cluster_id_col : needed if point_gdf_w_cluster is given. Name of the column
        containing the cluster IDs for grouping.
    point_weight_col : needed if point_gdf_w_cluster is given. Name of the column
        containing the weights each point (e.g. population).
    point_projected_epsg : needed if point_gdf_w_cluster is given. EPSG code for
        the projected coordinate reference system to use for calculating the radius of
        the clusters. Default is 26191 (morocco).
    y_human_readable : human-readable name of the weight column. Default is "Rooftops".
    output_filepath : path to save the plot. If None, simply returns the plot.

    Returns
    -------
    sns.JointGrid : seaborn jointgrid object containing the plot.
    """

    # check if cluster_pivot_gdf or df_w_clusters is provided
    if cluster_df is None and point_gdf_w_cluster is None:
        raise ValueError("Either cluster_df or point_gdf_w_cluster must be provided.")

    if cluster_df is None:
        cluster_df = get_cluster_pivot_gdf(
            gdf_w_clusters=point_gdf_w_cluster,
            cluster_id_col=point_gdf_cluster_id_col,
            weight_col=point_weight_col,
            with_stats=True,
            epsg=point_projected_epsg,
        )

    # plot
    g = sns.JointGrid()
    x = cluster_df["minimum_bounding_radius"]
    y = cluster_df["cluster_weight"]

    # scale alpha and hist linewidth by number of clusters
    n = len(cluster_df)
    # alpha range: 1 for small n and 0.1 for large n
    alpha = min(0.9 / n * 50, 0.9) + 0.1
    # linewidth range: 1 for small n and 0.01 for large n
    linewidth = min(0.99 / n * 50, 0.99) + 0.01

    g.ax_joint.scatter(x, y, alpha=alpha)
    sns.histplot(x=x, linewidth=linewidth, ax=g.ax_marg_x)
    sns.histplot(y=y, linewidth=linewidth, ax=g.ax_marg_y)
    g.ax_joint.set_xlabel("Cluster radius (m)")
    g.ax_joint.set_ylabel(f"Cluster weight ({y_human_readable})")

    plt.tight_layout()

    if output_filepath is not None:
        plt.savefig(output_filepath, dpi=300)

    return g


def save_shapefiles(
    gdf: gpd.GeoDataFrame,
    folderpath: Path,
    filename: str = "processed_grids",
    formats: list[str] = ["parquet", "kml", "geojson", "csv"],
) -> None:
    """
    Save a GeoDataFrame to multiple file formats.

    Parameters
    ----------
    gdf : A GeoDataFrame.
    folderpath : The directory in which to save the files.
    filename : The name of the file to save, without extensions.
    formats : A list of file formats to save the GeoDataFrame to. Must be a subset of
        ["parquet", "kml", "geojson", "csv"].

    Returns
    -------
    None
    """

    # raise error if disallowed formats are passed
    allowed_formats = ["parquet", "kml", "geojson", "csv"]
    for format in formats:
        if format not in allowed_formats:
            raise ValueError(
                f"{format} not allowed. formats must be a list containing any of the "
                f"following: {allowed_formats}"
            )

    # create output folder if it doesn't exist
    folderpath.mkdir(parents=True, exist_ok=True)

    # with geometries
    if "parquet" in formats:
        gdf.to_parquet(folderpath / f"{filename}.parquet")

    if "kml" in formats:
        gdf.to_file(folderpath / f"{filename}.kml", driver="KML")

    if "geojson" in formats:
        gdf.to_file(folderpath / f"{filename}.geojson", driver="GeoJSON")

    # without geometries
    if "csv" in formats:
        gdf.drop(columns=["geometry"]).to_csv(folderpath / f"{filename}.csv")


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
