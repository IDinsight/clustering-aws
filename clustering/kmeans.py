from functools import partial
from multiprocessing import Pool
from typing import Optional, Union

import geopandas as gpd
import numpy as np
import optuna
import pandas as pd
from sklearn.cluster import KMeans

optuna.logging.set_verbosity(optuna.logging.WARNING)

from .reporting import get_cluster_pivot_gdf
from .utils import create_ids


# Use for local or EC2 instance
def parallel_kmeans_secondpass(
    gdf_w_clusters: gpd.GeoDataFrame,
    oversized_cluster_ids: list[str],
    desired_cluster_weight: Union[float, int],
    desired_cluster_radius: Union[float, int],
    id_col: str = "rooftop_id",
    lat_col: str = "Lat_centroid",
    lon_col: str = "Lon_centroid",
    weight_col: Optional[str] = None,
    weight_importance_factor: Union[float, int] = 1,
    epsg: int = 26191,  # for morocco
    n_trials: int = 100,
    python_n_jobs: int = 4,
    optuna_n_jobs: int = 4,
) -> gpd.GeoDataFrame:

    # make a copy of the gdf
    gdf_w_clusters_doublepass = gdf_w_clusters.copy()

    # rerun optimisation for each cluster that is oversized to split it up into smaller ones.
    # Note: this "partial" usage can be replace with a Class structure
    # (similar to KMeansObjective)
    run_optuna_kmeans_secondpass_w_args = partial(
        recluster,
        gdf_w_clusters=gdf_w_clusters,
        desired_cluster_weight=desired_cluster_weight,
        desired_cluster_radius=desired_cluster_radius,
        id_col=id_col,
        lat_col=lat_col,
        lon_col=lon_col,
        weight_col=weight_col,
        weight_importance_factor=weight_importance_factor,
        epsg=epsg,
        n_trials=n_trials,
        n_jobs=optuna_n_jobs,
    )

    # without progress bar - can add another time
    with Pool(processes=python_n_jobs) as pool:
        oversized_clusters = pool.map(
            run_optuna_kmeans_secondpass_w_args, oversized_cluster_ids
        )

    # replace the cluster_id of the oversized cluster with the new subclusters
    gdf_w_clusters_doublepass = gdf_w_clusters_doublepass[
        ~gdf_w_clusters_doublepass["cluster_id"].isin(oversized_cluster_ids)
    ]
    gdf_w_clusters_doublepass = pd.concat(
        [gdf_w_clusters_doublepass] + oversized_clusters
    )

    return gdf_w_clusters_doublepass


# Use for AWS Lambda
def kmeans_secondpass(
    gdf_w_clusters: gpd.GeoDataFrame,
    oversized_cluster_ids: list[str],
    desired_cluster_weight: Union[float, int],
    desired_cluster_radius: Union[float, int],
    id_col: str = "rooftop_id",
    lat_col: str = "Lat_centroid",
    lon_col: str = "Lon_centroid",
    weight_col: Optional[str] = None,
    weight_importance_factor: Union[float, int] = 1,
    epsg: int = 26191,  # for morocco
    n_trials: int = 100,
    n_jobs: int = -1,
) -> gpd.GeoDataFrame:

    # make a copy of the gdf
    gdf_w_clusters_doublepass = gdf_w_clusters.copy()

    # rerun optimisation for each cluster that is oversized to split it up into smaller ones.
    for cluster_id in oversized_cluster_ids:
        # subset data
        oversized_cluster = recluster(
            cluster_id=cluster_id,
            gdf_w_clusters=gdf_w_clusters,
            desired_cluster_weight=desired_cluster_weight,
            desired_cluster_radius=desired_cluster_radius,
            id_col=id_col,
            lat_col=lat_col,
            lon_col=lon_col,
            weight_col=weight_col,
            weight_importance_factor=weight_importance_factor,
            epsg=epsg,
            n_trials=n_trials,
            n_jobs=n_jobs,
        )

        # replace the cluster_id of the oversized cluster with the new subclusters
        gdf_w_clusters_doublepass = gdf_w_clusters_doublepass[
            gdf_w_clusters_doublepass["cluster_id"] != cluster_id
        ]
        gdf_w_clusters_doublepass = pd.concat(
            [gdf_w_clusters_doublepass, oversized_cluster]
        )

    return gdf_w_clusters_doublepass


def get_oversized_clusters(
    gdf_w_clusters: Union[gpd.GeoDataFrame, pd.DataFrame],
    cutoff_weight: Union[float, int],
) -> list[str]:

    unique_clusters_df = (
        gdf_w_clusters[["cluster_id", "cluster_weight"]]
        .drop_duplicates(subset="cluster_id")
        .copy()
    )
    oversized_clusters_df = unique_clusters_df[
        unique_clusters_df["cluster_weight"] > cutoff_weight
    ]
    oversized_cluster_ids = oversized_clusters_df["cluster_id"].tolist()

    return oversized_cluster_ids


def recluster(
    cluster_id: str,
    gdf_w_clusters: gpd.GeoDataFrame,
    desired_cluster_weight: Union[float, int],
    desired_cluster_radius: Union[float, int],
    id_col: str = "rooftop_id",
    lat_col: str = "Lat_centroid",
    lon_col: str = "Lon_centroid",
    weight_col: Optional[str] = None,
    weight_importance_factor: Union[float, int] = 1,
    epsg: int = 26191,  # for morocco
    n_trials: int = 100,
    n_jobs: int = -1,
) -> gpd.GeoDataFrame:

    # subset data
    oversized_cluster = gdf_w_clusters[gdf_w_clusters["cluster_id"] == cluster_id].drop(
        columns=["cluster_id", "cluster_weight"]
    )

    # run optimiser to find best n_clusters
    study = run_optuna_kmeans_study(
        gdf=oversized_cluster,
        desired_cluster_weight=desired_cluster_weight,
        desired_cluster_radius=desired_cluster_radius,
        id_col=id_col,
        lat_col=lat_col,
        lon_col=lon_col,
        weight_col=weight_col,
        weight_importance_factor=weight_importance_factor,
        epsg=epsg,
        n_trials=n_trials,
        n_jobs=n_jobs,
    )

    # run best kmeans
    subclusters = custom_kmeans(
        df=oversized_cluster,
        n_clusters=study.best_params["n_clusters"],
        id_col=id_col,
        lat_col=lat_col,
        lon_col=lon_col,
        weight_col=weight_col,
        cluster_id_prefix=f"{cluster_id}_",
    )
    oversized_cluster = oversized_cluster.merge(subclusters, on=id_col)

    return oversized_cluster


def run_optuna_kmeans_study(
    gdf: gpd.GeoDataFrame,
    desired_cluster_weight: Union[float, int],
    desired_cluster_radius: Union[float, int],
    id_col: str = "rooftop_id",
    lat_col: str = "Lat_centroid",
    lon_col: str = "Lon_centroid",
    weight_col: Optional[str] = None,
    weight_importance_factor: Union[float, int] = 1,
    epsg: int = 26191,  # for morocco
    n_trials: int = 100,
    n_jobs: int = -1,
    show_progress_bar: bool = False,
) -> optuna.Study:
    """Run optuna to find the optimal number of clusters for kmeans."""

    if weight_col is None:
        weight_col = "weight"
        gdf[weight_col] = 1
        total_weight = len(gdf)
    else:
        total_weight = gdf[weight_col].sum()

    expected_n_clusters = int(total_weight / desired_cluster_weight)
    n_samples = len(gdf)
    min_n_clusters, max_n_clusters, search_space = get_min_max_search_space(
        expected_n_clusters, n_samples
    )
    study = optuna.create_study(
        sampler=optuna.samplers.GridSampler(search_space, seed=42),
        direction="minimize",
        study_name="kmeans_optimisation",
    )
    kmeans_objective = KMeansObjective(
        gdf,
        min_n_clusters,
        max_n_clusters,
        id_col=id_col,
        lat_col=lat_col,
        lon_col=lon_col,
        weight_col=weight_col,
        epsg=epsg,
        weight_importance_factor=weight_importance_factor,
        target_weight=desired_cluster_weight,
        target_radius=desired_cluster_radius,
    )
    study.optimize(
        kmeans_objective,
        n_trials=n_trials,
        n_jobs=n_jobs,
        gc_after_trial=True,
        show_progress_bar=show_progress_bar,
    )
    return study


def get_min_max_search_space(
    n_clusters: int, n_samples: int, scale_factor: int = 2
) -> tuple[int, int, dict[str, list[int]]]:

    max_n_clusters = max(n_clusters * scale_factor, scale_factor)
    if n_samples < max_n_clusters:
        # KMeans can't have more clusters than samples
        max_n_clusters = n_samples
        min_n_clusters = max(1, max_n_clusters // 4)
    else:
        min_n_clusters = max(n_clusters // scale_factor, 1)

    search_space = {"n_clusters": list(range(min_n_clusters, max_n_clusters + 1))}

    return min_n_clusters, max_n_clusters, search_space


class KMeansObjective:
    def __init__(
        self,
        gdf: gpd.GeoDataFrame,
        min_n_clusters: int,
        max_n_clusters: int,
        id_col: str = "rooftop_id",
        lat_col: str = "Lat_centroid",
        lon_col: str = "Lon_centroid",
        weight_col: Optional[str] = None,
        epsg: int = 26191,  # for morocco
        weight_importance_factor: Union[float, int] = 1,
        target_weight: Union[float, int] = 30,
        target_radius: Union[float, int] = 550,
    ) -> None:
        self.gdf = gdf
        self.min_n_clusters = min_n_clusters
        self.max_n_clusters = max_n_clusters
        self.id_col = id_col
        self.lat_col = lat_col
        self.lon_col = lon_col
        self.weight_col = weight_col
        self.epsg = epsg
        self.weight_importance_factor = weight_importance_factor
        self.target_weight = target_weight
        self.target_radius = target_radius

    def __call__(self, trial) -> float:
        n_clusters = trial.suggest_int(
            "n_clusters",
            self.min_n_clusters,
            self.max_n_clusters,
        )
        clusters = custom_kmeans(
            df=self.gdf,
            n_clusters=n_clusters,
            id_col=self.id_col,
            lat_col=self.lat_col,
            lon_col=self.lon_col,
            weight_col=self.weight_col,
        )
        gdf_w_clusters = self.gdf.merge(clusters, on=self.id_col)

        if self.weight_col is None:
            self.weight_col = "weight"

        return _compute_clustering_score(
            gdf_w_clusters,
            weight_col=self.weight_col,
            epsg=self.epsg,
            weight_importance_factor=self.weight_importance_factor,
            target_weight=self.target_weight,
            target_radius=self.target_radius,
        )


def _compute_clustering_score(
    gdf_w_clusters: gpd.GeoDataFrame,
    weight_col: str,
    epsg: int,
    weight_importance_factor: Union[float, int],
    target_weight: Union[float, int],
    target_radius: Union[float, int],
) -> float:
    # stats that depend on cluster geometries need pivot
    cluster_pivot_gdf = get_cluster_pivot_gdf(
        gdf_w_clusters=gdf_w_clusters,
        cluster_id_col="cluster_id",
        weight_col=weight_col,
        epsg=epsg,
        with_stats=False,
    )
    cluster_pivot_gdf = cluster_pivot_gdf.to_crs(epsg=epsg)

    # get median cluster radius and weight
    radii = cluster_pivot_gdf["geometry"].minimum_bounding_radius()

    # median
    median_radius = radii.median()
    median_weight = cluster_pivot_gdf["cluster_weight"].median()

    # objective function
    auto_scale_factor = target_radius / target_weight
    score = (
        weight_importance_factor
        * auto_scale_factor
        * abs(target_weight - median_weight)
    ) + abs(target_radius - median_radius)

    return score


def custom_kmeans(
    df: pd.DataFrame,
    n_clusters: int,
    id_col: str,
    lat_col: str,
    lon_col: str,
    weight_col: Optional[str] = None,
    cluster_id_prefix: str = "CLUSTER_",
) -> pd.DataFrame:
    if weight_col is None:
        weight_col = "weight"
        df[weight_col] = 1

    # fit kmeans
    kmeans = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42)
    kmeans_clusters = kmeans.fit_predict(
        df[[lon_col, lat_col]], sample_weight=df[weight_col]
    )

    # replace integer cluster IDs with `CLUSTER_001` format
    ids = create_ids(n_clusters, cluster_id_prefix)
    cluster_id_replace_dict = dict(zip(range(n_clusters), ids))
    kmeans_clusters_renamed = [cluster_id_replace_dict[c] for c in kmeans_clusters]

    # create df with cluster IDs
    clusters = pd.DataFrame(
        {
            id_col: df[id_col],
            "cluster_id": kmeans_clusters_renamed,
            weight_col: df[weight_col],
        },
        index=df.index,
    )

    clusters["cluster_weight"] = clusters.groupby("cluster_id")[weight_col].transform(
        np.nansum
    )
    clusters = clusters[[id_col, "cluster_id", "cluster_weight"]]

    return clusters
