from multiprocessing import Pool
from typing import Optional, Union
import logging

import geopandas as gpd
import numpy as np
import optuna
import pandas as pd
from sklearn.cluster import KMeans
from tqdm.notebook import tqdm

optuna.logging.set_verbosity(optuna.logging.WARNING)

from .utils import create_ids, pivot_by_cluster


# # Use for local or EC2 instance
# def parallel_kmeans_secondpass(
#     gdf_w_clusters: gpd.GeoDataFrame,
#     oversized_cluster_ids: list[str],
#     desired_cluster_weight: Union[float, int],
#     desired_cluster_radius: Union[float, int],
#     id_col: str = "rooftop_id",
#     lat_col: str = "Lat_centroid",
#     lon_col: str = "Lon_centroid",
#     weight_col: Optional[str] = None,
#     weight_importance_factor: Union[float, int] = 1,
#     epsg: int = 26191,  # for morocco
#     n_trials: int = 100,
#     python_n_jobs: int = 4,
#     optuna_n_jobs: int = 4,
# ) -> gpd.GeoDataFrame:

#     # make a copy of the gdf
#     gdf_w_clusters_doublepass = gdf_w_clusters.copy()

#     # rerun optimisation for each cluster that is oversized to split it up into smaller ones.
#     # Note: this "partial" usage can be replace with a Class structure
#     # (similar to KMeansObjective)
#     run_optuna_kmeans_secondpass_w_args = partial(
#         recluster,
#         gdf_w_clusters=gdf_w_clusters,
#         desired_cluster_weight=desired_cluster_weight,
#         desired_cluster_radius=desired_cluster_radius,
#         id_col=id_col,
#         lat_col=lat_col,
#         lon_col=lon_col,
#         weight_col=weight_col,
#         weight_importance_factor=weight_importance_factor,
#         epsg=epsg,
#         n_trials=n_trials,
#         n_jobs=optuna_n_jobs,
#     )

#     # without progress bar - can add another time
#     with Pool(processes=python_n_jobs) as pool:
#         oversized_clusters = pool.map(
#             run_optuna_kmeans_secondpass_w_args, oversized_cluster_ids
#         )

#     # replace the cluster_id of the oversized cluster with the new subclusters
#     gdf_w_clusters_doublepass = gdf_w_clusters_doublepass[
#         ~gdf_w_clusters_doublepass["cluster_id"].isin(oversized_cluster_ids)
#     ]
#     gdf_w_clusters_doublepass = pd.concat(
#         [gdf_w_clusters_doublepass] + oversized_clusters
#     )

#     return gdf_w_clusters_doublepass


def get_optimised_kmeans_clusters_doublepass(
    gdf: gpd.GeoDataFrame,
    lat_col: str,
    lon_col: str,
    epsg: int,
    desired_cluster_weight: Union[float, int],
    max_cluster_weight: Union[float, int],
    desired_cluster_radius: Union[float, int],
    weight_importance_factor: Union[float, int] = 1,
    weight_col: Optional[str] = None,
    doubplepass: bool = True,
    initial_n_trials: int = 100,
    secondpass_n_trials: int = 100,
    n_jobs: int = -1,
    show_progress_bar: bool = False,
    return_type: str = "list",
) -> Union[list[str], gpd.GeoDataFrame]:

    # add uniform weight column if none given (all other functions require this)
    if weight_col is None:
        weight_col = "weight"
        gdf.loc[:, "weight"] = 1

    # initial pass
    gdf_w_clusters = get_optimised_kmeans_clusters(
        gdf=gdf,
        lat_col=lat_col,
        lon_col=lon_col,
        weight_col=weight_col,
        epsg=epsg,
        desired_cluster_weight=desired_cluster_weight,
        desired_cluster_radius=desired_cluster_radius,
        weight_importance_factor=weight_importance_factor,
        cluster_id_prefix="CLUSTER_",
        n_trials=initial_n_trials,
        n_jobs=n_jobs,
        show_progress_bar=show_progress_bar,
    )

    # # add urban_guess column
    # gdf_w_clusters.loc[:, "dense_area_guess"] = 0
    # gdf_w_clusters.loc[
    #     gdf_w_clusters["cluster_weight"] > secondpass_cutoff_weight,
    #     "dense_area_guess",
    # ] = 1

    # second pass
    if doubplepass:
        oversized_cluster_ids = _get_oversized_clusters(
            gdf_w_clusters=gdf_w_clusters, cutoff_weight=max_cluster_weight
        )

        n_oversized = len(oversized_cluster_ids)
        print(f"Oversized clusters: {n_oversized}")

        oversized_cluster_gdf_list = []
        for cluster_id in tqdm(oversized_cluster_ids):

            oversized_cluster_gdf = recluster(
                cluster_id=cluster_id,
                gdf_w_clusters=gdf_w_clusters,
                lat_col=lat_col,
                lon_col=lon_col,
                weight_col=weight_col,
                epsg=epsg,
                desired_cluster_weight=desired_cluster_weight,
                desired_cluster_radius=desired_cluster_radius,
                weight_importance_factor=weight_importance_factor,
                n_trials=secondpass_n_trials,
                n_jobs=n_jobs,
            )

            oversized_cluster_gdf_list.append(oversized_cluster_gdf)

        # replace the cluster_ids of the oversized clusters with the new subclusters
        gdf_w_clusters = gdf_w_clusters[
            ~gdf_w_clusters["cluster_id"].isin(oversized_cluster_ids)
        ]
        gdf_w_clusters = pd.concat([gdf_w_clusters] + oversized_cluster_gdf_list)

    if return_type == "list":
        return gdf_w_clusters["cluster_id"].tolist()
    else:
        return gdf_w_clusters


def recluster(
    cluster_id: str,
    gdf_w_clusters: gpd.GeoDataFrame,
    lat_col: str,
    lon_col: str,
    weight_col: str,
    epsg: int,
    desired_cluster_weight: Union[float, int],
    desired_cluster_radius: Union[float, int],
    weight_importance_factor: Union[float, int] = 1,
    n_trials: int = 100,
    n_jobs: int = -1,
) -> gpd.GeoDataFrame:

    # subset data to selected cluster
    oversized_cluster_gdf = (
        gdf_w_clusters[gdf_w_clusters["cluster_id"] == cluster_id]
        .drop(columns=["cluster_id", "cluster_weight"])
        .copy()
    )

    # run optimiser to find best n_clusters
    oversized_cluster_gdf = get_optimised_kmeans_clusters(
        gdf=oversized_cluster_gdf,
        lat_col=lat_col,
        lon_col=lon_col,
        weight_col=weight_col,
        epsg=epsg,
        desired_cluster_weight=desired_cluster_weight,
        desired_cluster_radius=desired_cluster_radius,
        weight_importance_factor=weight_importance_factor,
        cluster_id_prefix=f"{cluster_id}_",  # results like "CLUSTER_001_01"
        n_trials=n_trials,  # maybe have different smaller n_trials here
        n_jobs=n_jobs,
        show_progress_bar=False,
    )

    return oversized_cluster_gdf


def get_optimised_kmeans_clusters(
    gdf: gpd.GeoDataFrame,
    lat_col: str,
    lon_col: str,
    weight_col: str,
    epsg: int,
    desired_cluster_weight: Union[float, int],
    desired_cluster_radius: Union[float, int],
    weight_importance_factor: Union[float, int] = 1,
    cluster_id_prefix: str = "CLUSTER_",
    n_trials: int = 100,
    n_jobs: int = -1,
    show_progress_bar: bool = False,
) -> gpd.GeoDataFrame:

    # run optimiser to find best n_clusters
    study = _run_optuna_kmeans_study(
        gdf=gdf,
        lat_col=lat_col,
        lon_col=lon_col,
        weight_col=weight_col,
        epsg=epsg,
        desired_cluster_weight=desired_cluster_weight,
        desired_cluster_radius=desired_cluster_radius,
        weight_importance_factor=weight_importance_factor,
        n_trials=n_trials,
        n_jobs=n_jobs,
        show_progress_bar=show_progress_bar,
    )

    # run best kmeans
    clusters = get_kmeans_clusters(
        df=gdf,
        n_clusters=study.best_params["n_clusters"],
        lat_col=lat_col,
        lon_col=lon_col,
        weight_col=weight_col,
        cluster_id_prefix=cluster_id_prefix,
    )

    # add cluster_id and cluster_weight columns to gdf
    gdf_w_clusters = gdf.copy()
    gdf_w_clusters.loc[:, "cluster_id"] = clusters
    gdf_w_clusters.loc[:, "cluster_weight"] = gdf_w_clusters.groupby("cluster_id")[
        weight_col
    ].transform("sum")

    return gdf_w_clusters


def get_kmeans_clusters(
    df: pd.DataFrame,
    n_clusters: int,
    lat_col: str,
    lon_col: str,
    weight_col: Optional[str] = None,
    cluster_id_prefix: Optional[str] = None,
) -> list:

    # get data
    X = df[[lon_col, lat_col]].values
    sample_weight = df[weight_col]

    # fit
    kmeans = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42)
    kmeans_clusters = list(kmeans.fit_predict(X=X, sample_weight=sample_weight))

    # rename if required
    if cluster_id_prefix is not None:
        kmeans_clusters = _rename_clusters(
            kmeans_clusters=kmeans_clusters,
            n_clusters=n_clusters,
            cluster_id_prefix=cluster_id_prefix,
        )

    return kmeans_clusters


def _run_optuna_kmeans_study(
    gdf: gpd.GeoDataFrame,
    lat_col: str,
    lon_col: str,
    weight_col: str,
    epsg: int,
    desired_cluster_weight: Union[float, int],
    desired_cluster_radius: Union[float, int],
    weight_importance_factor: Union[float, int] = 1,
    n_trials: int = 100,
    n_jobs: int = -1,
    show_progress_bar: bool = False,
) -> optuna.Study:
    """Run optuna to find the optimal number of clusters for kmeans."""

    # get range of n_clusters to try
    total_weight = gdf[weight_col].sum()
    expected_n_clusters = int(total_weight / desired_cluster_weight)
    n_samples = len(gdf)
    min_n_clusters, max_n_clusters, search_space = _get_min_max_search_space(
        expected_n_clusters, n_samples
    )

    # run study
    study = optuna.create_study(
        sampler=optuna.samplers.GridSampler(search_space, seed=42),
        direction="minimize",
        study_name="kmeans_optimisation",
    )
    optuna_kmeans_objective = OptunaKMeansObjective(
        gdf=gdf,
        lat_col=lat_col,
        lon_col=lon_col,
        weight_col=weight_col,
        epsg=epsg,
        min_n_clusters=min_n_clusters,
        max_n_clusters=max_n_clusters,
        target_weight=desired_cluster_weight,
        target_radius=desired_cluster_radius,
        weight_importance_factor=weight_importance_factor,
    )
    study.optimize(
        optuna_kmeans_objective,
        n_trials=n_trials,
        n_jobs=n_jobs,
        gc_after_trial=True,
        show_progress_bar=show_progress_bar,
    )
    return study


class OptunaKMeansObjective:
    def __init__(
        self,
        gdf: gpd.GeoDataFrame,
        lat_col: str,
        lon_col: str,
        weight_col: str,
        epsg: int,
        min_n_clusters: int,
        max_n_clusters: int,
        target_weight: Union[float, int],
        target_radius: Union[float, int],
        weight_importance_factor: Union[float, int] = 1,
    ) -> None:
        self.gdf = gdf
        self.min_n_clusters = min_n_clusters
        self.max_n_clusters = max_n_clusters
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
        clusters = get_kmeans_clusters(
            df=self.gdf,
            n_clusters=n_clusters,
            lat_col=self.lat_col,
            lon_col=self.lon_col,
            weight_col=self.weight_col,
            cluster_id_prefix=None,
        )
        gdf_w_clusters = self.gdf.copy()
        gdf_w_clusters.loc[:, "cluster_id"] = clusters

        return _compute_clustering_score(
            gdf_w_clusters=gdf_w_clusters,
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
    target_weight: Union[float, int],
    target_radius: Union[float, int],
    weight_importance_factor: Union[float, int] = 1,
) -> float:
    """Requires gdf_w_clusters to have a 'cluster_id' and weight_col column."""
    # stats that depend on cluster geometries need pivot
    cluster_pivot_gdf = pivot_by_cluster(
        gdf_w_clusters=gdf_w_clusters,
        cluster_id_col="cluster_id",
        weight_col=weight_col,
        cols_to_keep=[],
    )

    # get median cluster radius and weight
    cluster_pivot_gdf = cluster_pivot_gdf.to_crs(epsg=epsg)
    median_radius = cluster_pivot_gdf["geometry"].minimum_bounding_radius().median()
    median_weight = cluster_pivot_gdf["cluster_weight"].median()

    # objective function
    auto_scale_factor = target_radius / target_weight
    score = (
        weight_importance_factor
        * auto_scale_factor
        * abs(target_weight - median_weight)
    ) + abs(target_radius - median_radius)

    return score


def _get_min_max_search_space(
    expected_n_clusters: int, n_samples: int, scale_factor: int = 2
) -> tuple[int, int, dict[str, list[int]]]:
    """
    Given the expected_n_clusters and n_samples, returns the min and max number of
    clusters to try as well as an Optuna-digested search_space based on this.

    The scale factor is used to determine how many clusters to try. Expected
    behaviour is:
        min_n_clusters = expected_n_clusters // scale_factor
        max_n_clusters = expected_n_clusters * scale_factor
    However, if n_samples < max_n_clusters, KMeans will fail on the upper bound
    since it can't have more clusters than samples. In this case, max_n_clusters
    is set to n_samples and min_n_clusters is set to max_n_clusters // 4.

    Also, the min and max are always at least 1.
    """

    max_n_clusters = max(expected_n_clusters * scale_factor, scale_factor)
    if n_samples < max_n_clusters:
        # KMeans can't have more clusters than samples
        max_n_clusters = n_samples
        min_n_clusters = max(1, max_n_clusters // 4)
    else:
        min_n_clusters = max(expected_n_clusters // scale_factor, 1)

    search_space = {"n_clusters": list(range(min_n_clusters, max_n_clusters + 1))}

    return min_n_clusters, max_n_clusters, search_space


def _get_oversized_clusters(
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


def _rename_clusters(
    kmeans_clusters: np.ndarray,
    n_clusters: int,
    cluster_id_prefix: str,
) -> list[str]:

    # replace integer cluster IDs with `CLUSTER_001` format
    ids = create_ids(n_clusters, cluster_id_prefix)
    cluster_id_replace_dict = dict(zip(range(n_clusters), ids))

    return [cluster_id_replace_dict[c] for c in kmeans_clusters]
