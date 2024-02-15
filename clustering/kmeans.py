from multiprocessing import Pool
from typing import Optional, Union, Literal

import geopandas as gpd
import numpy as np
import optuna
import pandas as pd
from sklearn.cluster import KMeans
from tqdm.notebook import tqdm

optuna.logging.set_verbosity(optuna.logging.WARNING)

from .utils import create_ids, pivot_by_cluster, split_n_jobs


def get_multipass_optimised_clusters(
    gdf: gpd.GeoDataFrame,
    lat_col: str,
    lon_col: str,
    projected_epsg: int,
    desired_cluster_weight: Union[float, int],
    desired_cluster_radius: Union[float, int],
    weight_importance_factor: Union[float, int] = 1,
    weight_col: Optional[str] = None,
    initial_max_trials: int = 100,
    n_jobs: int = -1,
    n_passes: int = 1,
    parallel_reclustering: bool = True,
    subsequent_max_trials: int = 100,
    max_cluster_weight: Optional[Union[float, int]] = None,
    show_progress_bar: bool = False,
    return_type: Literal["geodataframe", "list"] = "geodataframe",
) -> Union[list[str], gpd.GeoDataFrame]:
    """
    Cluster data using KMeans with the option of running multiple passes to handle
    oversized clusters.

    Features:
    - Ability to set the `weight_importance_factor` to control the importance of
        cluster weight versus cluster radius in the clustering process.
    - Parallelised optimisation using Optuna.
    - Ability to run multiple passes to break up oversized clusters.
    - Ability to toggle progress bar on/off.

    Steps:
    - Run initial pass to get clusters
    - Check for oversized clusters
    - If any oversized clusters found, recluster them
    - Repeat until no oversized clusters found or until max number of passes reached

    Note: Cluster IDs are of the format "CLUSTER_001". If clusters are borken up
        over different passes, subcluster numbers are added on for every breakup,
        e.g. "CLUSTER_001_01_...".

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame to cluster.
    lat_col : str
        Name of the column containing latitude data.
    lon_col : str
        Name of the column containing longitude data.
    projected_epsg : int
        EPSG code for the projected coordinate reference system for the region of
        interest. This is used to calculate the minimum bounding radius of clusters.
    desired_cluster_weight : float or int
        Desired weight of each cluster. This is used to optimise clustering.
    desired_cluster_radius : float or int
        Desired radius of each cluster. This is used to optimise clustering.
    weight_importance_factor : float or int, default=1
        Factor to control the importance of cluster weight versus cluster radius in the
        clustering process. Higher values will prioritise weight over radius.
    weight_col : str, default=None
        Name of the column containing weight data. If None, a uniform weight of 1 is
        used for all points.
    initial_max_trials : int, default=100
        Number of trials to run in the initial pass.
    n_jobs : int, default=-1
        Number of parallel jobs to run. -1 means using all processors.
    n_passes : int, default=1
        Number of passes to run. If 1, only the initial pass is run.
    parallel_reclustering : bool, default=True
        Must be set if n_passes > 1. Whether to parallelise the reclustering process.
        If False, the reclustering process will be run sequentially.
    subsequent_max_trials : int, default=100
        Must be set if n_passes > 1. Number of trials to run in subsequent passes.
    max_cluster_weight : float or int
        Must be set if n_passes > 1. Maximum weight above which a cluster is considered
        oversized and will be reclustered.
    show_progress_bar : bool, default=False
        Whether to show a progress bar for the optimisation process.
    return_type : "list" or "geodataframe", default="geodataframe"
        Whether to return a GeoDataFrame or a list of cluster IDs.

    Returns
    -------
    gpd.GeoDataFrame or list
        If "geodataframe", returns original GeoDataFrame with additional columns for
            "cluster_id", "cluster_weight", and "cluster_pass".
            The "cluster_pass" indicates which pass the cluster was formed in and
            could loosely be interpreted as a measure of "density".
        If return_type is "list", only returns a list of cluster IDs
    """

    # check inputs
    if max_cluster_weight is None and n_passes > 1:
        raise ValueError("max_cluster_weight must be set if n_passes > 1.")

    # check for positivity
    if desired_cluster_weight <= 0 or desired_cluster_radius <= 0:
        raise ValueError(
            "Both desired_cluster_weight and desired_cluster_radius must be positive."
        )
    if weight_importance_factor <= 0:
        raise ValueError("weight_importance_factor must be positive.")

    # check for integerhood and positivity
    if not isinstance(n_passes, int) or n_passes < 1:
        raise ValueError("n_passes must be a positive integer.")
    if not isinstance(initial_max_trials, int) or initial_max_trials < 1:
        raise ValueError("initial_max_trials must be a positive integer.")
    if not isinstance(subsequent_max_trials, int) or subsequent_max_trials < 1:
        raise ValueError("subsequent_max_trials must be a positive integer.")

    if return_type != "geodataframe" and return_type != "list":
        raise ValueError("return_type must be either 'geodataframe' or 'list'")

    # add uniform weight column if none given (all other functions require this)
    if weight_col is None:
        weight_col = "weight"
        gdf.loc[:, "weight"] = [1] * len(gdf)

    # initial pass
    gdf_w_clusters = get_optimised_clusters(
        gdf=gdf,
        lat_col=lat_col,
        lon_col=lon_col,
        weight_col=weight_col,
        projected_epsg=projected_epsg,
        desired_cluster_weight=desired_cluster_weight,
        desired_cluster_radius=desired_cluster_radius,
        weight_importance_factor=weight_importance_factor,
        cluster_id_prefix="CLUSTER_",
        max_trials=initial_max_trials,
        n_jobs=n_jobs,
        show_progress_bar=show_progress_bar,
    )

    # add cluster_pass column to track which pass each cluster was formed in
    gdf_w_clusters.loc[:, "cluster_pass"] = 1

    for i in range(1, n_passes):
        oversized_cluster_ids = _get_oversized_clusters(
            gdf_w_clusters=gdf_w_clusters, cutoff_weight=max_cluster_weight
        )

        # update cluster_pass column with which pass it was formed in
        gdf_w_clusters.loc[
            gdf_w_clusters["cluster_id"].isin(oversized_cluster_ids),
            "cluster_pass",
        ] = (
            i + 1
        )

        n_oversized = len(oversized_cluster_ids)
        if i == 1:
            print(f"{n_oversized} oversized clusters found in initial pass.")
        if i > 1:
            print(f"{n_oversized} oversized clusters left after {i} passes.")

        if n_oversized > 0:
            # initialise ReCluster object with correct parameters
            recluster = ReCluster(
                gdf_w_clusters=gdf_w_clusters,
                lat_col=lat_col,
                lon_col=lon_col,
                weight_col=weight_col,
                projected_epsg=projected_epsg,
                desired_cluster_weight=desired_cluster_weight,
                desired_cluster_radius=desired_cluster_radius,
                weight_importance_factor=weight_importance_factor,
                max_trials=subsequent_max_trials,
                n_jobs=n_jobs,
                parallel_reclustering=parallel_reclustering,
                show_progress_bar=show_progress_bar,
                progress_bar_desc=f"Pass {i+1} ({n_oversized} oversized clusters)",
            )
            # Run reclustering
            gdf_w_clusters = recluster(oversized_cluster_ids=oversized_cluster_ids)
        else:
            print("No more oversized clusters found, stopping early.")
            break

    if return_type == "geodataframe":
        return gdf_w_clusters
    elif return_type == "list":
        return gdf_w_clusters["cluster_id"].tolist()
    else:
        raise ValueError("return_type must be either 'geodataframe' or 'list'")


def _get_oversized_clusters(
    gdf_w_clusters: Union[gpd.GeoDataFrame, pd.DataFrame],
    cutoff_weight: Union[float, int],
) -> list[str]:
    """Get list of IDs of clusters that are oversized."""

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


class ReCluster:
    """
    Recluster all oversized clusters with IDs in oversized_cluster_ids.

    Initialise with the parameters with `recluster = ReCluster()` and then run
    `recluster()` to recluster the oversized clusters.

    Steps:
    - Run SingleRecluster.run() for each oversized cluster
    - Drop old original oversized clusters rows
    - Re-add rows with new subcluster labels and weights. E.g. "CLUSTER_01_1".

    ALERT:
    - Row order is not preserved.
    """

    def __init__(
        self,
        gdf_w_clusters: gpd.GeoDataFrame,
        lat_col: str,
        lon_col: str,
        weight_col: str,
        projected_epsg: int,
        desired_cluster_weight: Union[float, int],
        desired_cluster_radius: Union[float, int],
        weight_importance_factor: Union[float, int] = 1,
        max_trials: int = 100,
        n_jobs: int = -1,
        parallel_reclustering: bool = False,
        show_progress_bar: bool = False,
        progress_bar_desc: str = "Reclustering",
    ):
        self.gdf_w_clusters = gdf_w_clusters
        self.lat_col = lat_col
        self.lon_col = lon_col
        self.weight_col = weight_col
        self.projected_epsg = projected_epsg
        self.desired_cluster_weight = desired_cluster_weight
        self.desired_cluster_radius = desired_cluster_radius
        self.weight_importance_factor = weight_importance_factor
        self.max_trials = max_trials
        self.n_jobs = n_jobs
        self.parallel_reclustering = parallel_reclustering
        self.show_progress_bar = show_progress_bar
        self.progress_bar_desc = progress_bar_desc

    def __call__(self, oversized_cluster_ids: list[str]):

        if self.parallel_reclustering:
            reclustered_clusters = self.recluster_parallel(oversized_cluster_ids)
        else:
            reclustered_clusters = self.recluster_sequential(oversized_cluster_ids)
        reclustered_gdf_w_clusters = self.drop_and_add_rows(
            reclustered_clusters, oversized_cluster_ids
        )
        return reclustered_gdf_w_clusters

    def recluster_parallel(self, oversized_cluster_ids: list[str]):
        # split n_jobs into n_parallel_clusters and optuna_n_jobs
        n_parallel_clusters, optuna_n_jobs = split_n_jobs(self.n_jobs)
        # instantiate SingleReCluster with optuna_n_jobs
        single_recluster = SingleReCluster(
            gdf_w_clusters=self.gdf_w_clusters,
            lat_col=self.lat_col,
            lon_col=self.lon_col,
            weight_col=self.weight_col,
            projected_epsg=self.projected_epsg,
            desired_cluster_weight=self.desired_cluster_weight,
            desired_cluster_radius=self.desired_cluster_radius,
            weight_importance_factor=self.weight_importance_factor,
            max_trials=self.max_trials,
            n_jobs=optuna_n_jobs,
        )
        # recluster oversized clusters in parallel
        reclustered_clusters = []
        if self.show_progress_bar:
            with Pool(processes=n_parallel_clusters) as pool:
                reclustered_clusters = list(
                    tqdm(
                        pool.imap(single_recluster, oversized_cluster_ids),
                        total=len(oversized_cluster_ids),
                        desc=self.progress_bar_desc,
                    )
                )
        else:
            with Pool(processes=n_parallel_clusters) as pool:
                reclustered_clusters = pool.map(single_recluster, oversized_cluster_ids)
        return reclustered_clusters

    def recluster_sequential(self, oversized_cluster_ids: list[str]):
        # instantiate SingleReCluster with n_jobs
        single_recluster = SingleReCluster(
            gdf_w_clusters=self.gdf_w_clusters,
            lat_col=self.lat_col,
            lon_col=self.lon_col,
            weight_col=self.weight_col,
            projected_epsg=self.projected_epsg,
            desired_cluster_weight=self.desired_cluster_weight,
            desired_cluster_radius=self.desired_cluster_radius,
            weight_importance_factor=self.weight_importance_factor,
            max_trials=self.max_trials,
            n_jobs=self.n_jobs,
        )
        # recluster oversized clusters sequentially
        reclustered_clusters = []
        if self.show_progress_bar:
            for cluster_id in tqdm(oversized_cluster_ids, desc=self.progress_bar_desc):
                oversized_cluster_gdf = single_recluster(
                    cluster_id=cluster_id,
                )
                reclustered_clusters.append(oversized_cluster_gdf)
        else:
            for cluster_id in oversized_cluster_ids:
                oversized_cluster_gdf = single_recluster(
                    cluster_id=cluster_id,
                )
                reclustered_clusters.append(oversized_cluster_gdf)

        return reclustered_clusters

    def drop_and_add_rows(
        self,
        reclustered_clusters: list[gpd.GeoDataFrame],
        oversized_cluster_ids: list[str],
    ):
        """Drop old oversized clusters rows and re-add rows with new subclusters."""

        reclustered_gdf_w_clusters = self.gdf_w_clusters.copy()
        # drop old oversized clusters rows
        reclustered_gdf_w_clusters = reclustered_gdf_w_clusters[
            ~reclustered_gdf_w_clusters["cluster_id"].isin(oversized_cluster_ids)
        ]
        # re-add rows with new subclusters
        reclustered_gdf_w_clusters = pd.concat(
            [reclustered_gdf_w_clusters] + reclustered_clusters
        )

        return reclustered_gdf_w_clusters


class SingleReCluster:
    """
    Re-cluster a single oversized cluster running get_optimised_clusters().

    This function is used in ReCluster repeatedly to recluster every oversized
    cluster.

    Steps:
    - Subset data to selected cluster
    - Run optimiser to find best n_clusters
    - Return the original GeoDataFrame with added cluster_id and cluster_weight columns
    """

    def __init__(
        self,
        gdf_w_clusters: gpd.GeoDataFrame,
        lat_col: str,
        lon_col: str,
        weight_col: str,
        projected_epsg: int,
        desired_cluster_weight: Union[float, int],
        desired_cluster_radius: Union[float, int],
        weight_importance_factor: Union[float, int] = 1,
        max_trials: int = 100,
        n_jobs: int = -1,
    ):
        self.gdf_w_clusters = gdf_w_clusters
        self.lat_col = lat_col
        self.lon_col = lon_col
        self.weight_col = weight_col
        self.projected_epsg = projected_epsg
        self.desired_cluster_weight = desired_cluster_weight
        self.desired_cluster_radius = desired_cluster_radius
        self.weight_importance_factor = weight_importance_factor
        self.max_trials = max_trials
        self.n_jobs = n_jobs

    def __call__(
        self,
        cluster_id: str,
    ) -> gpd.GeoDataFrame:
        # subset data to selected cluster
        oversized_cluster_gdf = (
            self.gdf_w_clusters[self.gdf_w_clusters["cluster_id"] == cluster_id]
            .drop(columns=["cluster_id", "cluster_weight"])
            .copy()
        )
        # run optimiser to find best n_clusters
        oversized_cluster_gdf = get_optimised_clusters(
            gdf=oversized_cluster_gdf,
            lat_col=self.lat_col,
            lon_col=self.lon_col,
            weight_col=self.weight_col,
            projected_epsg=self.projected_epsg,
            desired_cluster_weight=self.desired_cluster_weight,
            desired_cluster_radius=self.desired_cluster_radius,
            weight_importance_factor=self.weight_importance_factor,
            cluster_id_prefix=f"{cluster_id}_",  # results like "CLUSTER_001_01"
            max_trials=self.max_trials,  # maybe have different smaller max_trials here
            n_jobs=self.n_jobs,
            show_progress_bar=False,
        )
        return oversized_cluster_gdf


def get_optimised_clusters(
    gdf: gpd.GeoDataFrame,
    lat_col: str,
    lon_col: str,
    weight_col: str,
    projected_epsg: int,
    desired_cluster_weight: Union[float, int],
    desired_cluster_radius: Union[float, int],
    weight_importance_factor: Union[float, int] = 1,
    cluster_id_prefix: str = "CLUSTER_",
    max_trials: int = 100,
    n_jobs: int = -1,
    show_progress_bar: bool = False,
) -> gpd.GeoDataFrame:
    """
    Run optimised KMeans and return GeoDataFrame with cluster_id and cluster_weight
    columns.

    Steps:
    - Runs an Optuna study to find the optimal number of clusters for KMeans
    - Runs KMeans with the optimal number of clusters
    - Returns the original GeoDataFrame with added cluster_id and cluster_weight columns
    """

    # check in case all weights are zero (avoid sparse matrix error in KMeans)
    if gdf[weight_col].sum() == 0:
        print("All weights are zero. Tagging everything with CLUSTER_0 and 0.0 weight.")
        gdf_w_clusters = gdf.copy()
        gdf_w_clusters.loc[:, "cluster_id"] = "CLUSTER_0"
        gdf_w_clusters.loc[:, "cluster_weight"] = 0.0
        return gdf_w_clusters

    # run optimiser to find best n_clusters
    study = _run_optuna_study(
        gdf=gdf,
        lat_col=lat_col,
        lon_col=lon_col,
        weight_col=weight_col,
        projected_epsg=projected_epsg,
        desired_cluster_weight=desired_cluster_weight,
        desired_cluster_radius=desired_cluster_radius,
        weight_importance_factor=weight_importance_factor,
        max_trials=max_trials,
        n_jobs=n_jobs,
        show_progress_bar=show_progress_bar,
    )

    # run kmeans with best n_clusters
    clusters = get_clusters(
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


def _run_optuna_study(
    gdf: gpd.GeoDataFrame,
    lat_col: str,
    lon_col: str,
    weight_col: str,
    projected_epsg: int,
    desired_cluster_weight: Union[float, int],
    desired_cluster_radius: Union[float, int],
    weight_importance_factor: Union[float, int] = 1,
    max_trials: int = 100,
    n_jobs: int = -1,
    show_progress_bar: bool = False,
) -> optuna.Study:
    """Run Optuna to find the optimal number of clusters for kmeans."""

    min_n_clusters, max_n_clusters, search_space = _get_min_max_search_space(
        gdf=gdf,
        weight_col=weight_col,
        desired_cluster_weight=desired_cluster_weight,
        scale_factor=2,
    )

    # run study
    study = optuna.create_study(
        sampler=optuna.samplers.GridSampler(search_space, seed=42),
        direction="minimize",
        study_name="optimisation",
    )
    optuna_objective = OptunaKMeansObjective(
        gdf=gdf,
        lat_col=lat_col,
        lon_col=lon_col,
        weight_col=weight_col,
        projected_epsg=projected_epsg,
        min_n_clusters=min_n_clusters,
        max_n_clusters=max_n_clusters,
        target_weight=desired_cluster_weight,
        target_radius=desired_cluster_radius,
        weight_importance_factor=weight_importance_factor,
    )
    study.optimize(
        optuna_objective,
        n_trials=max_trials,
        n_jobs=n_jobs,
        gc_after_trial=True,
        show_progress_bar=show_progress_bar,
    )
    return study


def _get_min_max_search_space(
    gdf: gpd.GeoDataFrame,
    weight_col: str,
    desired_cluster_weight: Union[float, int],
    scale_factor: int = 2,
) -> tuple[int, int, dict[str, list[int]]]:
    """
    NOTE: CHANGE THIS TO NORMAL DISTRIBUTION.

    This function generates the range of n_clusters for Optuna to try for KMeans.

    Given the expected_n_clusters and n_samples, it returns the min and max number of
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

    # get range of n_clusters to try
    total_weight = gdf[weight_col].sum()
    expected_n_clusters = int(total_weight / desired_cluster_weight)
    n_samples = len(gdf)

    max_n_clusters = max(expected_n_clusters * scale_factor, scale_factor)
    if n_samples < max_n_clusters:
        # KMeans can't have more clusters than samples
        max_n_clusters = n_samples
        min_n_clusters = max(1, max_n_clusters // 4)
    else:
        min_n_clusters = max(expected_n_clusters // scale_factor, 1)

    search_space = {"n_clusters": list(range(min_n_clusters, max_n_clusters + 1))}

    return min_n_clusters, max_n_clusters, search_space


class OptunaKMeansObjective:
    """
    Class to create the objective function for Optuna to use to find the optimal number
    of clusters for kmeans.

    Steps:
    - Fits a KMeans model to the data with the n_clusters suggested by Optuna
    - Returns the score of the clustering as per _compute_clustering_score()
    """

    def __init__(
        self,
        gdf: gpd.GeoDataFrame,
        lat_col: str,
        lon_col: str,
        weight_col: str,
        projected_epsg: int,
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
        self.projected_epsg = projected_epsg
        self.weight_importance_factor = weight_importance_factor
        self.target_weight = target_weight
        self.target_radius = target_radius

    def __call__(self, trial) -> float:
        n_clusters = trial.suggest_int(
            "n_clusters",
            self.min_n_clusters,
            self.max_n_clusters,
        )
        clusters = get_clusters(
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
            projected_epsg=self.projected_epsg,
            target_weight=self.target_weight,
            target_radius=self.target_radius,
            weight_importance_factor=self.weight_importance_factor,
        )


def _compute_clustering_score(
    gdf_w_clusters: gpd.GeoDataFrame,
    weight_col: str,
    projected_epsg: int,
    target_weight: Union[float, int],
    target_radius: Union[float, int],
    weight_importance_factor: Union[float, int] = 1,
) -> float:
    """Compute the objective score for a given clustering. Lower is better."""

    # stats that depend on cluster geometries need pivot
    cluster_pivot_gdf = pivot_by_cluster(
        gdf_w_clusters=gdf_w_clusters,
        cluster_id_col="cluster_id",
        weight_col=weight_col,
    )

    # get median cluster radius and weight
    cluster_pivot_gdf = cluster_pivot_gdf.to_crs(epsg=projected_epsg)
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


def get_clusters(
    df: pd.DataFrame,
    n_clusters: int,
    lat_col: str,
    lon_col: str,
    weight_col: Optional[str] = None,
    cluster_id_prefix: Optional[str] = None,
) -> list:
    """Run KMeans and return list of cluster IDs. Optionally rename clusters."""

    # get data
    X = df[[lon_col, lat_col]].values
    sample_weight = df[weight_col]

    # fit
    kmeans = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42)
    clusters = list(kmeans.fit_predict(X=X, sample_weight=sample_weight))

    # rename if required
    if cluster_id_prefix is not None:
        clusters = _rename_clusters(
            clusters=clusters,
            n_clusters=n_clusters,
            cluster_id_prefix=cluster_id_prefix,
        )

    return clusters


def _rename_clusters(
    clusters: np.ndarray,
    n_clusters: int,
    cluster_id_prefix: str,
) -> list[str]:
    """Create list of string cluster IDs with a prefix."""

    # replace integer cluster IDs with `CLUSTER_001` format
    ids = create_ids(n_clusters, cluster_id_prefix)
    cluster_id_replace_dict = dict(zip(range(n_clusters), ids))

    return [cluster_id_replace_dict[c] for c in clusters]
