from multiprocessing import Pool
from typing import Optional, Union, Literal

import geopandas as gpd
import numpy as np
from scipy.stats import norm
import optuna
import pandas as pd
from sklearn.cluster import KMeans
from tqdm.notebook import tqdm

optuna.logging.set_verbosity(optuna.logging.WARNING)

from .utils import create_ids, pivot_by_cluster, split_n_jobs


class TunedClustering:
    """
    Cluster data using KMeans with the option of running multiple passes to handle
    oversized clusters.

    Features:
    - Ability to set the `weight_importance_factor` to control the importance of
        cluster weight versus cluster radius in the clustering process.
    - Parallelised optimisation using Optuna (`n_jobs`).
    - Ability to run multiple passes to break up oversized clusters (`max_n_passes`).
    - Ability to toggle progress bars on/off (`show_progress_bar`).

    Steps:
    - Run initial pass to get clusters that are close to the desired weight and radius
    - Check for oversized clusters
    - If any oversized clusters found, recluster them
    - Repeat until no oversized clusters found or until max number of passes reached

    Note: Cluster IDs are of the format "CLUSTER_001". If clusters are borken up
        over different passes, subcluster numbers are added on for every breakup,
        e.g. "CLUSTER_001_01_...".

    Attributes
    ----------
    desired_cluster_weight : float or int
        Desired weight of each cluster. This is used to optimise clustering.
    desired_cluster_radius : float or int
        Desired radius of each cluster. This is used to optimise clustering.
    weight_importance_factor : float or int, default=1
        Factor to control the importance of cluster weight versus cluster radius in the
        clustering process. Higher values will prioritise weight over radius.
    initial_max_trials : int, default=100
        Number of trials to run in the initial pass.
    max_n_passes : int, default=1
        Max number of passes to run to breakup oversized clusters.
        The algorithm will stop early if there are no oversized clusters left or if the
        number of oversized clusters does not change over 3 passes.
        - Setting this to a high number is roughly equivalent to running the algorithm
            until no more benefit is gained from further passes, but with the safety
            of a hard stop at `max_n_passes`.
        - If set to 1, only the initial pass is run.
    max_cluster_weight : float or int
        Must be set if max_n_passes > 1. Maximum weight above which a cluster is
        considered oversized and will be reclustered.
    subsequent_max_trials : int, default=100
        Must be set if max_n_passes > 1. Number of trials to run in subsequent passes.
    n_jobs : int, default=-1
        Number of parallel jobs to run. -1 means using all processors.
    show_progress_bar : bool, default=False
        Whether to show a progress bar for the optimisation process.

    Methods
    -------
    run(gdf, lat_col, lon_col, projected_epsg, weight_col=None):
        Runs the clustering process and returns the result.
    """

    def __init__(
        self,
        desired_cluster_weight: Union[float, int],
        desired_cluster_radius: Union[float, int],
        weight_importance_factor: Union[float, int] = 1,
        initial_max_trials: int = 100,
        max_n_passes: int = 1,
        max_cluster_weight: Optional[Union[float, int]] = None,
        subsequent_max_trials: int = 100,
        n_jobs: int = -1,
        show_progress_bar: bool = False,
    ) -> None:

        self.desired_cluster_weight = desired_cluster_weight
        self.desired_cluster_radius = desired_cluster_radius
        self.weight_importance_factor = weight_importance_factor
        self.initial_max_trials = initial_max_trials
        self.n_jobs = n_jobs
        self.max_n_passes = max_n_passes
        self.max_cluster_weight = max_cluster_weight
        self.subsequent_max_trials = subsequent_max_trials
        self.show_progress_bar = show_progress_bar
        self._check_input_params()

    def run(
        self,
        gdf: gpd.GeoDataFrame,
        lat_col: str,
        lon_col: str,
        projected_epsg: int,
        weight_col: Optional[str] = None,
        return_type: Literal["geodataframe", "list"] = "geodataframe",
    ) -> Union[list[str], gpd.GeoDataFrame]:
        """
        Run the clustering process.

        Parameters
        ----------
        gdf : gpd.GeoDataFrame
            GeoDataFrame to cluster.
        lat_col : str
            Name of the column containing latitude.
        lon_col : str
            Name of the column containing longitude.
        projected_epsg : int
            EPSG code for the projected coordinate system to use.
        weight_col : str, default=None
            Name of the column containing weights. If None, a uniform weight of 1 is
            used for all rows.
        return_type : "list" or "geodataframe", default="geodataframe"
            Whether to return a GeoDataFrame or a list of cluster IDs.

        Returns
        -------
        gpd.GeoDataFrame or list
            If "geodataframe", returns original GeoDataFrame with additional columns for
                "cluster_id", "cluster_weight", and "cluster_pass".
                The "cluster_pass" indicates which pass the cluster
                was formed in and could loosely be interpreted as a measure of "density".
            If return_type is "list", only returns a list of cluster IDs
        """

        if return_type != "geodataframe" and return_type != "list":
            raise ValueError("return_type must be either 'geodataframe' or 'list'")

        # add uniform weight column if none given
        # (all other functions require a weight column to exist)
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
            desired_cluster_weight=self.desired_cluster_weight,
            desired_cluster_radius=self.desired_cluster_radius,
            weight_importance_factor=self.weight_importance_factor,
            cluster_id_prefix="CLUSTER_",
            max_trials=self.initial_max_trials,
            n_jobs=self.n_jobs,
            show_progress_bar=self.show_progress_bar,
        )

        # add cluster_pass column to track which pass each cluster was formed in
        gdf_w_clusters.loc[:, "cluster_pass"] = 1

        # passes 2 onwards...
        n_oversized_history = []
        for i in range(2, self.max_n_passes + 1):
            oversized_cluster_ids = self._get_oversized_clusters(
                gdf_w_clusters=gdf_w_clusters, cutoff_weight=self.max_cluster_weight
            )
            n_oversized = len(oversized_cluster_ids)
            n_oversized_history.append(n_oversized)
            print(f"{n_oversized} oversized clusters left after {i-1} passes.")

            # stopping conditions
            if n_oversized == 0:
                print("No more oversized clusters found. Stopping early.")
                break
            elif (
                i >= 3
                and n_oversized_history[-1]
                == n_oversized_history[-2]
                == n_oversized_history[-3]
            ):
                print(
                    f"Number of oversized clusters {n_oversized} has not changed "
                    f"compared to the previous 2 runs. Stopping early."
                )
                break
            # continue if not stopping
            else:
                recluster = ReCluster(
                    gdf_w_clusters=gdf_w_clusters,
                    lat_col=lat_col,
                    lon_col=lon_col,
                    weight_col=weight_col,
                    projected_epsg=projected_epsg,
                    desired_cluster_weight=self.desired_cluster_weight,
                    desired_cluster_radius=self.desired_cluster_radius,
                    weight_importance_factor=self.weight_importance_factor,
                    max_trials=self.subsequent_max_trials,
                    n_jobs=self.n_jobs,
                    show_progress_bar=self.show_progress_bar,
                    progress_bar_desc=f"Pass {i} ({n_oversized} oversized clusters)",
                )
                gdf_w_clusters = recluster.run(
                    oversized_cluster_ids=oversized_cluster_ids
                )

                # update cluster_pass column with which pass the cluster was formed in
                gdf_w_clusters.loc[
                    gdf_w_clusters["cluster_id"].isin(oversized_cluster_ids),
                    "cluster_pass",
                ] = i

        if return_type == "geodataframe":
            return gdf_w_clusters
        elif return_type == "list":
            return gdf_w_clusters["cluster_id"].tolist()
        else:
            raise ValueError("return_type must be either 'geodataframe' or 'list'")

    def _get_oversized_clusters(
        self,
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

    def _check_input_params(self):
        # check for valid parameters
        if self.desired_cluster_weight <= 0 or self.desired_cluster_radius <= 0:
            raise ValueError(
                "Both desired_cluster_weight and desired_cluster_radius "
                "must be non-zero and positive."
            )
        if self.weight_importance_factor < 0:
            raise ValueError("weight_importance_factor must be 0 or more.")
        if not isinstance(self.n_jobs, int) or self.n_jobs < -1 or self.n_jobs == 0:
            raise ValueError("n_jobs must be -1 or a positive integer.")
        if not isinstance(self.max_n_passes, int) or self.max_n_passes <= 0:
            raise ValueError("max_n_passes must be a positive integer.")
        if not isinstance(self.initial_max_trials, int) or self.initial_max_trials <= 0:
            raise ValueError("initial_max_trials must be a positive integer.")
        # check multi-pass parameters
        if self.max_n_passes > 1:
            if self.max_cluster_weight is None or self.max_cluster_weight <= 0:
                raise ValueError(
                    "max_cluster_weight must be set to a positive non-zero value "
                    "if max_n_passes > 1."
                )
            if (
                not isinstance(self.subsequent_max_trials, int)
                or self.subsequent_max_trials <= 0
            ):
                raise ValueError(
                    "subsequent_max_trials must be set to a positive integer "
                    "if max_n_passes > 1."
                )
        if not isinstance(self.show_progress_bar, bool):
            raise ValueError("show_progress_bar must be a boolean.")


class ReCluster:
    """
    Recluster all oversized clusters with IDs in oversized_cluster_ids.

    How to use:
    - Initialise with parameters with `recluster = ReCluster(...)`
    - Then run `recluster.run(oversized_cluster_ids)`

    Steps:
    - Run SingleReCluster.run() for each oversized cluster
    - Drop old original oversized clusters rows
    - Re-add rows with new subcluster labels and weights. E.g. "CLUSTER_01_1".

    ALERT:
    - Row order is not preserved. Sort result by index if necessary.
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
        self.show_progress_bar = show_progress_bar
        self.progress_bar_desc = progress_bar_desc

    def run(self, oversized_cluster_ids: list[str]):
        """External method to run reclustering."""

        if self.n_jobs == 1:
            reclustered_clusters = self._recluster_sequential(oversized_cluster_ids)
        else:
            reclustered_clusters = self._recluster_parallel(oversized_cluster_ids)

        reclustered_gdf_w_clusters = self._drop_and_add_rows(
            reclustered_clusters, oversized_cluster_ids
        )
        return reclustered_gdf_w_clusters

    def _recluster_parallel(self, oversized_cluster_ids: list[str]):
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
                        pool.imap(single_recluster.run, oversized_cluster_ids),
                        total=len(oversized_cluster_ids),
                        desc=self.progress_bar_desc,
                    )
                )
        else:
            with Pool(processes=n_parallel_clusters) as pool:
                reclustered_clusters = pool.map(
                    single_recluster.run, oversized_cluster_ids
                )
        return reclustered_clusters

    def _recluster_sequential(self, oversized_cluster_ids: list[str]):
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
                oversized_cluster_gdf = single_recluster.run(
                    cluster_id=cluster_id,
                )
                reclustered_clusters.append(oversized_cluster_gdf)
        else:
            for cluster_id in oversized_cluster_ids:
                oversized_cluster_gdf = single_recluster.run(
                    cluster_id=cluster_id,
                )
                reclustered_clusters.append(oversized_cluster_gdf)

        return reclustered_clusters

    def _drop_and_add_rows(
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

    How to use:
    - Initialise with parameters with `single_recluster = SingleReCluster(...)`
    - Then run `single_recluster.run(cluster_id)`

    This function is used in ReCluster repeatedly to recluster every oversized
    cluster.

    Steps:
    - Subset data to selected cluster
    - Run optimiser to find best n_clusters
    - Return the original GeoDataFrame with *new* cluster_id and cluster_weight columns
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

    def run(
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
            max_trials=self.max_trials,
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
        gdf_w_clusters.loc[:, "cluster_id"] = f"{cluster_id_prefix}0"
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

    search_space = _get_search_space(
        gdf=gdf,
        weight_col=weight_col,
        desired_cluster_weight=desired_cluster_weight,
        sample_size=max_trials,
    )

    # run study
    study = optuna.create_study(
        sampler=optuna.samplers.GridSampler({"n_clusters": search_space}, seed=42),
        direction="minimize",
        study_name="optimisation",
    )
    optuna_objective = OptunaKMeansObjective(
        gdf=gdf,
        lat_col=lat_col,
        lon_col=lon_col,
        weight_col=weight_col,
        projected_epsg=projected_epsg,
        min_n_clusters=min(search_space),
        max_n_clusters=max(search_space),
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


def _get_search_space(
    gdf: gpd.GeoDataFrame,
    weight_col: str,
    desired_cluster_weight: Union[float, int],
    sample_size: int,
) -> list[int]:
    """Generates values of n_clusters for Optuna to try."""

    # get required variables
    total_weight = gdf[weight_col].sum()
    expected_n_clusters = max(int(total_weight / desired_cluster_weight), 1)
    n_points = len(gdf)

    # get values picked from normal distribution around "desired_cluster_weight"
    search_space = get_normal_samples(
        n_points=n_points,
        expected_n_clusters=expected_n_clusters,
        sample_size=sample_size,
    )

    return search_space


def get_normal_samples(
    n_points: int,
    expected_n_clusters: int,
    sample_size: int,
    scale_factor: int = 3,
):
    """
    Get a sample of n_clusters options from a normal distribution around
    expected_n_clusters.

    Notes:
    - If expected_n_clusters is greater than n_points, it is set to n_points.
    - If expected_n_clusters is less than 1, it is set to 1.
    - Sigma of the distribution is proportional to `expected_n_clusters` (as in
        `expected_n_clusters/scale_factor`) to capture the idea that higher
        `expected_n_clusters`
        are probably less accurate and we should try a wider range of possible values.
    - The scale_factor is set to 3 to get a distribution that is not too narrow nor too wide.
    - Number of samples is not guaranteed to be `sample_size`, since in edge cases the function will
        filter out all options with a probability of 0 to avoid errors in np.random.choice().
    """

    if expected_n_clusters > n_points:
        expected_n_clusters = n_points
    if expected_n_clusters < 1:
        expected_n_clusters = 1

    # get list of all n_clusters options
    all_options = np.arange(1, n_points + 1)

    # get normal distribution around expected_n_clusters to pick from
    mu = expected_n_clusters
    sigma = expected_n_clusters / scale_factor
    p = norm.pdf(all_options, loc=mu, scale=sigma)
    # normalize the probabilities as required by np.random.choice()
    p_scaler = 1 / p.sum()
    p = p * p_scaler

    # filter out all options with a probability of 0 to avoid errors in np.random.choice()
    non_zero_filter = p > 0
    non_zero_p = p[non_zero_filter]
    non_zero_options = all_options[non_zero_filter]

    # select options
    sample_size = min(len(non_zero_options), sample_size)
    selected_choices = np.random.choice(
        non_zero_options, size=sample_size, p=non_zero_p, replace=False
    )
    selected_choices = np.sort(selected_choices)

    # convert to normal ints
    selected_choices = selected_choices.tolist()

    return selected_choices


class OptunaKMeansObjective:
    """
    Class to create the objective function for Optuna to use to find the optimal number
    of clusters for kmeans.

    Steps:
    - Fits a KMeans model to the data with the n_clusters suggested by Optuna
    - Returns the score of the clustering as per compute_clustering_score()
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

        return compute_clustering_score(
            gdf_w_clusters=gdf_w_clusters,
            weight_col=self.weight_col,
            projected_epsg=self.projected_epsg,
            target_weight=self.target_weight,
            target_radius=self.target_radius,
            weight_importance_factor=self.weight_importance_factor,
        )


def compute_clustering_score(
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
        ids = create_ids(n_clusters, cluster_id_prefix)
        id_replace_dict = dict(zip(range(n_clusters), ids))
        clusters = [id_replace_dict[c] for c in clusters]

    return clusters
