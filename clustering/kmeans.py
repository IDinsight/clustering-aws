from multiprocessing import Pool
from typing import Optional, Union, Literal

import geopandas as gpd
import optuna
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
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
    - Ability to run multiple passes to break up oversized clusters (`max_passes`).
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
    max_passes : int, default=1
        Max number of passes to run to breakup oversized clusters.
        The algorithm will stop early if there are no oversized clusters left or if the
        number of oversized clusters does not change over 3 passes.
        - Setting this to a high number is roughly equivalent to running the algorithm
            until no more benefit is gained from further passes, but with the safety
            of a hard stop at `max_passes`.
        - If set to 1, only the initial pass is run.
    max_cluster_weight : float or int
        Must be set if max_passes > 1. Maximum weight above which a cluster is
        considered oversized and will be reclustered.
    subsequent_max_trials : int, default=100
        Must be set if max_passes > 1. Number of trials to run in subsequent passes.
    n_jobs : int, default=-1
        Number of parallel jobs to run. -1 means using all processors.
    show_progress_bar : bool, default=False
        Whether to show a progress bar for the optimisation process.

    Methods
    -------
    run(gdf, projected_epsg, weight_col=None):
        Runs the clustering process and returns the result.
    """

    def __init__(
        self,
        desired_cluster_weight: Union[float, int],
        desired_cluster_radius: Union[float, int],
        weight_importance_factor: Union[float, int] = 1,
        initial_max_trials: int = 100,
        max_passes: int = 1,
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
        self.max_passes = max_passes
        self.max_cluster_weight = max_cluster_weight
        self.subsequent_max_trials = subsequent_max_trials
        self.show_progress_bar = show_progress_bar
        self._check_input_params()

    def run(
        self,
        gdf: gpd.GeoDataFrame,
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
        if weight_col is None:
            weight_col = "weight"
            gdf.loc[:, "weight"] = [1] * len(gdf)

        # get projected gdf to use going forward
        gdf_projected = gdf[[weight_col, "geometry"]].to_crs(epsg=projected_epsg)
        gdf_projected.loc[:, "x"] = gdf_projected.centroid.x
        gdf_projected.loc[:, "y"] = gdf_projected.centroid.y

        # initial pass
        projected_gdf_w_clusters = get_optimised_clusters(
            gdf=gdf_projected,
            x_col="x",
            y_col="y",
            weight_col=weight_col,
            desired_cluster_weight=self.desired_cluster_weight,
            desired_cluster_radius=self.desired_cluster_radius,
            weight_importance_factor=self.weight_importance_factor,
            cluster_id_prefix="CLUSTER_",
            max_trials=self.initial_max_trials,
            n_jobs=self.n_jobs,
            show_progress_bar=self.show_progress_bar,
        )

        # add cluster_pass column to track which pass each cluster was formed in
        projected_gdf_w_clusters.loc[:, "cluster_pass"] = 1

        # passes 2 onwards...
        n_oversized_history = []
        for i in range(2, self.max_passes + 1):
            oversized_cluster_ids = self._get_oversized_clusters(
                gdf_w_clusters=projected_gdf_w_clusters, cutoff_weight=self.max_cluster_weight
            )
            n_oversized = len(oversized_cluster_ids)
            print(f"{n_oversized} oversized clusters left after {i-1} passes.")

            # stopping conditions
            if n_oversized == 0:
                print("No more oversized clusters found. Stopping early.")
                break
            elif (
                i >= 3
                and n_oversized == n_oversized_history[-1] == n_oversized_history[-2]
            ):
                print(
                    f"Number of oversized clusters {n_oversized} has not changed "
                    f"compared to the previous 2 runs. Stopping early."
                )
                break
            # continue if not stopping
            else:
                n_oversized_history.append(n_oversized)
                projected_gdf_w_clusters.loc[
                    projected_gdf_w_clusters["cluster_id"].isin(oversized_cluster_ids),
                    "cluster_pass",
                ] = i

                # run recluster
                recluster = ReCluster(
                    gdf_w_clusters=projected_gdf_w_clusters,
                    x_col="x",
                    y_col="y",
                    weight_col=weight_col,
                    desired_cluster_weight=self.desired_cluster_weight,
                    desired_cluster_radius=self.desired_cluster_radius,
                    weight_importance_factor=self.weight_importance_factor,
                    max_trials=self.subsequent_max_trials,
                    n_jobs=self.n_jobs,
                    show_progress_bar=self.show_progress_bar,
                    progress_bar_desc=f"Pass {i} ({n_oversized} oversized clusters)",
                )
                projected_gdf_w_clusters = recluster.run(
                    oversized_cluster_ids=oversized_cluster_ids
                )

        # merge back to original gdf
        gdf_w_clusters = gdf.merge(
            projected_gdf_w_clusters[["cluster_id", "cluster_weight", "cluster_pass"]],
            left_index=True,
            right_index=True,
        )

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
        if not isinstance(self.max_passes, int) or self.max_passes <= 0:
            raise ValueError("max_passes must be a positive integer.")
        if not isinstance(self.initial_max_trials, int) or self.initial_max_trials <= 0:
            raise ValueError("initial_max_trials must be a positive integer.")
        # check multi-pass parameters
        if self.max_passes > 1:
            if self.max_cluster_weight is None or self.max_cluster_weight <= 0:
                raise ValueError(
                    "max_cluster_weight must be set to a positive non-zero value "
                    "if max_passes > 1."
                )
            if (
                not isinstance(self.subsequent_max_trials, int)
                or self.subsequent_max_trials <= 0
            ):
                raise ValueError(
                    "subsequent_max_trials must be set to a positive integer "
                    "if max_passes > 1."
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
        y_col: str,
        x_col: str,
        weight_col: str,
        desired_cluster_weight: Union[float, int],
        desired_cluster_radius: Union[float, int],
        weight_importance_factor: Union[float, int] = 1,
        max_trials: int = 100,
        n_jobs: int = -1,
        show_progress_bar: bool = False,
        progress_bar_desc: str = "Reclustering",
    ):
        self.gdf_w_clusters = gdf_w_clusters
        self.y_col = y_col
        self.x_col = x_col
        self.weight_col = weight_col
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
            y_col=self.y_col,
            x_col=self.x_col,
            weight_col=self.weight_col,
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
            y_col=self.y_col,
            x_col=self.x_col,
            weight_col=self.weight_col,
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
        y_col: str,
        x_col: str,
        weight_col: str,
        desired_cluster_weight: Union[float, int],
        desired_cluster_radius: Union[float, int],
        weight_importance_factor: Union[float, int] = 1,
        max_trials: int = 100,
        n_jobs: int = -1,
    ):
        self.gdf_w_clusters = gdf_w_clusters
        self.y_col = y_col
        self.x_col = x_col
        self.weight_col = weight_col
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
            y_col=self.y_col,
            x_col=self.x_col,
            weight_col=self.weight_col,
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
    y_col: str,
    x_col: str,
    weight_col: str,
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
        y_col=y_col,
        x_col=x_col,
        weight_col=weight_col,
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
        y_col=y_col,
        x_col=x_col,
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
    y_col: str,
    x_col: str,
    weight_col: str,
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
        y_col=y_col,
        x_col=x_col,
        weight_col=weight_col,
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
    - Returns the score of the clustering as per compute_clustering_score()
    """

    def __init__(
        self,
        gdf: gpd.GeoDataFrame,
        y_col: str,
        x_col: str,
        weight_col: str,
        min_n_clusters: int,
        max_n_clusters: int,
        target_weight: Union[float, int],
        target_radius: Union[float, int],
        weight_importance_factor: Union[float, int] = 1,
    ) -> None:
        self.gdf = gdf
        self.min_n_clusters = min_n_clusters
        self.max_n_clusters = max_n_clusters
        self.y_col = y_col
        self.x_col = x_col
        self.weight_col = weight_col
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
            y_col=self.y_col,
            x_col=self.x_col,
            weight_col=self.weight_col,
            cluster_id_prefix=None,
        )
        gdf_w_clusters = self.gdf.copy()
        gdf_w_clusters.loc[:, "cluster_id"] = clusters

        return compute_clustering_score(
            gdf_w_clusters=gdf_w_clusters,
            weight_col=self.weight_col,
            target_weight=self.target_weight,
            target_radius=self.target_radius,
            weight_importance_factor=self.weight_importance_factor,
        )


def compute_clustering_score(
    gdf_w_clusters: gpd.GeoDataFrame,
    weight_col: str,
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
    y_col: str,
    x_col: str,
    weight_col: Optional[str] = None,
    cluster_id_prefix: Optional[str] = None,
) -> list:
    """Run KMeans and return list of cluster IDs. Optionally rename clusters."""

    # get data
    X = df[[x_col, y_col]].values
    sample_weight = df[weight_col]

    # fit
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=1024,
        n_init=1,
        reassignment_ratio=0.05,  # gets rid of small clusters!
        random_state=42,
    )
    clusters = list(kmeans.fit_predict(X=X, sample_weight=sample_weight))

    # rename if required
    if cluster_id_prefix is not None:
        ids = create_ids(n_clusters, cluster_id_prefix)
        id_replace_dict = dict(zip(range(n_clusters), ids))
        clusters = [id_replace_dict[c] for c in clusters]

    return clusters
