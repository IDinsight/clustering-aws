import time

# import awswrangler as wr
import geopandas as gpd

from clustering.kmeans import (
    custom_kmeans,
    get_oversized_clusters,
    run_optuna_kmeans_study,
    parallel_kmeans_secondpass,
)


def download_data(bucket, filename):
    gdf = gpd.read_file(f"data/input/{filename}")
    return gdf


def cluster_data(gdf_for_cluster):

    # set parameters
    id_col = "grid_id"
    lat_col = "Lat"
    lon_col = "Lon"
    weight_col = "population"
    # general
    epsg = 3121  # philippines
    # first pass
    firstpass_weight = 240
    firstpass_factor = 1
    firstpass_n_jobs = 2
    firstpass_n_trials = 96  # 8 * firstpass_n_jobs  # can also set to static 96
    # second pass
    secondpass_cutoff_weight = 300
    secondpass_weight = firstpass_weight
    secondpass_factor = firstpass_factor
    secondpass_python_n_jobs = 1
    secondpass_optuna_n_jobs = 2
    secondpass_n_trials = 24  # 2 * (
    #     secondpass_python_n_jobs * secondpass_optuna_n_jobs
    # )  #  can also set this to static 24

    # dynamic radius parameter
    if gdf_for_cluster["urban"].iloc[0]:
        firstpass_radius = 1000
        secondpass_radius = firstpass_radius
    else:
        firstpass_radius = 2000
        secondpass_radius = firstpass_radius
    print(f"Grids in barangay: {len(gdf_for_cluster)}")

    # first pass
    study_firstpass = run_optuna_kmeans_study(
        gdf=gdf_for_cluster,
        desired_cluster_weight=firstpass_weight,
        desired_cluster_radius=firstpass_radius,
        id_col=id_col,
        lat_col=lat_col,
        lon_col=lon_col,
        weight_col=weight_col,
        weight_importance_factor=firstpass_factor,
        epsg=epsg,
        n_trials=firstpass_n_trials,
        n_jobs=firstpass_n_jobs,
        show_progress_bar=True,
    )

    # proper run with the best n_cluster
    clusters = custom_kmeans(
        df=gdf_for_cluster,
        n_clusters=study_firstpass.best_params["n_clusters"],
        id_col=id_col,
        lat_col=lat_col,
        lon_col=lon_col,
        weight_col=weight_col,
    )
    gdf_w_clusters = gdf_for_cluster.merge(clusters, on=id_col)
    gdf_w_clusters = gdf_w_clusters.sort_values(by="cluster_id")

    # second pass
    oversized_cluster_ids = get_oversized_clusters(
        gdf_w_clusters=gdf_w_clusters, cutoff_weight=secondpass_cutoff_weight
    )
    n_oversized = len(oversized_cluster_ids)
    print(f"Oversized clusters: {n_oversized}")

    # add urban_guess column
    gdf_w_clusters = gdf_w_clusters.copy()
    gdf_w_clusters.loc[:, "dense_area_guess"] = 0
    gdf_w_clusters.loc[
        gdf_w_clusters["cluster_weight"] > secondpass_cutoff_weight, "dense_area_guess"
    ] = 1

    # run re-clustering
    gdf_w_clusters_doublepass = parallel_kmeans_secondpass(
        gdf_w_clusters=gdf_w_clusters,
        oversized_cluster_ids=oversized_cluster_ids,
        desired_cluster_weight=secondpass_weight,
        desired_cluster_radius=secondpass_radius,
        id_col=id_col,
        lat_col=lat_col,
        lon_col=lon_col,
        weight_col=weight_col,
        weight_importance_factor=secondpass_factor,
        epsg=epsg,
        n_trials=secondpass_n_trials,
        python_n_jobs=secondpass_python_n_jobs,
        optuna_n_jobs=secondpass_optuna_n_jobs,
    )
    gdf_w_clusters_doublepass = gdf_w_clusters_doublepass.sort_values(by="cluster_id")

    return gdf_w_clusters_doublepass


def handler(event, context):
    print(event)

    input_filename = event["filename"]
    input_bucket = "r3-hpls-presampling-barangays"
    output_bucket = "r3-hpls-clustered-barangays"
    output_filename = f"clustered_{input_filename}"

    try:
        gdf_for_cluster = download_data(input_bucket, output_filename)
    except Exception:
        # retry once if fails
        time.sleep(3)
        gdf_for_cluster = download_data(input_bucket, output_filename)

    print(gdf_for_cluster.head())

    # CLUSTERING
    gdf_w_clusters = cluster_data(gdf_for_cluster)
    gdf_w_clusters.to_parquet(f"/tmp/{output_filename}")

    # clean up because some resources are shared across executions
    from subprocess import call

    call("rm -rf /tmp/*", shell=True)
