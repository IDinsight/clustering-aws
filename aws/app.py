import time

import awswrangler as wr
import geopandas as gpd

# from gridsample.clustering import make_clusters
from gridsample.clustering.kmeans import (
    custom_kmeans,
    run_optuna_kmeans,
    kmeans_optuna_secondpass,
)


def download_data(bucket, filename):
    wr.s3.download(path=f"s3://{bucket}/{filename}", local_file=f"/tmp/{filename}")

    gdf = gpd.read_parquet(f"s3://{bucket}/{filename}")

    return gdf


def cluster_data(gdf_for_cluster):

    # select parameters
    if gdf_for_cluster["urban"].iloc[0]:
        desired_cluster_radius = 1000
    else:
        desired_cluster_radius = 2000
    desired_cluster_weight = 240
    minimum_weight = 180
    maximum_weight = 300

    # # inhouse clustering
    # clusters, distance_matrix = make_clusters(
    #     df=gdf_for_cluster,
    #     id_col="grid_id",
    #     lat_col="Lat",
    #     lon_col="Lon",
    #     weight_col="population",
    #     radius=desired_cluster_radius,
    #     minimum_weight=minimum_weight,
    #     maximum_weight=maximum_weight,
    #     sufficient_condition="minimum_weight",
    #     random_seed=42,
    #     also_return_distance_matrix=True,
    # )
    # gdf_w_clusters = gdf_for_cluster.merge(clusters, on="grid_id")
    # gdf_w_clusters = gdf_w_clusters.sort_values(by="cluster_id")
    # gdf_w_clusters.to_parquet(f"/tmp/{output_filename}")

    # optimised kmeans
    study = run_optuna_kmeans(
        gdf=gdf_for_cluster,
        id_col="grid_id",
        lat_col="Lat",
        lon_col="Lon",
        weight_col="population",
        desired_cluster_weight=desired_cluster_weight,
        desired_cluster_radius=desired_cluster_radius,
        epsg=3121,
        n_trials=100,
        show_progress_bar=False,
    )
    clusters = custom_kmeans(
        df=gdf_for_cluster,
        n_clusters=study.best_params["n_clusters"],
        id_col="grid_id",
        lat_col="Lat",
        lon_col="Lon",
        weight_col="population",
    )
    gdf_w_clusters = gdf_for_cluster.merge(clusters, on="grid_id")
    gdf_w_clusters.sort_values(by="cluster_id", inplace=True)

    # second pass
    gdf_w_clusters_doublepass = kmeans_optuna_secondpass(
        gdf_w_clusters,
        cutoff_weight=maximum_weight,
        desired_cluster_weight=desired_cluster_weight,
        desired_cluster_radius=desired_cluster_radius,
        id_col="grid_id",
        lat_col="Lat",
        lon_col="Lon",
        weight_col="population",
        epsg=3121,
        n_trials=100,
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

    # CLUSTERING
    gdf_w_clusters = cluster_data(gdf_for_cluster)
    gdf_w_clusters.to_parquet(f"/tmp/{output_filename}")

    wr.s3.upload(
        local_file=f"/tmp/{output_filename}",
        path=f"s3://{output_bucket}/{output_filename}",
    )

    # clean up because some resources are shared across executions
    from subprocess import call

    call("rm -rf /tmp/*", shell=True)
