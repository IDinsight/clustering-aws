import json
import time
from subprocess import call
import awswrangler as wr
import geopandas as gpd

from clustering.kmeans import (
    custom_kmeans,
    get_oversized_clusters,
    run_optuna_kmeans_study,
    kmeans_secondpass,
)

# admin variables
id_col = "grid_id"
lat_col = "Lat"
lon_col = "Lon"
weight_col = "population"
epsg = 3121  # philippines
# both passes
weight_importance_factor = 1
n_jobs = 2
# first pass
desired_weight = 240
firstpass_n_trials = 96
# second pass
secondpass_cutoff_weight = 300
secondpass_n_trials = 24


def cluster_data(gdf_for_cluster):

    # dynamic radius parameter
    if gdf_for_cluster["urban"].iloc[0]:
        desired_radius = 1000
    else:
        desired_radius = 2000
    print(f"Grids in barangay: {len(gdf_for_cluster)}")

    # first pass
    study_firstpass = run_optuna_kmeans_study(
        gdf=gdf_for_cluster,
        desired_cluster_weight=desired_weight,
        desired_cluster_radius=desired_radius,
        id_col=id_col,
        lat_col=lat_col,
        lon_col=lon_col,
        weight_col=weight_col,
        weight_importance_factor=desired_weight,
        epsg=epsg,
        n_trials=firstpass_n_trials,
        n_jobs=n_jobs,
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
    gdf_w_clusters.loc[:, "dense_area_guess"] = 0
    gdf_w_clusters.loc[
        gdf_w_clusters["cluster_weight"] > secondpass_cutoff_weight,
        "dense_area_guess",
    ] = 1

    if n_oversized > 0:
        # run re-clustering
        gdf_w_clusters = kmeans_secondpass(
            gdf_w_clusters=gdf_w_clusters,
            oversized_cluster_ids=oversized_cluster_ids,
            desired_cluster_weight=desired_weight,
            desired_cluster_radius=desired_radius,
            id_col=id_col,
            lat_col=lat_col,
            lon_col=lon_col,
            weight_col=weight_col,
            weight_importance_factor=weight_importance_factor,
            epsg=epsg,
            n_trials=secondpass_n_trials,
            n_jobs=n_jobs,
        )
        gdf_w_clusters = gdf_w_clusters.sort_values(by="cluster_id")

    return gdf_w_clusters


def download_data(bucket, filename):
    wr.s3.download(path=f"s3://{bucket}/{filename}", local_file=f"/tmp/{filename}")
    gdf = gpd.read_parquet(f"s3://{bucket}/{filename}")

    return gdf


def handler(event, context):
    print(event)

    input_filename = event["filename"]
    input_bucket = event["input_bucket"]
    output_bucket = event["output_bucket"]

    # Download file from S3
    output_filename = f"clustered_{input_filename}"

    try:
        print("Downloading...")
        gdf_for_cluster = download_data(input_bucket, input_filename)
    except Exception:
        try:
            print("Retrying download...")
            time.sleep(3)
            gdf_for_cluster = download_data(input_bucket, output_filename)
        except Exception as e:
            raise Exception(f"{input_filename} failed. Error: {e}")

    # Cluster
    try:
        print("Clustering...")
        gdf_w_clusters = cluster_data(gdf_for_cluster)
        gdf_w_clusters.to_parquet(f"/tmp/{output_filename}")
    except Exception as e:
        raise Exception(f"{input_filename} failed. Error: {e}")

    # Upload file to S3
    try:
        print("Uploading...")
        wr.s3.upload(
            local_file=f"/tmp/{output_filename}",
            path=f"s3://{output_bucket}/{output_filename}",
        )
    except Exception as e:
        raise Exception(f"{input_filename} failed. Error: {e}")

    # clean up because some resources are shared across executions
    call("rm -rf /tmp/*", shell=True)

    return {
        "statusCode": 200,
        "body": json.dumps(f"Finished clustering {input_filename}!"),
    }
