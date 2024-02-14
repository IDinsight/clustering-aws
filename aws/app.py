import json
import time
from subprocess import call
import awswrangler as wr
import geopandas as gpd

from clustering.kmeans import get_multipass_optimised_clusters

# admin
id_col = "grid_id"
lat_col = "Lat"
lon_col = "Lon"
weight_col = "population"
projected_epsg = 3121  # philippines
# general
# desired_cluster_radius = dynamically set based on the "urban" column
desired_cluster_weight = 240
weight_importance_factor = 1
n_jobs = 1  # ALERT: Has to be 1 for AWS Lambdas!
n_passes = 2
# first pass
initial_max_trials = 100
# subsequent passe(es)
subsequent_max_trials = 30
max_cluster_weight = 300


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
        # set desired_cluster_radius parameter based on the "urban" column
        if gdf_for_cluster["urban"].iloc[0]:
            desired_cluster_radius = 1000
        else:
            desired_cluster_radius = 2000
        # run clustering
        gdf_w_clusters = get_multipass_optimised_clusters(
            gdf=gdf_for_cluster,
            lat_col=lat_col,
            lon_col=lon_col,
            weight_col=weight_col,
            projected_epsg=projected_epsg,
            desired_cluster_weight=desired_cluster_weight,
            desired_cluster_radius=desired_cluster_radius,
            weight_importance_factor=weight_importance_factor,
            initial_max_trials=initial_max_trials,
            n_jobs=n_jobs,
            n_passes=n_passes,
            subsequent_max_trials=subsequent_max_trials,
            max_cluster_weight=max_cluster_weight,
            show_progress_bar=False,
            return_type="geodataframe",
        )
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
