import json
from subprocess import call
import awswrangler as wr
import pandas as pd


def download_data(bucket, filename):
    wr.s3.download(path=f"s3://{bucket}/{filename}", local_file=f"/tmp/{filename}")
    gdf = pd.read_parquet(f"s3://{bucket}/{filename}")

    return gdf


def handler(event, context):

    input_filename = event["filename"]
    input_bucket = event["input_bucket"]
    output_bucket = event["output_bucket"]

    # Download file from S3
    try:
        print("Downloading...")
        all_grids_gdf = download_data(input_bucket, input_filename)
    except Exception as e:
        raise Exception(e)

    # Separate file for each barangay
    psgc_list = all_grids_gdf["PSGC"].unique()
    for psgc in psgc_list:
        try:
            output_filename = f"{psgc}.parquet"
            all_grids_gdf[all_grids_gdf["PSGC"] == psgc].to_parquet(
                f"/tmp/{output_filename}"
            )
            wr.s3.upload(
                local_file=f"/tmp/{output_filename}",
                path=f"s3://{output_bucket}/{output_filename}",
            )
        except Exception as e:
            print(f"{psgc} failed: {e}")

    # clean up
    call("rm -rf /tmp/*", shell=True)

    return {"statusCode": 200, "body": json.dumps("Done my job!")}
