# Temporary Repo for Clustering on AWS Lambdas

This repo holds the code for running the latest optimised-k-means clustering algorithm (derived from the `gridsample` repo on 9th Feb 2024) on AWS Lambda for the HPLS project (round 3).

## Docker

Write your Dockerfile with these [AWS docs](https://docs.aws.amazon.com/lambda/latest/dg/python-image.html#python-image-clients) in mind.

Build with:

    docker build -t hpls-r3-kmeans:latest .

Prepare for emulating Lambdas locally:

    mkdir -p ~/.aws-lambda-rie && \
    curl -Lo ~/.aws-lambda-rie/aws-lambda-rie https://github.com/aws/aws-lambda-runtime-interface-emulator/releases/latest/download/aws-lambda-rie && \
    chmod +x ~/.aws-lambda-rie/aws-lambda-rie

Run locally with:

    docker run --env-file .env -d\
    -v ~/.aws-lambda-rie:/aws-lambda -p 9000:8080 \
    --entrypoint /aws-lambda/aws-lambda-rie \
    hpls-r3-kmeans:latest /opt/conda/bin/python -m awslambdaric app.handler

Note: If you want to actually access the S3 buckets, set the following in a .env file and load it with the `--env-file .env` flag when running `docker run`:

    AWS_ACCESS_KEY_ID=
    AWS_SECRET_ACCESS_KEY=

Test locally with:

    curl "http://localhost:9000/2015-03-31/functions/function/invocations" \
    -d '{"filename":"data_200902010.parquet", "input_bucket":"r3-hpls-presampling-barangays",  "output_bucket":"r3-hpls-clustered-barangays"}'

## Docker push to ECR instructions

Example - ECR repo URL is 865894278225.dkr.ecr.ap-south-1.amazonaws.com.

Tag the image with:

    docker tag hpls-r3-kmeans:latest 865894278225.dkr.ecr.ap-south-1.amazonaws.com/hpls-r3-kmeans:latest

Login with:

    aws ecr get-login-password --region ap-south-1 --profile <YOUR-AWS-PROFILE> | docker login --username AWS --password-stdin 865894278225.dkr.ecr.ap-south-1.amazonaws.com

Note - follow these [docs](https://docs.aws.amazon.com/cli/latest/userguide/sso-configure-profile-token.html) if you're not authenticated or run into issues here or in the next step.

Push to ECR with:

    docker push 865894278225.dkr.ecr.ap-south-1.amazonaws.com/hpls-r3-kmeans:latest

## AWS Lambda Performance

Example stats:

    | Grids     | Duration     | Billed Duration  | Max Memory Used |
    |-----------|--------------|------------------|-----------------|
    | 13691     | 8,547.17 ms  | 16,270 ms*       | 422 MB          | 
    | 69226     | 98,398.83 ms | 98,399 ms        | 518 MB          |
    | 101519    | 35,207.47 ms | 42,906 ms*       | 568 MB          | 
    | ^largest  |              |                  |                 |
*Roughly 7-second initialisation time included in billed duration

Based on this, I set the memory to 1024MB and timeout to 5mins.

## Notes

- Input/output buckets are currently hardcoded into the Step Function inside Parallel Process's Payload editor right now.

## To do

- Clean up
- Add reasoning why we use a miniconda Docker image base and add Lambda functionality on top (and not an official Lambda image)
- Move all clustering-related code here and keep `gridsample` for data processing and pipeline code.

Contact: Amir Emami (@amiraliemami)
