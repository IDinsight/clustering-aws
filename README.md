# Clustering + AWS Lambdas

This repo holds the code for:

- Our latest optimised-k-means clustering algorithm
- Running clustering on parallel AWS Lambdas

## AWS Architecture

<p align="center">
<img src="./aws_diagram.png" alt="AWS Architecture" width=700/>
</p>

Components:

- **S3 buckets**
  - One "input_bucket" for files before clustering
  - One "output_bucket" to save clustered files to

- **Kickoff Lambda**: For sending filenames to the Step Function to distribute across clustering Lambdas
  - Code: `aws/kickoff_lambda.py`. Copy this to a new Lambda.
  - Inputs:
    - S3 input bucket name
    - S3 output bucket name
    - No. files to send for processing

  - Output
        - List of filenames to process

- **Step Function**: For orchestrating the running of one Clustering Lambda per file
  - Code: `aws/state_machine.asl.json`. Copy this to a new Step Function state machine.
  - Inputs:
    - S3 input bucket name
    - S3 output bucket name
    - List of filenames to process automatically sent by Kickoff Lambda
    - Max number of Lambdas to run at once
  - Outputs:
    - Kicks off Clustering Lambdas with the following payload:
      - `{filename:"...", input_bucket:"...", output_bucket:"..."}`

- **Clustering Lambda**: For clustering an input file and saving the output
  - Code: `Dockerfile` and `aws/app.py`.  See "Clustering Lambda Setup" section below for details.
  - Inputs:
    - The following payload `{filename:"...", input_bucket:"...", output_bucket:"..."}`
  - Outputs:
    - Saves clustered file to the output bucket

## Clustering Lambda Setup

### Docker

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

### Push Docker image to ECR

Example - ECR repo URL is 865894278225.dkr.ecr.ap-south-1.amazonaws.com.

Tag the image with:

    docker tag hpls-r3-kmeans:latest 865894278225.dkr.ecr.ap-south-1.amazonaws.com/hpls-r3-kmeans:latest

Login with:

    aws ecr get-login-password --region ap-south-1 --profile <YOUR-AWS-PROFILE> | docker login --username AWS --password-stdin 865894278225.dkr.ecr.ap-south-1.amazonaws.com

Note - follow these [docs](https://docs.aws.amazon.com/cli/latest/userguide/sso-configure-profile-token.html) if you're not authenticated or run into issues here or in the next step.

Push to ECR with:

    docker push 865894278225.dkr.ecr.ap-south-1.amazonaws.com/hpls-r3-kmeans:latest

### Create Lambda

Create a Lambda function from Docker image, select correct image and load. Test it with a suitable payload (see testing locally above).

## Clustering Lambda Performance

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

- Input/output buckets are currently hardcoded into:
  - The kickoff Lambda function definition. See `aws/kickoff_lambda.py`.
  - The Step Function inside Parallel Process's Payload editor. Also see `aws/state_machine.asl.json`

## To do

- Clean up
- Add reasoning why we use a miniconda Docker image base and add Lambda functionality on top (and not an official Lambda image)
- Move all clustering-related code here and keep `gridsample` for data processing and pipeline code.

Contact: Amir Emami (@amiraliemami)
