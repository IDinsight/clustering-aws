import json
import boto3

input_bucket = "r3-individual-precluster-barangays"
output_bucket = "r3-individual-postcluster-barangays"
step_function_arn = (
    "arn:aws:states:ap-south-1:865894278225:stateMachine:r3_hpls_processing"
)


def lambda_handler(event, context):
    print(event)

    sfn = boto3.client("stepfunctions")
    s3r = boto3.resource("s3")

    source = set(s3r.Bucket(input_bucket).objects.all())
    target = set(
        x.key.replace("clustered_", "") for x in s3r.Bucket(output_bucket).objects.all()
    )

    filenames = [{"filename": x.key} for x in source if x.key not in target]

    if len(filenames) == 0:
        return {
            "statusCode": 200,
            "body": json.dumps(
                f"No unprocessed files remaining. "
                f"No tasks left to kick off."
            ),
        }

    else:
        # Send max 7k unprocessed files per run.
        # We're limited by how big the HTTP request to the state machine can be.
        selected_remaining_filenames = filenames[:2000]

        # trigger sfn
        response = sfn.start_execution(
            stateMachineArn=step_function_arn,
            input=json.dumps(
                {
                    "filenames": selected_remaining_filenames,
                }
            ),
        )

        return {
            "statusCode": 200,
            "body": json.dumps(
                f"{len(filenames)} unprocessed files remaining. "
                f"Kicked off {len(selected_remaining_filenames)} processes..."
            ),
        }
