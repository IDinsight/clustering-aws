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

    remaining_barangays = [{"filename": x.key} for x in source if x.key not in target]

    # batch to less than 7k
    print(f"Remaining barangays: {len(remaining_barangays)}")
    remaining_barangays = remaining_barangays[:7000]

    # trigger sfn
    response = sfn.start_execution(
        stateMachineArn=step_function_arn,
        input=json.dumps(
            {
                "filenames": remaining_barangays,
            }
        ),
    )

    return {
        "statusCode": 200,
        "body": json.dumps(f"Kicked off {len(remaining_barangays)} processes..."),
    }
