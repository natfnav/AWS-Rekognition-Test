import boto3

def verify_aws_credentials():
    boto3.Session(profile_name='fetch-drones')
    sts = boto3.client('sts')
    print(sts.get_caller_identity())