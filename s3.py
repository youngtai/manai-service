import boto3

s3 = boto3.client('s3')

bucket_objects = s3.list_objects_v2(
    Bucket='manai-test'
)

print(bucket_objects)