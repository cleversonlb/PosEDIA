# Codigo que sera importado para o AWS lamba para fazer a predicao utilizando uma REST API

import os
import io
import boto3
import json

ENDPOINT_NAME = "{sagemaker-scikit-learn-2022-11-22-05-49-56-689}"
runtime= boto3.client('runtime.sagemaker')

def lambda_handler(event, context):
    print("Received event: " + json.dumps(event, indent=2))
    
    data = json.loads(json.dumps(event))
    payload = data['data']
    print(payload)
    
    response = runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME,
                                       Body=json.dumps(payload))
    print(response)
    result = json.loads(response['Body'].read().decode())
    print(result)
    
    return result[0]
