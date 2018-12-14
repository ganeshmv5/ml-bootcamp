#!/usr/bin/env python2.7

import os
import random
import pandas as pd
import numpy as np
import copy
import requests
import json
from PIL import Image

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

from grpc.beta import implementations

from mnist import MNIST # pylint: disable=no-name-in-module
import argparse

TF_MODEL_SERVER_HOST = os.getenv("TF_MODEL_SERVER_HOST", "127.0.0.1")
TF_MODEL_SERVER_PORT = int(os.getenv("TF_MODEL_SERVER_PORT", 9000))
TF_DATA_DIR = os.getenv("TF_DATA_DIR", "/tmp/data/")
#TF_MNIST_IMAGE_PATH = os.getenv("TF_MNIST_IMAGE_PATH", None)
TF_MNIST_TEST_IMAGE_NUMBER = int(os.getenv("TF_MNIST_TEST_IMAGE_NUMBER", -1))


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description='Arguments to be passed for \
                                     row index')
    PARSER.add_argument("-i", "--index",
                        type=int,
                        help='Specify the sample row index',
                        required=True)
    final_data = pd.read_csv("test_features.csv")
    args = PARSER.parse_args()
    index = args.index
    data_array = np.array(final_data)
    one_list = data_array[index]
    print one_list
    
    channel = implementations.insecure_channel(
        TF_MODEL_SERVER_HOST, TF_MODEL_SERVER_PORT)
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

    request = predict_pb2.PredictRequest()
    request.model_spec.name = "mnist"
    request.model_spec.signature_name = "serving_default"
    request.inputs['data'].CopyFrom(
        tf.contrib.util.make_tensor_proto(one_list,shape=[1,177],dtype=tf.float32))
        
    result = stub.Predict(request, 10.0)  # 10 secs timeout

    print(result)
