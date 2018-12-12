#!/usr/local/env bash

## Namespace to be used in k8s cluster for your application
NAMESPACE=kubeflowteam3

## Ksonnet app name
APP_NAME=mnist

## GITHUB version for official kubeflow components
KUBEFLOW_GITHUB_VERSION=v0.3.0-rc.3

## GITHUB version for ciscoai components
CISCOAI_GITHUB_VERSION=master

## Ksonnet environment name
KF_ENV=nativek8s

## Name of the NFS Persistent Volume
NFS_PVC_NAME=nfs
NFS_STORAGE=1Gi

## Used in training.bash
# Enviroment variables for mnist training job (See mnist_model.py)
TF_DATA_DIR=/mnt/data
TF_MODEL_DIR=/mnt/model
NFS_MODEL_PATH=/mnt/export
TF_EXPORT_DIR=${NFS_MODEL_PATH}

# If you want to use your own image,
# make sure you have a dockerhub account and change
# DOCKER_BASE_URL and IMAGE below.
#DOCKER_BASE_URL=gcr.io/cpsg-ai-demo
DOCKER_BASE_URL=ganeshmv
#IMAGE=${DOCKER_BASE_URL}/tf-mnist-demo:v1
IMAGE=${DOCKER_BASE_URL}/team3-demo
#docker build . --no-cache  -f Dockerfile -t ${IMAGE}
#docker push ${IMAGE}

# Used in portf.bash and webapp.bash
# If using without an application, source this file before using
PORT=9000
export TF_MODEL_SERVER_PORT=${PORT}

# Used in webapp.bash
#DOCKER_HUB=gcr.io
#DOCKER_USERNAME=cpsg-ai-demo
DOCKER_USERNAME=ganeshmv
#DOCKER_IMAGE=mnist-client
DOCKER_IMAGE=team3-client
WEBAPP_FOLDER=webapp

