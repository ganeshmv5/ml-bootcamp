{
  global: {},
  components: {
    // Component-level parameters, defined initially from 'ks prototype use ...'
    // Each object below should correspond to a component in the components/ directory
    centraldashboard: {
      image: 'gcr.io/kubeflow-images-public/centraldashboard:v0.3.0',
      name: 'centraldashboard',
    },
    "tf-job-operator": {
      cloud: 'null',
      deploymentNamespace: 'null',
      deploymentScope: 'cluster',
      name: 'tf-job-operator',
      namespace: 'null',
      tfDefaultImage: 'null',
      tfJobImage: 'gcr.io/kubeflow-images-public/tf_operator:v0.3.0',
      tfJobUiServiceType: 'ClusterIP',
      tfJobVersion: 'v1alpha2',
    },
    "nfs-server": {
      name: 'nfs-server',
      namespace: 'null',
    },
    "nfs-volume": {
      capacity: '1Gi',
      mountpath: '/',
      name: 'nfs',
      namespace: 'kubeflowteam3',
      nfs_server_ip: '10.35.240.201',
      storage_request: '1Gi',
    },
    "tf-mnistjob": {
      args: 'null',
      envs: 'TF_DATA_DIR=/mnt/data,TF_EXPORT_DIR=/mnt/export,TF_MODEL_DIR=/mnt/model',
      image: 'ganeshmv/team3-demo',
      image_gpu: 'null',
      image_pull_secrets: 'null',
      name: 'tf-mnistjob',
      namespace: 'null',
      num_gpus: 0,
      num_masters: 1,
      num_ps: 1,
      num_workers: 1,
      volume_mount_path: 'null',
    },
  },
}