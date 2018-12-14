#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""This showcases how simple it is to build image classification networks.

It follows description from this TensorFlow tutorial:
    https://www.tensorflow.org/versions/master/tutorials/mnist/pros/index.html#deep-mnist-for-experts
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import os
import sys
import numpy as np
import tensorflow as tf
import shutil
from tensorflow import keras

import pandas as pd
import copy
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import tag_constants, signature_constants, signature_def_utils_impl

# Configure model options
TF_DATA_DIR = os.getenv("TF_DATA_DIR", "/tmp/data/")
TF_MODEL_DIR = os.getenv("TF_MODEL_DIR", None)
TF_EXPORT_DIR = os.getenv("TF_EXPORT_DIR", "./mnist/")

def main(unused_args):
  tf.logging.set_verbosity(tf.logging.INFO)
  
  if not os.path.exists(TF_DATA_DIR):
      os.mkdir(TF_DATA_DIR)
  
  data = pd.read_csv("/opt/data_csv_filtered1.csv")
  data = data.fillna(0)
  data.drop(['egQDropPkts', 'bufferDrop'], axis=1)
  column_names = list(data.head(0))

  unique_streams = set()
  for val in data['srcIp']:
      unique_streams.add(val)

  unique_ts = set()
  for val in data['timestamp']:
      unique_ts.add(val)

  final_result = list(dict())
  vals_to_transpose = ('byteCount', 'pktCount', 'egQOcc')
  malicious_stm = ('mStream10', 'mStream20', 'mStream30')

  while len(unique_ts):
      temp_unique_streams = copy.deepcopy(unique_streams)
      temp_dict = dict()
      
      curr_ts = unique_ts.pop()
      temp_dict['timestamp'] = curr_ts

      test = data[data.timestamp == curr_ts]
      m_stream10 = list(test[malicious_stm[0]])
      m_stream20 = list(test[malicious_stm[1]])
      m_stream30 = list(test[malicious_stm[2]])

      for val in test[test.columns[1:7]].iterrows():
          temp = val[1]
          for each in vals_to_transpose:
              new_index = temp['srcIp'] + "_" + each
              temp_dict[new_index] = temp[each]
          temp_unique_streams.remove(temp['srcIp'])
      for strm in temp_unique_streams:
          for each in vals_to_transpose:
              new_index = strm + "_" + each
              temp_dict[new_index] = 0

      for strms in unique_streams:
          temp_dict[strms] = 0

      if 1.0 in m_stream10:
          temp_dict['mStream10'] = 1
      else:
          temp_dict['mStream10'] = 0

      if 1.0 in m_stream20:
          temp_dict['mStream20'] = 1
      else:
          temp_dict['mStream20'] = 0

      if 1.0 in m_stream30:
          temp_dict['mStream30'] = 1
      else:
          temp_dict['mStream30'] = 0

      final_result.append(temp_dict)

  final_data = pd.DataFrame(final_result)

  label_columns = list((data.loc[:, 'mStream10':'stream9']).head(0))
  label_columns

  final_data[:10]

  feature_columns = list(final_data.head(0))
  for col in label_columns:
      if col in feature_columns:
          feature_columns.remove(col)

  feature_data = final_data.drop(['timestamp'], axis=1)
  feature_columns.remove('timestamp')
  feature_data[0:2]

  label_data = final_data.drop(feature_columns, axis=1)
  label_data = label_data.drop(['timestamp'], axis=1)
  label_data[0:2]

  #Normalize the byte and packet counts to get so that all features are in the scale 0 to 1
  normalized_data = final_data[:]
  normalized_data = normalized_data.drop(['timestamp'], axis=1)

  for col in feature_columns:
      if "bytecount" in col.lower(): 
          normalized_data[col] = normalized_data[col]/(10*(10**9))
      elif "pktcount" in col.lower():
          normalized_data[col] = normalized_data[col]/820209
      elif "egqocc" in col.lower():
          normalized_data[col] = normalized_data[col]/100

  normalized_data[:10]

  # Splitting the data into Training and Testing
  # In order to test our algorithm, we'll split the data into a Training and a Testing set. The size of the testing set will be 10% of the total data.

  sample = np.random.choice(normalized_data.index, size=int(len(normalized_data)*0.9), replace=False)
  train_data, test_data = normalized_data.iloc[sample], normalized_data.drop(sample)

  print("Number of training samples is", len(train_data))
  print("Number of testing samples is", len(test_data))
  print(train_data[:10])
  print(test_data[:10])


  # Splitting the data into features and targets (labels)
  # Now, as a final step before the training, we'll split the data into features (X) and targets (y).

  # Separate data and one-hot encode the output
  # Note: We're also turning the data into numpy arrays, in order to train the model in Keras
  features = np.array(train_data.drop(label_columns, axis=1))
  targets = np.array(train_data.drop(feature_columns, axis=1))

  features_test = np.array(test_data.drop(label_columns, axis=1))
  targets_test = np.array(test_data.drop(feature_columns, axis=1))

  print(features[:2])
  print(targets[:2])

  # Building the model
  model = keras.Sequential()
  model.add(keras.layers.Dense(256, activation='sigmoid', input_shape=(177,)))
  model.add(keras.layers.Dropout(.2))
  model.add(keras.layers.Dense(128, activation='sigmoid'))
  model.add(keras.layers.Dropout(.2))
  model.add(keras.layers.Dense(64, activation='sigmoid'))
  model.add(keras.layers.Dropout(.1))
  model.add(keras.layers.Dense(59, activation='sigmoid'))

  # Compiling the model
  model.compile(loss='categorical_crossentropy',
                optimizer=tf.keras.optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0),
                metrics=['categorical_accuracy'])

  model.summary()

  # Training the model
  history = model.fit(features, targets, epochs=25, batch_size=100, verbose=2)
  # Scoring the model
  # Evaluating the model on the training and testing set
  score = model.evaluate(features, targets)
  print("\n Training Loss:", score[0])
  print("\n Training Accuracy:", score[1])
  score = model.evaluate(features_test, targets_test)
  print("\n Testing Loss:", score[0])
  print("\n Testing Accuracy:", score[1])

  print("Model.input - ", model.input)
  print("Model.output - ", model.output)

  tf.saved_model.simple_save(keras.backend.get_session(),
          TF_EXPORT_DIR + "/" + str(int(time.time())),
          inputs={'data': model.input},
          outputs={t.name:t for t in model.outputs})

if __name__ == '__main__':
  tf.app.run()

