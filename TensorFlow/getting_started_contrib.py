__author__ = 'socurites'

import tensorflow as tf
import numpy as np

# Declare list of features of one real-valued feature
features = [tf.contrib.layers.real_valued_column("x", dimension=1)]

# Declare an estimator for linear regression
# An estimator is the front end to invoke training (fitting) and evaluation(inference)
estimator = tf.contrib.learn.LinearRegressor(feature_columns=features)

# Read training datasets
x = np.array([1., 2., 3., 4.])
y = np.array([0., -1., -2., -3.])
input_fn = tf.contrib.learn.io.numpy_input_fn({"x":x}, y, batch_size=4, num_epochs=1000)

# Invoke fit with training datasets and 1000 steps
estimator.fit(input_fn=input_fn, steps=1000)

# Evaluate the learned model with test datasets
estimator.evaluate(input_fn=input_fn)