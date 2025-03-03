"""
Write your unit tests here. Some tests to include are listed below.
This is not an exhaustive list.

- check that prediction is working correctly
- check that your loss function is being calculated correctly
- check that your gradient is being calculated correctly
- check that your weights update during training
"""

# Imports
import pytest
import numpy as np
from regression import (logreg, utils)

def test_prediction():
	log_model = logreg.LogisticRegressor(num_feats=6, learning_rate=0.00001, tol=0.01, max_iter=10, batch_size=10)
	log_model.W = np.array([1,2])
	x = np.array([[2,2],[-1,3]])
	pred = log_model.make_prediction(x)
	expected = np.array([0.9975,0.9933])
	for i in range(2):
		assert abs(pred[i] - expected[i]) < 0.001

def test_loss_function():
	log_model = logreg.LogisticRegressor(num_feats=6, learning_rate=0.00001, tol=0.01, max_iter=10, batch_size=10)
	y_true = [0.,1.]
	y_pred = [0.1,0.6]
	loss = log_model.loss_function(y_true,y_pred)
	expected = 0.1338
	assert abs(expected - loss) < 0.001

def test_gradient():
	log_model = logreg.LogisticRegressor(num_feats=6, learning_rate=0.00001, tol=0.01, max_iter=10, batch_size=10)
	log_model.W = np.array([1,2])
	x = np.array([[2,2],[-1,3]])
	y_pred = log_model.make_prediction(x)
	y_true = [0.,1.]
	gradient = log_model.calculate_gradient(y_true,x,y_pred)
	expected = np.array([1.00085,0.98745])
	for i in range(2):
		assert abs(gradient[i] - expected[i]) < 0.0001

def test_training():
	log_model = logreg.LogisticRegressor(num_feats=6, learning_rate=1, tol=0.01, max_iter=10, batch_size=10)
	log_model.W = np.array([1,2])
	x = np.array([[2,2],[-1,3]])
	y_pred = log_model.make_prediction(x)
	y_true = [0.,1.]
	gradient = log_model.calculate_gradient(y_true,x,y_pred)
	new_W = log_model.W - log_model.lr * gradient
	assert new_W[0] - (-0.00085) < 0.0001
	assert new_W[1] - 1.01255 < 0.0001