import numpy as np
import argparse
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

def sigmoid_activation(x):
	return 1.0/(1+np.exp(-x))

def predict(x,w):
	pre=sigmoid_activation(x.dot(w))
#for binary
	if pre<=0.5:
		pre=0
	else:
		pre=1
return pre

ar=argparse.ArgumentParser()
ar.add_argument("-e","--epochs",type=float,default=1)
ar.add_argument("-a","--alpha",type=float,default=0.01)
args=vars(ar.parse_args())


