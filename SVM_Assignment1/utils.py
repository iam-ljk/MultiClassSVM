import numpy as np
import pandas as pd
from typing import Tuple
from matplotlib import pyplot as plt
from tqdm import tqdm


def get_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	# load the data
	train_df = pd.read_csv('data/mnist_train.csv')
	test_df = pd.read_csv('data/mnist_test.csv')

	X_train = train_df.drop('label', axis=1).values
	y_train = train_df['label'].values

	X_test = test_df.drop('label', axis=1).values
	y_test = test_df['label'].values

	return X_train, X_test, y_train, y_test


def normalize(X_train, X_test) -> Tuple[np.ndarray, np.ndarray]:
	# normalize the training data
	for j in tqdm(range(0, len(X_train[0]))) :
		_min = 256
		_max = 0
		for i in range(0, len(X_train)) :
			_min = min(_min, X_train[i][j])
			_max = max(_max, X_train[i][j])
		# print("Done0")
		for i in range(0,len(X_train)) :		
			diff = _max - _min
			if diff != 0 :
				X_train[i][j] = (X_train[i][j] - _min) / diff
			else :
				X_train[i][j] = (X_train[i][j] - _min)
			X_train[i][j] = (2.0*X_train[i][j]) - 1.0
	# np.savetxt('./data/train.csv',X_train,header=",".join(str(x) for x in X_train[0]),delimiter=',',fmt='%f', comments='')
	# Normalioze the test data
	#####################################################################
	for j in tqdm(range(0, len(X_test[0]))):
		_min = 256
		_max = 0
		for i in range(0, len(X_test)) :
			_min = min(_min, X_test[i][j])
			_max = max(_max, X_test[i][j])
		# print("Done0")
		for i in range(0,len(X_test)) :		
			diff = _max - _min
			if diff != 0 :
				X_test[i][j] = (X_test[i][j] - _min) / diff
			else :
				X_test[i][j] = (X_test[i][j] - _min)
			X_test[i][j] = (2.0*X_test[i][j]) - 1.0
	# np.savetxt('./data/test.csv',X_test,header=",".join(str(x) for x in X_test[0]),delimiter=',',fmt='%f', comments='')
	######################################################################
	# load the data
	# print(X_train[12352][13])
	# train_df = pd.read_csv('data/train.csv')
	# test_df = pd.read_csv('data/test.csv')
	# X_train = train_df.values
	# X_test = test_df.values
	return X_train,X_test



	#######################################################################
	# print("DOne1")
	# plt.scatter(X_reduced[:,0],X_reduced[:,1],c=Y_train,cmap='viridis')
	# plt.show()
	# plt.savefig('fig.png')
	return X_train, X_test


def plot_metrics(metrics) -> None:
	# plot and save the results
	metrixDF = pd.DataFrame(metrics,columns=["K","Accuracy","Precesion","Recall","F1 score"])
	metrixDF.plot(y = [1,2,3,4],x = 0, kind = 'bar', figsize=(10,6))
	plt.savefig('diagram.png')
	plt.show()
	# raise NotImplementedError