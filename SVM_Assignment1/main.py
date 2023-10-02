from utils import get_data, plot_metrics, normalize
from model import MultiClassSVM, PCA
from typing import Tuple
from matplotlib import pyplot as plt
import numpy as np



def get_hyperparameters() -> Tuple[float, int, float]:
  # get the hyperparameters
  learning_rate = 0.0002
  nums_iters = 50000
  C = 10
  return learning_rate,nums_iters,C
  raise NotImplementedError


def main() -> None:
  # hyperparameters
  learning_rate, num_iters, C = get_hyperparameters()

  # get data
  X_train, X_test, y_train, y_test = get_data()

  # normalize the data
  X_train = X_train.astype('float32')
  X_test = X_test.astype('float32')
  X_train, X_test = normalize(X_train, X_test)

  metrics = []
  for k in [5, 10, 20, 50, 100, 200, 500]:
    # reduce the dimensionality of the data
    pca = PCA(n_components=k)
    PCA_X_train = pca.fit_transform(X_train)
    PCA_X_test = pca.transform(X_test)
    
    # create a model
    svm = MultiClassSVM(num_classes=10)

  #   # fit the model
    svm.fit(
        PCA_X_train, y_train, C=C,
        learning_rate=learning_rate,
        num_iters=num_iters,
    )

  #   # evaluate the model
    accuracy = svm.accuracy_score(PCA_X_test, y_test)
    precision = svm.precision_score(PCA_X_test, y_test)    
    recall = svm.recall_score(PCA_X_test, y_test)
    
    f1_score = svm.f1_score(PCA_X_test, y_test)
    # print(accuracy, precision, recall,f1_score)
    

    metrics.append((k, accuracy, precision, recall, f1_score))

    # print(f'k={k}, accuracy={accuracy}, precision={precision}, recall={recall}, f1_score={f1_score}')

  # # plot and save the results
  plot_metrics(metrics)


if __name__ == '__main__':
    main()
