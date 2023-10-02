import numpy as np
from tqdm import tqdm


class PCA:
    def __init__(self, n_components: int) -> None:
        self.n_components = n_components
        self.components = None

    def fit(self, X) -> None:
        # fit the PCA model
        # shift the data such that mean is zero
        X = X - np.mean(X, axis=0)
        # find the covarience matrix
        convarience_matrix = np.cov(X, rowvar=False)
        # find the eigne value and eigen vectors
        eigen_values, eigen_vectors = np.linalg.eig(convarience_matrix)
        # find the index of sorted eigen values
        sorted_index = np.argsort(eigen_values)[::-1]
        # get eigrn values in sorted manner
        sorted_eigenvalue = eigen_values[sorted_index]
        top_K_sorted_eigenvalues = sorted_eigenvalue[0:self.n_components]
        print(sum(top_K_sorted_eigenvalues))
        # get eigrn vectors in sorted manner
        sorted_eigenvectors = eigen_vectors[:, sorted_index]
        # store the first n_components eigrn vectors in class variable
        self.components = sorted_eigenvectors[:, 0:self.n_components]
        # raise NotImplementedError

    def transform(self, X) -> np.ndarray:
        # transform the data
        # shift the data such that mean is zero
        X = X - np.mean(X, axis=0)
        return np.dot(X, self.components)
        raise NotImplementedError

    def fit_transform(self, X) -> np.ndarray:
        # fit the model and transform the data
        self.fit(X)
        return self.transform(X)


class SupportVectorModel:
    def __init__(self) -> None:
        self.w = None
        self.b = None

    def _initialize(self, X) -> None:
        self.w = np.zeros(len(X[1]))
        self.b = 0
        # initialize the parameters
        # pass

    def fit(
        self, X, y,
        learning_rate: float,
        num_iters: int,
        C: float = 1.0,
    ) -> None:
        self._initialize(X)
        # fit the SVM model using stochastic gradient descent
        for i in tqdm(range(1, num_iters + 1)):
          idx = np.random.randint(len(X))
          z = y[idx]*(np.dot(X[idx],self.w) + self.b)
          if z >= 1 :
            self.w = self.w - learning_rate*self.w
          else :
            self.w = self.w - learning_rate*(self.w - C*y[idx]*X[idx])
            self.b = self.b - learning_rate*(-1*C*y[idx])

    def predict(self, X) -> np.ndarray:
        # make predictions for the given data
        raise NotImplementedError

    def accuracy_score(self, X, y) -> float:
        # compute the accuracy of the model (for debugging purposes)
        return np.mean(self.predict(X) == y)


class MultiClassSVM:
    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.models = []
        for i in range(self.num_classes):
            self.models.append(SupportVectorModel())

    def fit(self, X, y, **kwargs) -> None:
        # first preprocess the data to make it suitable for the 1-vs-rest SVM model
        for i in range(self.num_classes) :
          Y_1vsR = []
          for idx,y_i in enumerate(y) :
            if y_i == i:
              Y_1vsR.append(1)
            else :
              Y_1vsR.append(-1)
          self.models[i].fit(X,Y_1vsR,kwargs['learning_rate'],kwargs['num_iters'],kwargs['C'])    
        # then train the 10 SVM models using the preprocessed data for each class
        # raise NotImplementedError

    def predict(self, X) -> np.ndarray:
        # pass the data through all the 10 SVM models and return the class with the highest score
        pred_y = []
        for i,x_i in enumerate(X) :
          pred = []
          for j in range(len(self.models)) :
            pred.append(np.dot(x_i,self.models[j].w) + self.models[j].b)
          pred_y.append(np.argmax(pred))
        return pred_y
        raise NotImplementedError

    def accuracy_score(self, X, y) -> float:
        return np.mean(self.predict(X) == y)

    def precision_score(self, X, y) -> float:
        total_predicted_class = np.zeros(10)
        corrected_predicted_class = np.zeros(10)
        predicted_array = self.predict(X)
        for i in range(0,len(X)) :
            if predicted_array[i] == y[i] :
                corrected_predicted_class[y[i]]+=1
            total_predicted_class[predicted_array[i]]+=1
        return np.mean(np.divide(corrected_predicted_class,total_predicted_class))
        raise NotImplementedError

    def recall_score(self, X, y) -> float:
        total_data_class = np.zeros(10)
        corrected_predicted_class = np.zeros(10)
        predicted_array = self.predict(X)
        for i in range(0,len(X)) :
            if predicted_array[i] == y[i] :
                corrected_predicted_class[y[i]]+=1
            total_data_class[y[i]]+=1
        return np.mean(np.divide(corrected_predicted_class,total_data_class))
        raise NotImplementedError

    def f1_score(self, X, y) -> float:
        pre_score = self.precision_score(X,y)
        rec_score = self.recall_score(X,y)
        return (2*rec_score*pre_score) / (rec_score+pre_score)
        raise NotImplementedError
