from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_val_score
from sklearn import metrics
import numpy as np


class Resampler:
    def __init__(self, model, resampler):
        """
        Up-sample the minority class in a dataset using the input resampler. The input resampler should be a sklearn
        imbalance model from the SMOTE family
        """
        self.model = model
        self.resampler = resampler
        self.best_params = {}

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Performs CV on the data to determine hyper-parameters of the resampler
        """
        # define a range for the number of neighbors
        k_neighbors_range = range(2, 7)
        best_score = 0

        # for each value of k_neighbors, resample the data and then evaluate the metric of interest using CV with the
        # model passed to the resampler
        for k in k_neighbors_range:
            over_sampler = resampler(k_neighbors=k)
            X_res, y_res = over_sampler.fit_resample(X, y)
            scores = cross_val_score(self.model, X_res, y_res, cv=5, scoring='f1_macro')
            mean_score = np.mean(scores)
            if mean_score > best_score:
                best_score = mean_score
                best_k = k
        self.best_params['k_neighbors'] = best_k

    def fit_resample(self, X: pd.DataFrame, y: pd.Series):
        self.fit(X,y)
        # resample with the best k and return the resampled data
        best_over_sampler = resampler(k_neighbors=self.best_params['k_neighbors'])
        X_res, y_res = best_over_sampler.fit_resample(X, y)
        return X_res, y_res


