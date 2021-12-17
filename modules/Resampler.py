from imblearn.over_sampling import SMOTE,SMOTENC,SMOTEN
from sklearn.model_selection import cross_val_score
from sklearn import metrics
import numpy as np

features_to_resampler_mapping = {"num_only": SMOTE, "cat_only": SMOTEN, "mixed": SMOTENC}

class Resampler:
    def __init__(self, model, categorical_features, feature_composition="mixed"):
        """
        Up-sample the minority class in a dataset using the input resampler. The input resampler should be a sklearn
        imbalance model from the SMOTE family
        """
        self.model = model
        if feature_composition not in features_to_resampler_mapping:
            raise ValueError("Unsupported feature composition type")
        self.resampler = features_to_resampler_mapping[feature_composition]
        if feature_composition == 'mixed':
            self.best_params = {"categorical_features": categorical_features}
        else:
            self.best_params = {}
        

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Performs CV on the data to determine hyper-parameters of the resampler
        """
        # define a range for the number of neighbors
        k_neighbors_range = range(2, 7)
        best_score = 0

        # for each value of k_neighbors, resample the data and then evaluate the metric of interest using CV with the
        # model passed to the resampler
        for k in k_neighbors_range:
            
            self.best_params['k_neighbors'] = k
            over_sampler = self.resampler(**self.best_params)
            X_res, y_res = over_sampler.fit_resample(X, y)
            scores = cross_val_score(self.model, X_res, y_res, cv=5, scoring='f1_macro')
            mean_score = np.mean(scores)
            if mean_score > best_score:
                best_score = mean_score
                best_k = k
        self.best_params['k_neighbors'] = best_k

    def fit_resample(self, X: np.ndarray, y: np.ndarray):
        self.fit(X,y)
        # resample with the best k and return the resampled data
        best_over_sampler = self.resampler(**self.best_params)
        X_res, y_res = best_over_sampler.fit_resample(X, y)
        return X_res, y_res


