import numpy as np
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix, roc_curve, roc_auc_score, auc, accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import PreprocessingTransformer as ppt
import AnomalyDetector as ad
import Resampler as rs
import CDD as cdd
from visualization import get_roc


class Pipeline:
    def __init__(self,
                 model,
                 categorical_features: list = [],
                 numerical_features: list = [],
                 max_discarded_samples_ratio=0.05,
                 anomaly_detector_bypass=False,
                 data_feature_composition="mixed",
                 resampler_bypass=False,
                 over_sample_all_classes=False,
                 verbosity=False):
        """
        :param model: evaluation model
        :param categorical_features: list of categorical columns names
        :param numerical_features: list of numerical columns names
        :param max_discarded_samples_ratio: the maximal percentage of train samples that can be discarded by the
        anomaly detection step.
        :param anomaly_detector_bypass: if true, do not remove outliers
        :param data_feature_composition: num_only, cat_only or mixed. reveals the feature composition of the data
        :param resampler_bypass: if true, do not resample the data
        :param over_sample_all_classes: if true, try to add samples to all classes.

        """
        self.model = model
        self.transformer = ppt.PreprocessingTransformer(categorical_features=categorical_features, numerical_features=numerical_features)
        self.anomaly_detector = ad.AnomalyDetector(max_discarded_samples_ratio=max_discarded_samples_ratio, bypass=anomaly_detector_bypass, verbosity=verbosity)
        self.resampler = rs.Resampler(model=model, feature_composition=data_feature_composition,over_sample_all=over_sample_all_classes)
        self.cdd = cdd.CDD()
        self.verbosity = verbosity
        self.anomaly_detector_bypass = anomaly_detector_bypass
        self.resampler_bypass = resampler_bypass

    def fit_predict(self, X: pd.DataFrame, y: pd.Series):
        """
        This function corresponds to the development phase in the pipe. Performs the following steps:
        1. Split the data to train an test sets. Fits the transformer on the train set, and transforms both sets
        2. Fits an anomaly detection model to the transformed train set, and removes abnormal samples.
        3. Fits a resampler to the train set, resamples the minority classes and produces a balanced set.
        4. Fits the model to the balanced train set.
        :param X: features data frame
        :param y: target value
        """
        # step 1 - split and transform
        X_train, X_test, y_train, y_test = self.transformer.split_fit_transform(X, y)
        # get the names of the categorial features - we'll need it for the resampling step
        cat_cols_names = self.transformer.fitted_categorical_columns
        cat_features_idx = [X_train.columns.get_loc(c) for c in cat_cols_names if c in X_train]

        # move to numpy arrays
        X_train = X_train.to_numpy()
        X_test = X_test.to_numpy()
        # remove outliers
        X_reduced, y_reduced = self.anomaly_detector.fit_predict(X_train, y_train)
        # resampling
        X_res, y_res = self.resampler.fit_resample(X_reduced, y_reduced, categorical_features=cat_features_idx)
        # fitting the input model to the training set, make prediction on the test set and report classification
        self.model.fit(X_res, y_res)
        y_pred = self.model.predict(X_test)
        # Create and print confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1], normalize='true')
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["bad", "good"])
        disp.plot()
        plt.show()
        # classification report
        print(classification_report(y_test, y_pred))

        # Predict probabilities target variables y for test data
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        get_roc(y_test, y_pred_proba)

        # cache the test set (from the development phase) for future drift detection
        self.cdd.add_sample(y_pred_proba)


    def predict(self, X: np.ndarray, X_ref: np.ndarray=None):
        """
        This function corresponds to the deployment phase in the pipe. It is utilizing a fitted pipe to predict for new data
        Performs the following steps:
        1. Transforms the input data per the transformation learned in the fitting phase
        2. Provides prediction from the trained model.
        3. If reference data is given, detects concept drift between the data and reference data
        :param X: features data frame
        :param X_ref: reference data
        """

        X_transformed,_ = self.transformer.transform(X)
        X_transformed = X_transformed.to_numpy()
        y_pred = self.model.predict(X_transformed)
        y_pred_proba = self.model.predict_proba(X_transformed)[:, 1]
        ks_stat_res, ks_pval_res = self.cdd.evaluate(y_pred_proba)
        return y_pred,ks_stat_res, ks_pval_res