import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from scipy.stats import iqr
from scipy.spatial.distance import jensenshannon
import matplotlib.pyplot as plt


class AnomalyDetector:
    def __init__(self, max_discarded_samples_ratio=0.05, bypass=False, verbosity=False):
        """
        Per-class anomaly detection in a labeled dataset. The main functions are
        a. find an appropriate detector for the data
        b. fit the detector to the data and remove abnormal points. this process is done per class - i.e. each class data
        is treated as a separate dataset.

        """
        self.max_discarded_samples_ratio = max_discarded_samples_ratio
        self.anomaly_detector = None
        self.verbosity = verbosity
        self.bypass = bypass
        self.class_discarded_samples = []
        self.class_num_discarded_samples = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        This function examines two algorithms:
        a. Isolation Forest
        b. One-Class SVM
        and evaluate the detection based on cross scoring. i.e., it fits the detector to data from single class,
        and then evaluates the anomaly scores for data from all other classes. for reasonably separable data, the scores
        distribution for the fitted data should be very different from the scores distribution of the evaluated data.
        """
        # find all classes
        classes = np.unique(y)
        # define a list of detectors (meanwhile with default parameters)
        detectors = [IsolationForest(n_estimators=10*X.shape[1],max_samples=1.0, random_state=11,contamination=self.max_discarded_samples_ratio), OneClassSVM(nu=self.max_discarded_samples_ratio)]
        detector_scores = []
        if self.verbosity:
            fig, axs = plt.subplots(len(detectors), len(classes), figsize=(12, 12))
        for i, d in enumerate(detectors):
            js_div = 0
            for j, c in enumerate(classes):
                # separate samples corresponding to that label from all other samples
                class_idx = np.squeeze(np.argwhere(y == c))
                class_data = X[class_idx, :]

                other_class_idx = np.squeeze(np.argwhere(y != c))
                other_class_data = X[other_class_idx, :]
                # train an anomaly detector on the current class
                clf = d.fit(class_data)
                # get scores for this class
                this_class_scores = clf.decision_function(class_data) #this_class_scores = clf.score_samples(class_data)
                # get scores for all data not in this class
                other_class_scores = clf.decision_function(other_class_data) #other_class_scores = clf.score_samples(other_class_data)
                # create a unified support
                a = np.concatenate((this_class_scores, other_class_scores))
                bins = np.linspace(np.amin(a), np.amax(a), num=100)
                bin_width = (np.amax(a) - np.amin(a))/bins.size
                this_class_hist, _ = np.histogram(this_class_scores, bins=bins)
                other_class_hist, _ = np.histogram(other_class_scores, bins=bins)
                if self.verbosity:
                    t = "Anomaly scores distribution for detector "+d.__class__.__name__+" class "+str(c)
                    axs[i, j].bar((bins-bin_width/2)[:-1], this_class_hist/this_class_scores.size, width=0.8*bin_width, alpha=0.5, label=f"fitted class: {c}")
                    axs[i, j].bar((bins-bin_width/2)[:-1], other_class_hist/other_class_scores.size, width=0.8*bin_width, alpha=0.5, label="other classes")
                    axs[i, j].set_xlabel('Anomaly scores')
                    axs[i, j].set_ylabel('Count')
                    #axs[i, j].set_title(t)
                    axs[i, j].legend()

                # calculate the Jensen-Shannon divergence between the distributions
                js_div += jensenshannon(this_class_hist, other_class_hist)

            # record the number of correct predictions for this detector
            detector_scores.append(js_div)

        # get the best detector - we want the JS divergence to be maximal
        self._print("accumulated JS values: \n", self.verbosity)
        self._print(detector_scores, self.verbosity)
        self.anomaly_detector = detectors[np.argmax(detector_scores)]

    def fit_predict(self, X: np.ndarray, y: np.ndarray):
        print("Anomaly detection step")
        print("-------------------------------------------------------")
        if self.bypass:
            X_reduced = X
            y_reduced = y
            print("bypassing....")
        else:
            # find the best detector for this data
            self.fit(X, y)
            # go over the classes one by one
            classes = np.unique(y)
            for i, c in enumerate(classes):
                # separate samples corresponding to that label from all other samples
                class_idx = np.squeeze(np.argwhere(y == c))
                class_data = X[class_idx, :]
                class_labels = y[class_idx]
                # find out the maximal number of samples we can discard
                max_discarded_samples = int(np.floor(self.max_discarded_samples_ratio * class_data.shape[0]))
                # fit the anomaly detector to the class data
                det = self.anomaly_detector.fit(class_data)
                # predict which samples are anomalous
                scores = det.decision_function(class_data) #scores = det.score_samples(class_data)
                # lower score means more abnormal. we'll calculate the IQR of the scores and take only the points that
                # are at least 1 IQR below the 1st quartile
                threshold = np.quantile(scores, 0.25) - 1*iqr(scores)
                # find how many points have scores below the threshold
                abnormal_pts_idx = np.squeeze(np.argwhere((scores < threshold) & (scores < 0)))
                self._print(f"outlier indices class {c}:", self.verbosity)
                self._print(abnormal_pts_idx, self.verbosity)
                if abnormal_pts_idx.size > 0:
                    # discard samples
                    if abnormal_pts_idx.size <= max_discarded_samples:
                        # we can discard all samples predicted as anomalies
                        if abnormal_pts_idx.size == 1:
                            self.class_discarded_samples.append(np.reshape(class_data[abnormal_pts_idx,:],newshape=(1,class_data.shape[1])))
                        else:
                            self.class_discarded_samples.append(class_data[abnormal_pts_idx,:])
                        
                        self.class_num_discarded_samples.append(abnormal_pts_idx.size)
                        class_data_reduced = np.delete(class_data, abnormal_pts_idx, axis=0)
                        class_labels_reduced = np.delete(class_labels, abnormal_pts_idx, axis=0)
                        print(f"class {c} - discarded {abnormal_pts_idx.size} samples")
                        
                    else:
                        # we need to discard the most anomalous samples. sort the scores in ascending order and find the first
                        # max_discarded_samples
                        sorted_idx = np.argsort(scores)
                        discarded_idx = sorted_idx[:max_discarded_samples]
                        self._print(discarded_idx)
                        if discarded_idx.size == 1:
                            self.class_discarded_samples.append(np.reshape(class_data[discarded_idx,:],newshape=(-1,class_data.shape[1])))
                        else:
                            self.class_discarded_samples.append(class_data[discarded_idx,:])
                        class_data_reduced = np.delete(class_data, discarded_idx, axis=0)
                        class_labels_reduced = np.delete(class_labels, discarded_idx, axis=0)
                        print(f"class {c} - discarded {discarded_idx.size} samples")
        
                else:
                    class_data_reduced = class_data
                    class_labels_reduced = class_labels
                    self.class_discarded_samples.append(None)
                    self.class_num_discarded_samples.append(0)
                if i == 0:
                    X_reduced = class_data_reduced
                    y_reduced = class_labels_reduced
                else:
                    X_reduced = np.concatenate((X_reduced, class_data_reduced))
                    y_reduced = np.concatenate((y_reduced, class_labels_reduced))

        return X_reduced, y_reduced

    @staticmethod
    def _print(message, verbose=False):
        """ Simple wrapper around print to make it conditional """
        if verbose:
            print(message)

