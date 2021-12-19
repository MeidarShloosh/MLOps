from scipy import stats
import json
import numpy as np
from utils.NpEncoder import NpEncoder


class CDD():
    def __init__(self, path=None, prev_sample: list = None, apply_hist=True, hist_bins=50):
        """
        Loaded previous predict histograms if available or uses the given sample to compare to.
        :param path: path to cache file (json type)
        :param prev_sample: list of previous samples
        :param apply_hist: flag to indicate if need to apply histogram on the observations
        :param hist_bins: number of histogram bins
        """
        if path is not None:
            with open(path) as f:
                self.cache = json.load(f)
        elif prev_sample is not None:
            prev_sample = list(prev_sample) if not apply_hist else np.histogram(prev_sample, hist_bins)[0].tolist()
            self.cache = {
                "hist_bins":  hist_bins,
                "apply_hist": apply_hist,
                "prev_samples": [prev_sample]
            }
        else:
            raise Exception("Must provide either previous sample histogram to compare or cache path")

    def evaluate(self, y, save_to_cache=False):
        """
        Evaluates the current prediction of the `model` on `X` with the previous histograms.
        :param model:  a model with predict_proba method
        :param X: data to classify
        :param save_to_cache: A flag that indicates if to save the prediction histogram to cache
        :return: ks_statistics with previous histograms, pvalue of ks results with previous histograms
        """
        dist = y
        if self.cache["apply_hist"]:
            dist = np.histogram(y, self.cache["hist_bins"])[0].tolist()
        ks_stat_res = []
        ks_pval_res = []
        for prev_sample in self.cache["prev_samples"]:
            res = stats.ks_2samp(dist, prev_sample)
            ks_stat_res.append(res[0])
            ks_pval_res.append(res[1])

        if save_to_cache:
            self.cache["prev_samples"].append(dist)
        return ks_stat_res, ks_pval_res

    def save(self, path):
        """
        Saves the metadata of the histograms to file
        :param path: path to save the data as json format
        """
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=4, cls=NpEncoder)


