from collections import Sequence

from scipy import stats
import json
import numpy as np
from utils.NpEncoder import NpEncoder


class CDD():
    def __init__(self, path=None, prev_sample: list = None):
        """
        Loaded previous predict histograms if available or uses the given sample to compare to.
        :param path: path to cache file (json type)
        :param prev_sample: list of previous samples
        """
        if path is not None:
            with open(path) as f:
                self.cache = json.load(f)
        elif prev_sample is not None:
            self.add_sample(prev_sample)
        else:
            self.cache = None

    def add_sample(self, sample: Sequence):
        """
        Adds another sample to compare in the evaluation phase
        :param sample: list of previouse predictions
        """
        if self.cache is None:
            self.cache = {"prev_samples": [sample]}
        else:
            self.cache["prev_samples"].append(sample)

    def evaluate(self, y, save_to_cache=True):
        """
        Evaluates the current prediction of the `model` on `X` with the previous histograms.
        :param model:  a model with predict_proba method
        :param X: data to classify
        :param save_to_cache: A flag that indicates if to save the prediction histogram to cache
        :return: ks_statistics with previous histograms, pvalue of ks results with previous histograms
        """
        if self.cache is None:
            raise Exception("No previous samples to evaluate")
        ks_stat_res = []
        ks_pval_res = []
        for prev_sample in self.cache["prev_samples"]:
            res = stats.ks_2samp(y, prev_sample)
            ks_stat_res.append(res[0])
            ks_pval_res.append(res[1])

        if save_to_cache:
            self.cache["prev_samples"].append(y)
        return ks_stat_res, ks_pval_res

    def save(self, path):
        """
        Saves the metadata of the histograms to file
        :param path: path to save the data as json format
        """
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=4, cls=NpEncoder)


