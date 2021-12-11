from scipy import stats
import json
import numpy as np
from utils.NpEncoder import NpEncoder

class CDD():
    def __init__(self, path=None, prev_sample:list=None):
        if path:
            with open(path) as f:
                self.cache = json.load(f)
        elif prev_sample is not None:
            self.cache = {
                "hist_bins":  len(prev_sample),
                "prev_samples": [list(prev_sample)]
            }
        else:
            raise Exception("Must provide either previous sample histogram to compare or cache path")

    @staticmethod
    def get_hist(model, X, hist_bins=20):
        y_pred = model.predict_proba(X)
        hist = np.histogram(y_pred, hist_bins)
        return hist

    def evaluate(self, model, X, save_to_cache=False):
        if not hasattr(model, 'predict_proba'):
            raise Exception("Model doesn't have soft prediction method")
        y_pred = model.predict_proba(X)
        hist = np.histogram(y_pred, self.cache["hist_bins"])[0].tolist()
        ks_stat_res = []
        ks_pval_res = []
        for prev_sample in self.cache["prev_samples"]:
            res = stats.ks_2samp(hist, prev_sample)
            ks_stat_res.append(res[0])
            ks_pval_res.append(res[1])

        if save_to_cache:
            self.cache["prev_samples"].append(hist)
        return ks_stat_res, ks_pval_res

    def save(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=4, cls=NpEncoder)


