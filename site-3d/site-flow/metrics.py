"""Functions to calculate training loss on single image."""

from typing import List, Optional, Tuple, Union
import warnings
import importlib

import numpy as np
import pandas as pd
import scipy.optimize

from utils.util import _EPSILON, offset_euclidean, _get_offsets, linear_sum_assignment



class F1Score:

    def __init__(self):
        pass

    def f1_at_cutoff(
        self,
        pred: np.ndarray,
        true: np.ndarray,
        cutoff: float,
        return_raw=False,
    ):
        matrix = scipy.spatial.distance.cdist(pred, true, metric="euclidean")

        # Cannot assign coordinates on empty matrix
        if matrix.size == 0:
            return 0.0 if not return_raw else (0.0, [], [])

        # Assignment of pred<-true and true<-pred
        pred_true_r, _ = linear_sum_assignment(matrix, cutoff)
        true_pred_r, true_pred_c = linear_sum_assignment(matrix.T, cutoff)

        # Calculation of tp/fn/fp based on number of assignments
        tp = len(true_pred_r)
        fn = len(true) - len(true_pred_r)
        fp = len(pred) - len(pred_true_r)

        recall = tp / (tp + fn + _EPSILON)
        precision = tp / (tp + fp + _EPSILON)
        f1_value = (2 * precision * recall) / (precision + recall + _EPSILON)

        if return_raw:
            return f1_value, recall, precision, true_pred_r, true_pred_c

        return f1_value, recall, precision

    def f1_integral(
            self,
            pred: np.ndarray,
            true: np.ndarray,
            mdist: float = 3.0,
            n_cutoffs: int = 50,
            return_raw: bool = False,
        ) -> Union[float, tuple]:

        assert pred.ndim == true.ndim == 2 and pred.shape[1] == 2

        cutoffs = np.linspace(start=0, stop=mdist, num=n_cutoffs)

        if pred.size == 0 or true.size == 0:
            warnings.warn(
                f"Pred ({pred.shape}) and true ({true.shape}) must have size != 0.",
                RuntimeWarning,
            )
            return 0.0 if not return_raw else (np.zeros(n_cutoffs), np.zeros(n_cutoffs), cutoffs)

        if not return_raw:
            f1_scores, _, _ = [self.f1_at_cutoff(pred, true, cutoff) for cutoff in cutoffs]
            return np.trapz(f1_scores, cutoffs) / mdist  # Norm. to 0-1

        f1_scores = []
        precision_scores = []
        recall_socres = []
        offsets = []
        for cutoff in cutoffs:
            f1_value, recall, precision, rows, cols = self.f1_at_cutoff(
                pred, true, cutoff, return_raw=True
            )
            f1_scores.append(f1_value)
            precision_scores.append(precision)
            recall_socres.append(recall)
            offsets.append(_get_offsets(pred, true, rows, cols))

        return (f1_scores, precision_scores, recall_socres, offsets, list(cutoffs))
    
    def compute_metrics(
        pred: np.ndarray, true: np.ndarray, mdist: float = 3.0
    ) -> pd.DataFrame:
        """Calculate metric scores across cutoffs.

        Args:
            pred: Predicted set of coordinates.
            true: Ground truth set of coordinates.
            mdist: Maximum euclidean distance in px to which F1 scores will be calculated.

        Returns:
            DataFrame with one row per cutoff containing columns for:
                * f1_score: Harmonic mean of precision and recall based on the number of coordinates
                    found at different distance cutoffs (around ground truth).
                * precision_score, recall_score.
                * abs_euclidean: Average euclidean distance at each cutoff.
                * offset: List of (r, c) coordinates denoting offset in pixels.
                * f1_integral: Area under curve f1_score vs. cutoffs.
                * mean_euclidean: Normalized average euclidean distance based on the total number of assignments.
        """
        f1_scores, precision_scores, recall_socres, offsets, cutoffs = f1_integral(pred, true, mdist=mdist, n_cutoffs=50, return_raw=True)  # type: ignore

        abs_euclideans = []
        total_euclidean = 0
        total_assignments = 0

        # Find distances through offsets at every cutoff
        for c_offset in offsets:
            abs_euclideans.append(np.mean(offset_euclidean(c_offset)))
            total_euclidean += np.sum(offset_euclidean(c_offset))
            try:
                total_assignments += len(c_offset)
            except TypeError:
                continue

        df = pd.DataFrame(
            {
                "cutoff": cutoffs,
                "f1_score": f1_scores,
                "precision_scores": precision_scores,
                "recall_socres": recall_socres,
                "abs_euclidean": abs_euclideans,
                "offset": offsets,
            }
        )
        df["f1_integral"] = np.trapz(df["f1_score"], cutoffs) / mdist  # Norm. to 0-1
        df["mean_euclidean"] = total_euclidean / (total_assignments + 1e-10)

        return df

    
class IoUScore:
    pass


class RmseSroce:
    pass


# ==============
# Entry Funciton
# ==============

def get_evaluation_metric(config):
    """Returns the evaluation metric function based on provided configuration

    Args:
        config (dict): a top level configuration object containing the 'eval_metric' key
    
    Return:
        an instance of the evaluation metric
    """

    def _metric_class(class_name):
        m = importlib.import_module('pytorch3dunet.unet3d.metrics')
        clazz = getattr(m, class_name)
        return clazz

    assert 'eval_metric' in config, 'Could not find evaluation metric configuration'
    metric_config = config['eval_metric']
    metric_class = _metric_class(metric_config['name'])
    return metric_class(**metric_config)