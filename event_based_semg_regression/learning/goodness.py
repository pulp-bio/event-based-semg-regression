from __future__ import annotations
import pprint

import numpy as np
from sklearn import metrics


THR_ANGLES_DEG_LIST = [5.0, 10.0, 15.0, 20.0]


def make_results_dict_singlesubject(
    rgr,
    ytrue: np.ndarray,
    ypred_raw: np.ndarray,
    thr_angles_deg: list[float] = THR_ANGLES_DEG_LIST,
):

    results = {}

    results['rgr'] = rgr

    results['ytrue'] = ytrue
    results['ypred_raw'] = ypred_raw

    # validation coefficient of determination R2
    results['r2_raw'] = metrics.r2_score(
        ytrue.T, ypred_raw.T, multioutput='raw_values')

    # validation mean absolute error
    results['mae_raw'] = metrics.mean_absolute_error(
        ytrue.T, ypred_raw.T, multioutput='raw_values')

    # validation mean squared error
    results['mse_raw'] = metrics.mean_squared_error(
        ytrue.T, ypred_raw.T, multioutput='raw_values')

    # accuracy
    # fraction of time y_pred lies within a fixed angle threshold (e.g. 10° or
    # 15°) from y_true, as per M. Zanghieri et al., "sEMG-based Regression of
    # Hand Kinematics with Temporal Convolutional Networks on a Low-Power Edge
    # Microcontroller", 2021, https://ieeexplore.ieee.org/document/9524188

    results['acc_raw'] = {'thr_angle_deg': {}}

    for angle in thr_angles_deg:
        results['acc_raw']['thr_angle_deg'][angle] = \
            np.mean(np.abs(ypred_raw - ytrue) < angle, axis=1)

    return results
