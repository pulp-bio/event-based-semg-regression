"""
    Author(s):
    Marcello Zanghieri <marcello.zanghieri2@unibo.it>
    
    Copyright (C) 2023 University of Bologna and ETH Zurich
    
    Licensed under the GNU Lesser General Public License (LGPL), Version 2.1
    (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
    
        https://www.gnu.org/licenses/lgpl-2.1.txt
    
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
"""

from __future__ import annotations
import copy

import numpy as np
from sklearn import preprocessing
from sklearn import linear_model
from sklearn import metrics

from semg_spike_regression.learning import goodness as good


def hasmeth(obj: object, name: str):
    if hasattr(obj, name):
        attr = getattr(obj, name)
        if callable(attr):
            return True
    return False


class MultiRegressor():

    def __init__(self, template_rgr) -> None:
        assert hasmeth(template_rgr, 'fit'), \
            "Received template regressor has no fit() method!"
        assert hasmeth(template_rgr, 'predict'), \
            "Received template regressor has no predict() method!"
        self.template_rgr = template_rgr

    def init_models(self) -> None:
        self.uni_regressors = [
            copy.deepcopy(self.template_rgr) for _ in range(self.num_y_vars)]

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        self.num_y_vars = y.shape[0]
        self.init_models()

        x = x.T
        y = y.T

        self.scaler_x = preprocessing.StandardScaler()
        self.scaler_y = preprocessing.StandardScaler()

        self.scaler_x.fit(x)
        self.scaler_y.fit(y)

        x = self.scaler_x.transform(x)
        y = self.scaler_y.transform(y)

        for idx_y_var in range(self.num_y_vars):
            self.uni_regressors[idx_y_var].fit(x, y[:, idx_y_var])

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = x.T
        x = self.scaler_x.transform(x)

        # just a patch
        x[np.abs(x) == np.inf] = 0.0

        y_list = [
            self.uni_regressors[idx_y_var].predict(x)
            for idx_y_var in range(self.num_y_vars)
        ]
        y = np.stack(y_list, axis=1)
        del y_list
        y = self.scaler_y.inverse_transform(y)
        y = y.T
        return y


def univar_mae(ytrue: np.ndarray, ypred: np.ndarray) -> float:
    # check that y's are actually univariate
    assert len(ytrue.shape) == 1 or ytrue.shape[0] == 1
    assert len(ypred.shape) == 1 or ypred.shape[0] == 1
    # return the Mean Absolute Error
    return metrics.mean_absolute_error(ytrue, ypred)


def train_regressor(
    train_set: tuple[np.ndarray, np.ndarray],
    valid_set: tuple[np.ndarray, np.ndarray],
    downsampling: int = 1,
    verbose: bool = True,
):  # returns a Scikit-learn regressor

    # unpack the data-subset arrays
    xtrain, ytrain = train_set
    del train_set
    xvalid, yvalid = valid_set
    del valid_set

    # downsampling
    xtrain = xtrain[:, ::downsampling]
    xvalid = xvalid[:, ::downsampling]
    ytrain = ytrain[:, ::downsampling]
    yvalid = yvalid[:, ::downsampling]

    # create regressor
    template_rgr = linear_model.LassoCV(
        eps=0.001,
        n_alphas=100,
        alphas=None,
        fit_intercept=False,
        precompute=False,
        cv=2,
        n_jobs=-1,
        positive=False,
    )
    rgr = MultiRegressor(template_rgr)

    # training
    rgr.fit(xtrain, ytrain)

    # inference on training set and validation set
    ytrain_pred_raw = rgr.predict(xtrain)
    yvalid_pred_raw = rgr.predict(xvalid)

    # compute results dictionaries
    # training results
    results_train = good.make_results_dict_singlesubject(
        rgr=rgr,
        ytrue=ytrain,
        ypred_raw=ytrain_pred_raw,
    )
    # validation results
    results_valid = good.make_results_dict_singlesubject(
        rgr=rgr,
        ytrue=yvalid,
        ypred_raw=yvalid_pred_raw,
    )
    # merger
    results = {
        'train': results_train,
        'valid': results_valid,
    }

    if verbose:

        # quick temporary MAEs
        mae_train = metrics.mean_absolute_error(
            ytrain.T, ytrain_pred_raw.T, multioutput='raw_values')
        mae_valid = metrics.mean_absolute_error(
            yvalid.T, yvalid_pred_raw.T, multioutput='raw_values')
        mae_train_avg = mae_train.mean()
        mae_valid_avg = mae_valid.mean()

        print('\n')
        print("MAE on TRAINING set:")
        print("By channel:")
        print(mae_train.round(2))
        print("Average:")
        print(mae_train_avg.round(2))

        print("MAE on VALIDATION set:")
        print("By channel:")
        print(mae_valid.round(2))
        print("Average:")
        print(mae_valid_avg.round(2))
        print('\n')

    return results


def main():
    pass


if __name__ == '__main__':
    main()
