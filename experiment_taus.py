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

from itertools import product
import pickle
from pathlib import Path

import numpy as np

from semg_spike_regression.dataset import ninaprodb8 as db8
from semg_spike_regression.cochlear import lif
from semg_spike_regression.learning import regressor as reg


RESULTS_FOLDER = 'results/'
RESULTS_FILE = 'results_gain_3.0e5_refractory_2ms.pkl'

SIMULATION_TIMESTEP_S = db8.TS_S  # must be multiple of the dataset's t_sample
INFERENCE_TIMESTEP_S = 0.016  # must be multiple of the simulation timestep
TRAINING_TIMESTEP_S = 0.100  # must be multiple of the simulation timestep

INFER_DOWN_FACTOR = int(INFERENCE_TIMESTEP_S / db8.TS_S)  # downsampling
TRAIN_DOWN_FACTOR = int(TRAINING_TIMESTEP_S / db8.TS_S)  # downsampling

# TAU_S_LIST = [
#     5.000, 2.000, 1.000,
#     0.500, 0.200, 0.100,
#     0.050, 0.020, 0.010,
#     0.005, 0.002, 0.001,
#     0.0005,
# ]
TAU_S_LIST = [
    0.500,
]
NUM_TAUS = len(TAU_S_LIST)


results_taus = {
    'tau_s': {
        tau_s: {
            'subject': {
                idx_subj: {} for idx_subj in range(db8.NUM_SUBJECTS)
            }
        } for tau_s in TAU_S_LIST
    }
}


for idx_tau, idx_subj in product(range(NUM_TAUS), range(db8.NUM_SUBJECTS)):

    print(
        f"\n\n\n "
        f"TAU VALUE {1 + idx_tau}/{NUM_TAUS}\n"
        f"SUBJECT {1 + idx_subj}/{db8.NUM_SUBJECTS}\n"
        f"\n\n\n"
    )

    # ------------------------------------------------------------------- #
    # Load the spikified data
    # ------------------------------------------------------------------- #

    # load first acquisition
    x_lif_presynaptic_acq0, y_doa_acq0 = \
        db8.load_processed_session(
            idx_subject=idx_subj, idx_exercise=0, idx_acquisition=0,
            done_stage=db8.ProcessingStage.SPIKIFY,
        )
    spike_times_s_acq0 = x_lif_presynaptic_acq0['spike_times_s']
    spike_neuron_ids_acq0 = x_lif_presynaptic_acq0['spike_neuron_ids']
    del x_lif_presynaptic_acq0

    # load second acquisition
    x_lif_presynaptic_acq1, y_doa_acq1 = \
        db8.load_processed_session(
            idx_subject=idx_subj, idx_exercise=0, idx_acquisition=1,
            done_stage=db8.ProcessingStage.SPIKIFY,
        )
    spike_times_s_acq1 = x_lif_presynaptic_acq1['spike_times_s']
    spike_neuron_ids_acq1 = x_lif_presynaptic_acq1['spike_neuron_ids']
    del x_lif_presynaptic_acq1

    # load third acquision
    x_lif_presynaptic_acq2, y_doa_acq2 = \
        db8.load_processed_session(
            idx_subject=idx_subj, idx_exercise=0, idx_acquisition=2,
            done_stage=db8.ProcessingStage.SPIKIFY,
        )
    spike_times_s_acq2 = x_lif_presynaptic_acq2['spike_times_s']
    spike_neuron_ids_acq2 = x_lif_presynaptic_acq2['spike_neuron_ids']
    del x_lif_presynaptic_acq2

    # ------------------------------------------------------------------- #
    # Feature Extraction
    # ------------------------------------------------------------------- #

    num_samples_acq0 = y_doa_acq0.shape[1]
    time_total_s_acq0 = num_samples_acq0 * db8.TS_S

    num_samples_acq1 = y_doa_acq1.shape[1]
    time_total_s_acq1 = num_samples_acq1 * db8.TS_S

    num_samples_acq2 = y_doa_acq2.shape[1]
    time_total_s_acq2 = num_samples_acq2 * db8.TS_S

    x_lif_postsynaptic_acq0 = lif.lif_postsynaptic(
        inspike_times_s=spike_times_s_acq0,
        inspike_neuron_ids=spike_neuron_ids_acq0,
        time_total_s=time_total_s_acq0,
        dt_sim_s=SIMULATION_TIMESTEP_S,
        monitor_dt_s=TRAINING_TIMESTEP_S,  # will be used for training
        tau_s=TAU_S_LIST[idx_tau],
        report='stdout',
    )

    x_lif_postsynaptic_acq1 = lif.lif_postsynaptic(
        inspike_times_s=spike_times_s_acq1,
        inspike_neuron_ids=spike_neuron_ids_acq1,
        time_total_s=time_total_s_acq1,
        dt_sim_s=SIMULATION_TIMESTEP_S,
        monitor_dt_s=TRAINING_TIMESTEP_S,  # will be used for training
        tau_s=TAU_S_LIST[idx_tau],
        report='stdout',
    )

    x_lif_postsynaptic_acq2 = lif.lif_postsynaptic(
        inspike_times_s=spike_times_s_acq2,
        inspike_neuron_ids=spike_neuron_ids_acq2,
        time_total_s=time_total_s_acq2,
        dt_sim_s=SIMULATION_TIMESTEP_S,
        monitor_dt_s=INFERENCE_TIMESTEP_S,  # will be used for validation
        tau_s=TAU_S_LIST[idx_tau],
        report='stdout',
    )
    print("Done.")

    # synchronize y's
    y_doa_acq0 = y_doa_acq0[:, TRAIN_DOWN_FACTOR - 1:: TRAIN_DOWN_FACTOR]
    y_doa_acq1 = y_doa_acq1[:, TRAIN_DOWN_FACTOR - 1:: TRAIN_DOWN_FACTOR]
    y_doa_acq2 = y_doa_acq2[:, INFER_DOWN_FACTOR - 1:: INFER_DOWN_FACTOR]

    del spike_times_s_acq0, spike_neuron_ids_acq0
    del spike_times_s_acq1, spike_neuron_ids_acq1
    del spike_times_s_acq2, spike_neuron_ids_acq2

    # ------------------------------------------------------------------- #
    # Dataset split
    # ------------------------------------------------------------------- #

    xtrain = np.concatenate(
        (x_lif_postsynaptic_acq0, x_lif_postsynaptic_acq1), axis=1)
    ytrain = np.concatenate((y_doa_acq0, y_doa_acq1), axis=1)

    xvalid = x_lif_postsynaptic_acq2
    yvalid = y_doa_acq2

    del x_lif_postsynaptic_acq0, \
        x_lif_postsynaptic_acq1, \
        x_lif_postsynaptic_acq2
    del y_doa_acq0, y_doa_acq1, y_doa_acq2

    # ------------------------------------------------------------------- #
    # Feature normalization
    # ------------------------------------------------------------------- #

    phi_refr = lif.TREFR_S / TAU_S_LIST[idx_tau]
    max_theor = 1.0 / (1.0 - np.exp(-phi_refr))
    xtrain /= max_theor
    xvalid /= max_theor

    # ------------------------------------------------------------------- #
    # Training
    # ------------------------------------------------------------------- #

    print("Training regressor...")

    # to shut up LassoCV
    def warn(*args, **kwargs): pass
    import warnings
    warnings.warn = warn

    results_onesubj = reg.train_regressor(
        train_set=(xtrain, ytrain),
        valid_set=(xvalid, yvalid),
        downsampling=1,
    )
    print("Regressor done.")

    # ------------------------------------------------------------------- #
    # Gather and save the results
    # ------------------------------------------------------------------- #

    results_taus['tau_s'][TAU_S_LIST[idx_tau]]['subject'][idx_subj] = \
        results_onesubj

    Path(RESULTS_FOLDER).mkdir(parents=True, exist_ok=True)
    RESULTS_FILE_PATH = RESULTS_FOLDER + RESULTS_FILE
    with open(RESULTS_FILE_PATH, 'wb') as f:
        pickle.dump({'results_taus': results_taus}, f)
