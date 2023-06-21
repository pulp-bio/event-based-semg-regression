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

from semg_spike_regression.dataset import ninaprodb8 as db8
from semg_spike_regression.cochlear import bands
from semg_spike_regression.cochlear import lif

# set to process, zero-based indexed)
subj_list = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
]
ex_list = [
    0,
]
acq_list = [
    0, 1, 2,
]

for idx_subj, idx_ex, idx_acq in product(subj_list, ex_list, acq_list):

    print(
        f"\n\n\n "
        f"SUBJECT {1 + idx_subj}/{db8.NUM_SUBJECTS}, "
        f"EXERCISE {1 + idx_ex}/{db8.NUM_EXERCISES}, "
        f"ACQUISITION {1 + idx_acq}/{db8.NUM_ACQUISITIONS} "
        f"\n\n\n"
    )

    # ----------------------------------------------------------------------- #
    # Load the original released raw data
    # ----------------------------------------------------------------------- #
    x_raw, y_doa = db8.load_downloaded_session(
        idx_subj, idx_ex, idx_acq, verbose=True)

    # ----------------------------------------------------------------------- #
    # Band-pass filtering
    # ----------------------------------------------------------------------- #

    x_bp = bands.filter_multi_bands_multi_channel(
        x=x_raw,
        f_hz=db8.FS_HZ,
        order=bands.ORDER,
        bands_hz_list=bands.BANDS_HZ_LIST,
        bandplot=False,
    )

    # ----------------------------------------------------------------------- #
    # Pass into LIF
    # ----------------------------------------------------------------------- #

    x_drive = lif.data2xdrive(x_bp, lif.GAIN_DATA2XDRIVE)

    x, spike_times_s, spike_neuron_ids = lif.lif_presynaptic(
        x_drive=x_drive,
        fs_hz=db8.FS_HZ,
        dt_sim_s=db8.TS_S,
        monitors_dt_s=db8.TS_S,
        report='stdout',
    )

    num_spikes = len(spike_times_s)
    print(f"Total spikes: {num_spikes}")

    # save spikified session
    x_lif_presynaptic = {
        'x': x,
        'spike_times_s': spike_times_s,
        'spike_neuron_ids': spike_neuron_ids,
    }
    db8.save_processed_session(
        idx_subj, idx_ex, idx_acq,
        x=x_lif_presynaptic, y_doa=y_doa,
        done_stage=db8.ProcessingStage.SPIKIFY,
    )
