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

import numpy as np
import brian2 as b2

from semg_spike_regression.dataset import ninaprodb8 as db8
from semg_spike_regression.cochlear import bands as bds

NUM_NEURONS = db8.NUM_CHANNELS_X * bds.NUM_BANDS

GAIN_DATA2XDRIVE = 3.0e5
X_INIT = 0.0
X_RESET = 0.5      # Kubanek: 0.5 (i.e., -65.0mV in his units)
TAU_S = 0.010      # Kubanek: 10ms
TREFR_S = 0.002    # Kubanek:  2ms; Elisa: do not go above 2ms.


def data2xdrive(data: np.ndarray, gain_data2xdrive: float = 1.0):
    x_drive = data * gain_data2xdrive
    return x_drive


def lif_presynaptic(
    x_drive: np.ndarray,
    fs_hz: float,
    dt_sim_s: float,
    monitors_dt_s: float,
    x_init: float = X_INIT,
    x_reset: float = X_RESET,
    tau_s: float = TAU_S,
    trefr_s: float = TREFR_S,
    report: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    b2.start_scope()

    # convert all time and frequency arguments to Brian2 physical quantities
    fs = fs_hz * b2.hertz
    dt_sim = dt_sim_s * b2.second
    monitors_dt = monitors_dt_s * b2.second
    tau = tau_s * b2.second
    trefr = trefr_s * b2.second

    del fs_hz, dt_sim_s, monitors_dt_s, tau_s, trefr_s

    num_samples = x_drive.shape[2]
    time_total = num_samples / fs
    dt_sample = 1.0 / fs

    x_drive = x_drive.reshape((NUM_NEURONS, num_samples))
    x_drive = b2.TimedArray(values=x_drive.T, dt=dt_sample)  # transposed!

    eqn = \
        '''
        dx/dt = (-x + x_drive(t, i)) / tau : 1 (unless refractory)
        '''

    neurongroup = b2.NeuronGroup(
        N=NUM_NEURONS,
        model=eqn,
        method='exact',
        threshold='x > 1',
        reset='x = x_reset',
        refractory=trefr,
        dtype=np.float32,
        dt=dt_sim,
    )

    neurongroup.x = x_init

    statemonitor = b2.StateMonitor(
        source=neurongroup,
        variables='x',
        record=True,  # "True" monitors all indices
        dt=monitors_dt,
    )

    # spike monitor
    spikemonitor = b2.SpikeMonitor(
        source=neurongroup,
        variables=None,  # none in addition to timestamp and spiker's index
        record=True,  # "True" monitors all indices
    )

    # (the net is created just for the scheduling summary)
    network = b2.Network(neurongroup, statemonitor, spikemonitor)
    scheduling_summary = network.scheduling_summary()
    del network
    print(f"\n\n{scheduling_summary}\n\n")

    report_period = 5.0 * b2.second
    b2.run(duration=time_total, report=report, report_period=report_period)

    x = statemonitor.x
    spike_times_s = np.float32(spikemonitor.t / b2.second)
    spike_neuron_ids = np.uint16(spikemonitor.i)
    # check whether sorted, and sort if needed
    # (I do not know if they are created sorted by design from the run; it
    # looks so, but I enforce it to be sure.)
    nonproper_isis_s = np.diff(spike_times_s)
    sorted = np.all(nonproper_isis_s >= 0.0)
    if not sorted:
        sort_idxs = np.argsort(spike_times_s)
        pike_times_s = spike_times_s[sort_idxs]
        spike_neuron_ids = spike_neuron_ids[sort_idxs]

    return x, spike_times_s, spike_neuron_ids


def lif_postsynaptic(
    inspike_times_s: np.ndarray,
    inspike_neuron_ids: np.ndarray,
    time_total_s: float,
    dt_sim_s: float,
    monitor_dt_s: float,
    tau_s: float,
    report: str | None = None,
) -> np.ndarray:

    # sanity check: same number of spikes and spikers
    num_spikes = len(inspike_times_s)
    assert len(inspike_neuron_ids) == num_spikes, \
        "Spikers' indices must be as many as the spikes' timestamps!"

    # check whether sorted, and sort if needed
    nonproper_isis_s = np.diff(inspike_times_s)
    sorted = np.all(nonproper_isis_s >= 0.0)
    if not sorted:
        sort_idxs = np.argsort(inspike_times_s)
        inspike_times_s = inspike_times_s[sort_idxs]
        inspike_neuron_ids = inspike_neuron_ids[sort_idxs]

    b2.start_scope()

    # convert all time arguments to Brian2 physical quantities
    inspike_times = inspike_times_s * b2.second
    time_total = time_total_s * b2.second
    dt_sim = dt_sim_s * b2.second
    monitor_dt = monitor_dt_s * b2.second
    tau = tau_s * b2.second
    del inspike_times_s, time_total_s, dt_sim_s, monitor_dt_s, tau_s

    spikegengroup = b2.SpikeGeneratorGroup(
        N=NUM_NEURONS,
        indices=inspike_neuron_ids,
        times=inspike_times,
        dt=dt_sim,
        when='thresholds',
        sorted=True,  # because they have just been sorted prior to this
    )

    eqn = \
        '''
        dx/dt = - x / tau : 1
        '''

    neurongroup = b2.NeuronGroup(
        N=NUM_NEURONS,
        model=eqn,
        method='exact',
        dtype=np.float32,
        dt=dt_sim,
    )

    synapses = b2.Synapses(
        source=spikegengroup,
        target=neurongroup,
        on_pre='x += 1.0',
        dtype=np.float32,
        dt=dt_sim,
    )
    synapses.connect(j='i')

    neurongroup.x = 0.0

    statemonitor = b2.StateMonitor(
        source=neurongroup,
        variables='x',
        record=True,  # "True" monitors all indices
        dt=monitor_dt,
        when='end',
    )

    # (the net is created just for the scheduling summary)
    network = b2.Network(neurongroup, synapses, statemonitor)
    scheduling_summary = network.scheduling_summary()
    del network
    print(f"\n\n{scheduling_summary}\n\n")

    report_period = 5.0 * b2.second
    b2.run(duration=time_total, report=report, report_period=report_period)

    x = statemonitor.x[:, 1:]  # discard the first because it is time 0.0

    return x


def main():
    pass


if __name__ == '__main__':
    main()
