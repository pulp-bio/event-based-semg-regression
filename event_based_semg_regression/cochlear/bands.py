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
import scipy.signal as ssg
import matplotlib.pyplot as plt


ORDER = 4
BANDS_HZ_LIST = [
    (0.0, 32.1),
    (32.1, 119.2),
    (119.2, 356.1),
    (356.1, 1000.0),
]
NUM_BANDS = len(BANDS_HZ_LIST)


def create_butterworth_bank(
    f_hz: float,
    bands_hz_list: list[float],
    order: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    num_bands = len(bands_hz_list)

    z_bank = [None for idx_band in range(num_bands)]
    p_bank = [None for idx_band in range(num_bands)]
    k_bank = np.zeros((num_bands,), dtype=np.float64)
    sos_bank = [None for idx_band in range(num_bands)]

    for idx_band in range(num_bands):

        fnyquist_hz = f_hz / 2.0
        lowcut_hz, highcut_hz = bands_hz_list[idx_band]
        lowcut_norm = lowcut_hz / fnyquist_hz
        highcut_norm = highcut_hz / fnyquist_hz

        if lowcut_norm == 0.0 and highcut_norm == 1.0:
            raise NotImplementedError("Band %d is all-pass: no filtering!")
        elif lowcut_norm == 0.0 and highcut_norm < 1.0:
            Wn = highcut_norm
            btype = 'lowpass'
        elif lowcut_norm > 0.0 and highcut_norm < 1.0:
            Wn = [lowcut_norm, highcut_norm]
            btype = 'bandpass'
        elif lowcut_norm > 0.0 and highcut_norm == 1.0:
            Wn = lowcut_norm
            btype = 'highpass'

        z_bank[idx_band], p_bank[idx_band], k_bank[idx_band] = \
            ssg.butter(N=order, Wn=Wn, btype=btype, output='zpk')

        sos_bank[idx_band] = \
            ssg.zpk2sos(z_bank[idx_band], p_bank[idx_band], k_bank[idx_band])

    return sos_bank, z_bank, p_bank, k_bank


def visualize_bandfilter_bank(
    z_bank: np.ndarray,
    p_bank: np.ndarray,
    k_bank: np.ndarray,
    norm_freq: bool = True,
    fs_hz: float = None,
    num_freqs: int | None = None,
    xscale: str = 'linear',
    colors: list[str | tuple[float, float, float]] | None = None,
) -> None:

    num_bands = len(k_bank)
    w_bank = np.zeros((num_bands, num_freqs), dtype=np.float64)
    h_bank = np.zeros((num_bands, num_freqs), dtype=np.complex128)

    if norm_freq:
        fs = 2.0
        freq_axis_label = "Normalized frequency (dimensionless)"
    else:
        fs = fs_hz
        freq_axis_label = "Frequency (Hz)"

    xlim_lo = 0.0 if xscale == 'linear' else None
    xlim_hi = fs / 2.0

    for idx_band in range(num_bands):
        w_bank[idx_band], h_bank[idx_band] = ssg.freqz_zpk(
            z=z_bank[idx_band],
            p=p_bank[idx_band],
            k=k_bank[idx_band],
            worN=num_freqs,
            whole=False,
            fs=fs,
        )
    abs_h_bank = np.abs(h_bank)
    abs_db_h_bank = 20.0 * np.log10(abs_h_bank)
    angle_h_bank = np.angle(h_bank)

    fig, (ax1, ax2, ax3) = plt.subplots(
        nrows=3, ncols=1, sharex=True, figsize=(8.0, 8.0)
    )
    fig.suptitle("Frequency response of the band filters bank")
    ax1.set_title("Amplitude (linear)")
    ax2.set_title("Amplitude (logarithmic)")
    ax3.set_title("Phase")

    ax1.set_ylabel("Gain (dimensionless)")
    ax2.set_ylabel("Gain (dB)")
    ax3.set_ylabel("Phase (rad)")
    ax3.set_xlabel(freq_axis_label)

    for idx_band in range(num_bands):
        color = colors[idx_band] if colors is not None else None
        label = "band index %d" % (idx_band)
        ax1.plot(
            w_bank[idx_band], abs_h_bank[idx_band], color=color, label=label
        )
        ax2.plot(
            w_bank[idx_band], abs_db_h_bank[idx_band], color=color, label=label
        )
        ax3.plot(
            w_bank[idx_band], angle_h_bank[idx_band], color=color, label=label
        )

    ax1.legend(loc='best')
    ax2.legend(loc='best')
    ax3.legend(loc='best')

    ax1.grid(True)
    ax2.grid(True)
    ax3.grid(True)

    ax1.set_xscale(xscale)
    ax1.set_xlim([xlim_lo, xlim_hi])
    ax1.set_ylim([0.0, 1.0])
    ax2.set_ylim([-5.0, 0.0])
    ax3.set_ylim([-np.pi, +np.pi])

    fig.tight_layout()
    plt.show()

    return


def filter_multi_bands(sos_bank: np.ndarray, x: np.ndarray) -> np.ndarray:

    num_old_chnls, num_samples = x.shape
    num_bands = len(sos_bank)

    x_bp = np.zeros((num_old_chnls, num_bands, num_samples), dtype=np.float32)
    for idx_band in range(num_bands):
        x_bp[:, idx_band] = ssg.sosfilt(sos_bank[idx_band], x, axis=-1)

    return x_bp


def full_rectify(x: np.ndarray) -> np.ndarray:
    return np.abs(x)


def filter_multi_bands_multi_channel(
    x: np.ndarray,
    f_hz: float,
    bands_hz_list: list[float],
    order: int,
    bandplot: bool = False,
    bandplot_norm_freq: bool = False,
    bandplot_num_freqs: int | None = None,
    bandplot_xscale: str = 'log',
    bandplot_colors: list[str | tuple[float, float, float]] | None = None,
) -> np.ndarray:

    sos_bank, z_bank, p_bank, k_bank = create_butterworth_bank(
        f_hz=f_hz, bands_hz_list=bands_hz_list, order=order,
    )
    if bandplot:
        visualize_bandfilter_bank(
            z_bank,
            p_bank,
            k_bank,
            norm_freq=bandplot_norm_freq,
            fs_hz=f_hz,
            num_freqs=bandplot_num_freqs,
            xscale=bandplot_xscale,
            colors=bandplot_colors,
        )

    x = filter_multi_bands(sos_bank=sos_bank, x=x)
    x = full_rectify(x)

    return x


def main():
    pass


if __name__ == '__main__':
    main()
