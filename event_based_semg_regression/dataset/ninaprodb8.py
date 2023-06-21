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
from pathlib import Path
import pickle as pkl
import time
import enum

import scipy.io as sio
import numpy as np


NUM_SUBJECTS = 12
NUM_EXERCISES = 1
NUM_ACQUISITIONS = 3
NUM_CHANNELS_X = 16
NUM_CHANNELS_Y = 18
NUM_DOF = NUM_CHANNELS_Y
NUM_DOA = 5
FS_HZ = 2000.0
TS_S = 1.0 / FS_HZ

FILENAME_TEMPLATE = "S%d_E%d_A%d.%s"
FORMAT_DOWNLOADED = 'mat'
FORMAT_PROCESSED = 'pkl'


LOCAL_FOLDER_DOWNLOADED = \
    '/scratch/zanghieri/ninapro_db8/data/downloaded/'
LOCAL_FOLDER_BANDPASSED = \
    '/scratch/zanghieri/ninapro_db8/data/bandpassed/'
LOCAL_FOLDER_RECTIFIED = \
    '/scratch/zanghieri/ninapro_db8/data/rectified/'
LOCAL_FOLDER_SPIKIFIED = \
    '/scratch/zanghieri/ninapro_db8/data/spikified/'


# LOCAL_FOLDER_DOWNLOADED = \
#     '/home/marcello/datasets/ninapro_db8/data/downloaded/'
# LOCAL_FOLDER_BANDPASSED = \
#     '/home/marcello/datasets/ninapro_db8/data/bandpassed/'
# LOCAL_FOLDER_RECTIFIED = \
#     '/home/marcello/datasets/ninapro_db8/data/rectified/'
# LOCAL_FOLDER_SPIKIFIED = \
#     '/home/marcello/datasets/ninapro_db8/data/spikified/'


_MATRIX_DOF2DOA_TRANSPOSED = np.array(
    # https://www.frontiersin.org/articles/10.3389/fnins.2019.00891/full
    # Open supplemental data > Data Sheet 1.PDF >
    # > SUPPLEMENTARY METHODS > Eqn. S2
    # https://www.frontiersin.org/articles/file/downloadfile/461612_supplementary-materials_datasheets_1_pdf/octet-stream/Data%20Sheet%201.PDF/1/461612
    [
        [+0.6390,  +0.0000,  +0.0000,  +0.0000,  +0.0000],
        [+0.3830,  +0.0000,  +0.0000,  +0.0000,  +0.0000],
        [+0.0000,  +1.0000,  +0.0000,  +0.0000,  +0.0000],
        [-0.6390,  +0.0000,  +0.0000,  +0.0000,  +0.0000],
        [+0.0000,  +0.0000,  +0.4000,  +0.0000,  +0.0000],
        [+0.0000,  +0.0000,  +0.6000,  +0.0000,  +0.0000],
        [+0.0000,  +0.0000,  +0.0000,  +0.4000,  +0.0000],
        [+0.0000,  +0.0000,  +0.0000,  +0.6000,  +0.0000],
        [+0.0000,  +0.0000,  +0.0000,  +0.0000,  +0.0000],
        [+0.0000,  +0.0000,  +0.0000,  +0.0000,  +0.1667],
        [+0.0000,  +0.0000,  +0.0000,  +0.0000,  +0.3333],
        [+0.0000,  +0.0000,  +0.0000,  +0.0000,  +0.0000],
        [+0.0000,  +0.0000,  +0.0000,  +0.0000,  +0.1667],
        [+0.0000,  +0.0000,  +0.0000,  +0.0000,  +0.3333],
        [+0.0000,  +0.0000,  +0.0000,  +0.0000,  +0.0000],
        [+0.0000,  +0.0000,  +0.0000,  +0.0000,  +0.0000],
        [-0.1900,  +0.0000,  +0.0000,  +0.0000,  +0.0000],
        [+0.0000,  +0.0000,  +0.0000,  +0.0000,  +0.0000],
    ],
    dtype=np.float32,
)
MATRIX_DOF2DOA = _MATRIX_DOF2DOA_TRANSPOSED.T


class ProcessingStage(enum.Enum):
    DOWNLOAD = 'download'
    BANDPASS = 'bandpass'
    RECTIFY = 'rectify'
    SPIKIFY = 'spikify'


def sea2filename(s: int, e: int, a: int, fmt: str) -> str:
    filename = \
        FILENAME_TEMPLATE % (1 + s, 1 + e, 1 + a, fmt)
    return filename


def dof2doa(y_dof: np.ndarray) -> np.ndarray:
    y_doa = MATRIX_DOF2DOA @ y_dof
    return y_doa


def load_downloaded_session(
    idx_subject: int,
    idx_exercise: int,
    idx_acquisition: int,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:

    if verbose:
        print(
            f"Loading recording:\n"
            f"subject {1 + idx_subject}\n"
            f"exercise {1 + idx_exercise}\n"
            f"acquisition {1 + idx_acquisition}\n"
        )

    filename = sea2filename(
        s=idx_subject, e=idx_exercise, a=idx_acquisition,
        fmt=FORMAT_DOWNLOADED,
    )

    file_path = LOCAL_FOLDER_DOWNLOADED + filename

    if verbose:
        t0_s = time.time()

    data = sio.loadmat(file_path)
    x_raw = data['emg']
    y_dof = data['glove']
    del data

    x_raw = x_raw.astype(np.float32)
    y_dof = y_dof.astype(np.float32)

    x_raw = x_raw.T
    y_dof = y_dof.T

    y_doa = dof2doa(y_dof)
    del y_dof

    if verbose:
        t1_s = time.time()
        time_total_s = t1_s - t0_s
        print(f"Time taken: {time_total_s:.1f} seconds.\n")

    return x_raw, y_doa


def save_processed_session(
    idx_subject: int,
    idx_exercise: int,
    idx_acquisition: int,
    x: np.ndarray,
    y_doa: np.ndarray,
    done_stage: ProcessingStage,
) -> None:

    assert isinstance(done_stage, ProcessingStage)
    if done_stage == ProcessingStage.BANDPASS:
        dst_folder = LOCAL_FOLDER_BANDPASSED
    elif done_stage == ProcessingStage.RECTIFY:
        dst_folder = LOCAL_FOLDER_RECTIFIED
    elif done_stage == ProcessingStage.SPIKIFY:
        dst_folder = LOCAL_FOLDER_SPIKIFIED

    fmt = FORMAT_PROCESSED
    dst_filename = sea2filename(
        idx_subject, idx_exercise, idx_acquisition, fmt,
    )

    data = {
        'x': x,
        'y_doa': y_doa,
    }

    dst_full_filepath = dst_folder + dst_filename

    Path(dst_folder).mkdir(parents=True, exist_ok=True)
    with open(dst_full_filepath, 'wb') as dst_file:
        pkl.dump(data, dst_file)

    return


def load_processed_session(
    idx_subject: int,
    idx_exercise: int,
    idx_acquisition: int,
    done_stage: ProcessingStage,
) -> tuple[np.ndarray, np.ndarray]:

    assert isinstance(done_stage, ProcessingStage)
    if done_stage == ProcessingStage.BANDPASS:
        src_folder = LOCAL_FOLDER_BANDPASSED
    elif done_stage == ProcessingStage.RECTIFY:
        src_folder = LOCAL_FOLDER_RECTIFIED
    elif done_stage == ProcessingStage.SPIKIFY:
        src_folder = LOCAL_FOLDER_SPIKIFIED

    fmt = FORMAT_PROCESSED
    src_filename = sea2filename(
        idx_subject, idx_exercise, idx_acquisition, fmt,
    )

    src_full_filepath = src_folder + src_filename

    with open(src_full_filepath, 'rb') as src_file:
        data = pkl.load(src_file)

    x = data['x']
    y_doa = data['y_doa']

    return x, y_doa


def main() -> None:
    pass


if __name__ == '__main__':
    main()
