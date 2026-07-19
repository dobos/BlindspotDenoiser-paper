# Original simulation file without wavelength mask
SIM_FILE = '/datascope/subaru/user/swei20/data/bosz50000/z1/mag205_225_lowT_1M/train_200k_0/dataset.h5'

# Data used to train the networks
TRAINING_FILE = '/datascope/subaru/user/swei20/blindspot_rv/inputs/bosz50000/b1_l9_e48_k25_s1_bn1_d1_T0_S0_L0_snr3_b50000_rv500_ep5000_N50000_m0.npz'

# Wavelength mask used to train the networks
DENOISED_MASK = '/datascope/subaru/user/swei20/model/bosz50000_mask.npy'

# Blindspot denoised file
DENOISED_FILE = '/datascope/subaru/user/swei20/blindspot_rv/inputs/bosz50000_v3ep190/b1_l9_e48_k25_s1_bn1_d1_T0_S0_L0_snr3_b50000_rv500_ep5000_N50000_m0_v3ep190_N50000.npz'

# U-Net denoised file (trained on clean reference)
CLEANED_FILE = '/datascope/subaru/user/swei20/blindspot_rv/inputs/bosz50000_v3ep190/b1_l9_e48_k25_s1_bn1_d1_T0_S0_L0_snr3_b50000_rv500_ep5000_N50000_m0_v3ep190_E2_SUPERVISED_N50000.npz'

# DAE denoised file
# CLEANED_FILE = '/datascope/subaru/user/swei20/blindspot_rv/inputs/bosz50000_v3ep190/b1_l9_e48_k25_s1_bn1_d1_T0_S0_L0_snr3_b50000_rv500_ep5000_N50000_m0_v3ep190_DAE_N50000.npz'

N = 10000


##########################


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
from collections import defaultdict
from tqdm.notebook import tqdm


def load_simulation(param=None, N=N):
    # Load the pandas DataFrame from the HDF5 file stored in /dataset/params/table
    with h5py.File(SIM_FILE, 'r') as f:
        params = pd.read_hdf(SIM_FILE, key='dataset/params')

    print('Simulation params shape:', params.shape)

    if param is not None:
        # Select only those spectra that were interpolated in the log g direction
        data_mask = params['interp_param'] == param
    else:
        # Include all spectra
        data_mask = np.full_like(params['interp_param'], True)

    data_idx = np.where(data_mask)[0]
    print('Number of selected spectra:', data_mask.sum())

    # Load the selected spectra from the HDF5 file
    with h5py.File(SIM_FILE, 'r') as f:
        wave = f['spectrumdataset/wave'][:]
        flux = f['dataset/arrays/flux/value'][data_idx[:N], :]
        flux_err = f['dataset/arrays/error/value'][data_idx[:N], :]

    wave.shape, flux.shape, flux_err.shape

    return wave, flux, flux_err, params.iloc[data_idx[:N]].reset_index(drop=True)