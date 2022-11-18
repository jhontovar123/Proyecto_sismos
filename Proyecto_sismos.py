
import numpy as np
import math
import matplotlib.pyplot as plt
import os
import pandas as pd
from pandas.plotting import scatter_matrix
from scipy import stats

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import collections
import seaborn as sns
import csv


PROJECT_ROOT_DIR = "."
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "Figuras")
os.makedirs(IMAGES_PATH, exist_ok=True)
#
def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure: ", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
#
# ========================== MAIN SCRIPT ============================
# --- Reading the data

fpath = '.\\Data\\'

#Puente nuevo
fname1 = 'Ace_1.5luz.csv'  # '
fname_t1 = fpath + fname1
fname2 = 'Ace_1.25luz.csv'  # 
fname_t2 = fpath + fname2
fname3 = 'Ace_1.75luz.csv'  # 
fname_t3 = fpath + fname3
#

#Puente viejo
fname4 = 'Ace_1.5luz_viejo.csv'  # '
fname_t4 = fpath + fname4
fname5 = 'Ace_1.25luz_viejo.csv'  # 
fname_t5 = fpath + fname5
fname6 = 'Ace_1.75luz_viejo.csv'  # 
fname_t6 = fpath + fname6

# --- Reading the data
df_15luz = pd.read_csv(fname_t1)
df_125luz = pd.read_csv(fname_t2)
df_175luz = pd.read_csv(fname_t3)

df_15luz_viejo = pd.read_csv(fname_t4)
df_125luz_viejo = pd.read_csv(fname_t5)
df_175luz_viejo = pd.read_csv(fname_t6)

# --- Initial look of the data
print(df_15luz.head(20))
print(df_15luz.info())  # data types and general info
#
# ==== Part 2: Preprocessing the data
#
data_mid = df_15luz.copy()
data_left = df_125luz.copy()
data_right = df_175luz.copy()

data_mid_viejo = df_15luz_viejo.copy()
data_left_viejo = df_125luz_viejo.copy()
data_right_viejo = df_175luz_viejo.copy()

import eqsig.single
bf, sub_fig = plt.subplots()
a = data_mid['Linear Acceleration z (m/s^2)'][12000:60000] #min 2 a min 10
dt = 0.01  # time step of acceleration time series
periods = np.linspace(0.2, 5, 100)  # compute the response for 100 periods between T=0.2s and 5.0s
record = eqsig.AccSignal(a * 9.8, dt)
record.generate_response_spectrum(response_times=periods)
times = record.response_times
sub_fig.plot(times, record.s_a, label="eqsig")
save_fig('Data_mid')

bf, sub_fig = plt.subplots()
a = data_left['Linear Acceleration z (m/s^2)'][12000:60000] #min 2 a min 10
dt = 0.01  # time step of acceleration time series
periods = np.linspace(0.2, 5, 100)  # compute the response for 100 periods between T=0.2s and 5.0s
record = eqsig.AccSignal(a * 9.8, dt)
record.generate_response_spectrum(response_times=periods)
times = record.response_times
sub_fig.plot(times, record.s_a, label="eqsig")
save_fig('Data_left')

bf, sub_fig = plt.subplots()
a = data_right['Linear Acceleration z (m/s^2)'][12000:60000] #min 2 a min 10
dt = 0.01  # time step of acceleration time series
periods = np.linspace(0.2, 5, 100)  # compute the response for 100 periods between T=0.2s and 5.0s
record = eqsig.AccSignal(a * 9.8, dt)
record.generate_response_spectrum(response_times=periods)
times = record.response_times
sub_fig.plot(times, record.s_a, label="eqsig")
save_fig('Data_right')

#========== Produce Fourier spectrum and smoothed Fourier spectrum (https://github.com/eng-tools/eqsig/blob/master/examples/example_filtering_and_fourier_spectrum.ipynb)
plt.figure(figsize=(15,8))
acc = data_mid['Linear Acceleration z (m/s^2)'][12000:60000] #min 2 a min 10
dt = 0.01
asig = eqsig.AccSignal(acc, dt, label='name_of_record')
plt.plot(asig.fa_frequencies, abs(asig.fa_spectrum), c='b')
asig.smooth_fa_frequencies = np.logspace(-1, 1, 50)
plt.loglog(asig.smooth_fa_frequencies, asig.smooth_fa_spectrum, c='r', ls='--')
plt.xlabel('Frequency [Hz]')
lab = plt.ylabel('Fourier Amplitude [m/s]')
save_fig('Fourier_spectrum_DataMid')

##=====Filter record using  filter
plt.figure(figsize=(15,8))
plt.plot(asig.fa_frequencies, abs(asig.fa_spectrum), c='b')
asig.butter_pass((None, 8))  # Low pass filter at 8Hz (default 4th order)
plt.plot(asig.fa_frequencies, abs(asig.fa_spectrum), c='r', ls='--')
asig.butter_pass((0.1, None), filter_order=4)  # High pass filter at 0.1Hz (default 4th order)
plt.plot(asig.fa_frequencies, abs(asig.fa_spectrum), c='g', ls=':')
asig.butter_pass((0.2, 6))  # Band pass filter at 0.2 and 6Hz (default 4th order)
plt.loglog(asig.fa_frequencies, abs(asig.fa_spectrum), c='k', ls='--', lw=0.7)
plt.xlabel('Frequency [Hz]')
lab = plt.ylabel('Fourier Amplitude [m/s]')
save_fig('Butterworth_DataMid')

#===== Detrend a record
plt.figure(figsize=(15,8))
bf, sps = plt.subplots(nrows=3, sharex='col')
acc = data_mid['Linear Acceleration z (m/s^2)'][12000:60000] #min 2 a min 10
asig = eqsig.AccSignal(acc, dt, label='name_of_record')
asig.remove_poly(poly_fit=1)  # Remove any trend
sps[0].plot(asig.time, asig.values, label='Original')
sps[1].plot(asig.time, asig.velocity)
sps[2].plot(asig.time, asig.displacement)

# Add a trend
asig.add_series(np.linspace(0, 0.1, asig.npts))
sps[0].plot(asig.time, asig.values, ls='--', label='with trend')
sps[1].plot(asig.time, asig.velocity, ls='--')
sps[2].plot(asig.time, asig.displacement, ls='--')

# remove the trend
asig.remove_poly(poly_fit=1)
sps[0].plot(asig.time, asig.values, ls=':', label='trend removed')
sps[1].plot(asig.time, asig.velocity, ls=':')
sps[2].plot(asig.time, asig.displacement, ls=':')
sps[0].set_ylabel('Accel. [$m/s^2$]')
sps[1].set_ylabel('Velo. [$m/s$]')
sps[2].set_ylabel('Disp. [$m$]')
sps[-1].set_xlabel('Time [s]')
sps[0].legend(prop={'size': 6}, ncol=3)
save_fig('Detrend_DataMid')

##
plt.figure(figsize=(15,8))
bf, sps = plt.subplots(nrows=3, sharex='col')
acc = data_left['Linear Acceleration z (m/s^2)'][12000:60000] #min 2 a min 10
asig = eqsig.AccSignal(acc, dt, label='name_of_record')
asig.remove_poly(poly_fit=1)  # Remove any trend
sps[0].plot(asig.time, asig.values, label='Original')
sps[1].plot(asig.time, asig.velocity)
sps[2].plot(asig.time, asig.displacement)

# Add a trend
asig.add_series(np.linspace(0, 0.1, asig.npts))
sps[0].plot(asig.time, asig.values, ls='--', label='with trend')
sps[1].plot(asig.time, asig.velocity, ls='--')
sps[2].plot(asig.time, asig.displacement, ls='--')

# remove the trend
asig.remove_poly(poly_fit=1)
sps[0].plot(asig.time, asig.values, ls=':', label='trend removed')
sps[1].plot(asig.time, asig.velocity, ls=':')
sps[2].plot(asig.time, asig.displacement, ls=':')
sps[0].set_ylabel('Accel. [$m/s^2$]')
sps[1].set_ylabel('Velo. [$m/s$]')
sps[2].set_ylabel('Disp. [$m$]')
sps[-1].set_xlabel('Time [s]')
sps[0].legend(prop={'size': 6}, ncol=3)
save_fig('Detrend_DataLeft')

##
plt.figure(figsize=(15,8))
bf, sps = plt.subplots(nrows=3, sharex='col')
acc = data_right['Linear Acceleration z (m/s^2)'][12000:60000] #min 2 a min 10
asig = eqsig.AccSignal(acc, dt, label='name_of_record')
asig.remove_poly(poly_fit=1)  # Remove any trend
sps[0].plot(asig.time, asig.values, label='Original')
sps[1].plot(asig.time, asig.velocity)
sps[2].plot(asig.time, asig.displacement)

# Add a trend
asig.add_series(np.linspace(0, 0.1, asig.npts))
sps[0].plot(asig.time, asig.values, ls='--', label='with trend')
sps[1].plot(asig.time, asig.velocity, ls='--')
sps[2].plot(asig.time, asig.displacement, ls='--')

# remove the trend
asig.remove_poly(poly_fit=1)
sps[0].plot(asig.time, asig.values, ls=':', label='trend removed')
sps[1].plot(asig.time, asig.velocity, ls=':')
sps[2].plot(asig.time, asig.displacement, ls=':')
sps[0].set_ylabel('Accel. [$m/s^2$]')
sps[1].set_ylabel('Velo. [$m/s$]')
sps[2].set_ylabel('Disp. [$m$]')
sps[-1].set_xlabel('Time [s]')
sps[0].legend(prop={'size': 6}, ncol=3)
save_fig('Detrend_DataRight')
#=================== Calculate SDOF response spectra (https://github.com/eng-tools/eqsig/blob/master/examples/example_response_spectra.ipynb)

acc = data_mid['Linear Acceleration z (m/s^2)'][12000:60000] #min 2 a min 10
dt = 0.01

periods = np.linspace(0.01, 5, 40)
spectral_disp, spectral_velo, spectral_acc = eqsig.sdof.pseudo_response_spectra(acc, dt, periods, xi=0.05)

bf, sps = plt.subplots(nrows=3, figsize=(6, 6))

sps[0].plot(periods, spectral_acc / 9.8)
sps[1].plot(periods, spectral_velo)
sps[2].plot(periods, spectral_disp)

sps[0].set_ylabel('$S_a$ [g]')
sps[1].set_ylabel('$S_v$ [m/s]')
sps[2].set_ylabel('$S_d$ [m]')
sps[2].set_xlabel('Period [s]')
save_fig('SDOF_DataMid')

#
acc = data_left['Linear Acceleration z (m/s^2)'][12000:60000] #min 2 a min 10
dt = 0.01

periods = np.linspace(0.01, 5, 40)
spectral_disp, spectral_velo, spectral_acc = eqsig.sdof.pseudo_response_spectra(acc, dt, periods, xi=0.05)

bf, sps = plt.subplots(nrows=3, figsize=(6, 6))
sps[0].plot(periods, spectral_acc / 9.8)
sps[1].plot(periods, spectral_velo)
sps[2].plot(periods, spectral_disp)

sps[0].set_ylabel('$S_a$ [g]')
sps[1].set_ylabel('$S_v$ [m/s]')
sps[2].set_ylabel('$S_d$ [m]')
sps[2].set_xlabel('Period [s]')
save_fig('SDOF_DataLeft')

#
acc = data_right['Linear Acceleration z (m/s^2)'][12000:60000] #min 2 a min 10
dt = 0.01

periods = np.linspace(0.01, 5, 40)
spectral_disp, spectral_velo, spectral_acc = eqsig.sdof.pseudo_response_spectra(acc, dt, periods, xi=0.05)

bf, sps = plt.subplots(nrows=3, figsize=(6, 6))

sps[0].plot(periods, spectral_acc / 9.8)
sps[1].plot(periods, spectral_velo)
sps[2].plot(periods, spectral_disp)

sps[0].set_ylabel('$S_a$ [g]')
sps[1].set_ylabel('$S_v$ [m/s]')
sps[2].set_ylabel('$S_d$ [m]')
sps[2].set_xlabel('Period [s]')
save_fig('SDOF_DataRight')

#==============0000Elastic response time series
periods = np.array([0.5, 2.0])
acc = data_mid['Linear Acceleration z (m/s^2)'][12000:60000] #min 2 a min 10
dt = 0.01
response_disp, response_velo, response_accel = eqsig.sdof.response_series(acc, dt, periods, xi=0.05)

time = np.arange(0, len(acc)) * dt

bf, sps = plt.subplots(nrows=3, figsize=(6, 6))
sps[0].plot(time, response_accel[0] / 9.8, label='T=%.2fs' % periods[0])
sps[1].plot(time, response_velo[0])
sps[2].plot(time, response_disp[0])
sps[0].plot(time, response_accel[1] / 9.8, label='T=%.2fs' % periods[1])
sps[1].plot(time, response_velo[1])
sps[2].plot(time, response_disp[1])

sps[0].set_ylabel('Resp. Accel. [g]')
sps[1].set_ylabel('Resp. Velo [m/s]')
sps[2].set_ylabel('Resp Disp [m]')
sps[2].set_xlabel('Time [s]')
sps[0].legend()
save_fig('ElasticResponse_DataMid')

##
periods = np.array([0.5, 2.0])
acc = data_left['Linear Acceleration z (m/s^2)'][12000:60000] #min 2 a min 10
dt = 0.01
response_disp, response_velo, response_accel = eqsig.sdof.response_series(acc, dt, periods, xi=0.05)

time = np.arange(0, len(acc)) * dt

bf, sps = plt.subplots(nrows=3, figsize=(6, 6))
sps[0].plot(time, response_accel[0] / 9.8, label='T=%.2fs' % periods[0])
sps[1].plot(time, response_velo[0])
sps[2].plot(time, response_disp[0])
sps[0].plot(time, response_accel[1] / 9.8, label='T=%.2fs' % periods[1])
sps[1].plot(time, response_velo[1])
sps[2].plot(time, response_disp[1])

sps[0].set_ylabel('Resp. Accel. [g]')
sps[1].set_ylabel('Resp. Velo [m/s]')
sps[2].set_ylabel('Resp Disp [m]')
sps[2].set_xlabel('Time [s]')
sps[0].legend()
save_fig('ElasticResponse_DataLeft')

##
periods = np.array([0.5, 2.0])
acc = data_right['Linear Acceleration z (m/s^2)'][12000:60000] #min 2 a min 10 #total:127004
dt = 0.01
response_disp, response_velo, response_accel = eqsig.sdof.response_series(acc, dt, periods, xi=0.05)

time = np.arange(0, len(acc)) * dt

bf, sps = plt.subplots(nrows=3, figsize=(6, 6))
sps[0].plot(time, response_accel[0] / 9.8, label='T=%.2fs' % periods[0])
sps[1].plot(time, response_velo[0])
sps[2].plot(time, response_disp[0])
sps[0].plot(time, response_accel[1] / 9.8, label='T=%.2fs' % periods[1])
sps[1].plot(time, response_velo[1])
sps[2].plot(time, response_disp[1])

sps[0].set_ylabel('Resp. Accel. [g]')
sps[1].set_ylabel('Resp. Velo [m/s]')
sps[2].set_ylabel('Resp Disp [m]')
sps[2].set_xlabel('Time [s]')
sps[0].legend()
save_fig('ElasticResponse_DataRight')


#======================PUENTE VIEJO ============================================================
import eqsig.single
bf, sub_fig = plt.subplots()
a = data_mid_viejo['Linear Acceleration z (m/s^2)'][12000:60000] #min 2 a min 10
dt = 0.005  # time step of acceleration time series
periods = np.linspace(0.2, 5, 100)  # compute the response for 100 periods between T=0.2s and 5.0s
record = eqsig.AccSignal(a * 9.8, dt)
record.generate_response_spectrum(response_times=periods)
times = record.response_times
sub_fig.plot(times, record.s_a, label="eqsig")
save_fig('Data_mid_viejo')

bf, sub_fig = plt.subplots()
a = data_left_viejo['Linear Acceleration z (m/s^2)'][12000:60000] #min 2 a min 10
dt = 0.005  # time step of acceleration time series
periods = np.linspace(0.2, 5, 100)  # compute the response for 100 periods between T=0.2s and 5.0s
record = eqsig.AccSignal(a * 9.8, dt)
record.generate_response_spectrum(response_times=periods)
times = record.response_times
sub_fig.plot(times, record.s_a, label="eqsig")
save_fig('Data_left_viejo')

bf, sub_fig = plt.subplots()
a = data_right_viejo['Linear Acceleration z (m/s^2)'][12000:60000] #min 2 a min 10
dt = 0.005  # time step of acceleration time series
periods = np.linspace(0.2, 5, 100)  # compute the response for 100 periods between T=0.2s and 5.0s
record = eqsig.AccSignal(a * 9.8, dt)
record.generate_response_spectrum(response_times=periods)
times = record.response_times
sub_fig.plot(times, record.s_a, label="eqsig")
save_fig('Data_right_viejo')

#========== Produce Fourier spectrum and smoothed Fourier spectrum (https://github.com/eng-tools/eqsig/blob/master/examples/example_filtering_and_fourier_spectrum.ipynb)
plt.figure(figsize=(15,8))
acc = data_mid_viejo['Linear Acceleration z (m/s^2)'][12000:60000] #min 2 a min 10
dt = 0.01
asig = eqsig.AccSignal(acc, dt, label='name_of_record')
plt.plot(asig.fa_frequencies, abs(asig.fa_spectrum), c='b')
asig.smooth_fa_frequencies = np.logspace(-1, 1, 50)
plt.loglog(asig.smooth_fa_frequencies, asig.smooth_fa_spectrum, c='r', ls='--')
plt.xlabel('Frequency [Hz]')
lab = plt.ylabel('Fourier Amplitude [m/s]')
save_fig('Fourier_spectrum_DataMid_viejo')

##=====Filter record using  filter
plt.figure(figsize=(15,8))
plt.plot(asig.fa_frequencies, abs(asig.fa_spectrum), c='b')
asig.butter_pass((None, 8))  # Low pass filter at 8Hz (default 4th order)
plt.plot(asig.fa_frequencies, abs(asig.fa_spectrum), c='r', ls='--')
asig.butter_pass((0.1, None), filter_order=4)  # High pass filter at 0.1Hz (default 4th order)
plt.plot(asig.fa_frequencies, abs(asig.fa_spectrum), c='g', ls=':')
asig.butter_pass((0.2, 6))  # Band pass filter at 0.2 and 6Hz (default 4th order)
plt.loglog(asig.fa_frequencies, abs(asig.fa_spectrum), c='k', ls='--', lw=0.7)
plt.xlabel('Frequency [Hz]')
lab = plt.ylabel('Fourier Amplitude [m/s]')
save_fig('Butterworth_DataMid_viejo')

#===== Detrend a record
plt.figure(figsize=(15,8))
bf, sps = plt.subplots(nrows=3, sharex='col')
acc = data_mid_viejo['Linear Acceleration z (m/s^2)'][12000:60000] #min 2 a min 10
asig = eqsig.AccSignal(acc, dt, label='name_of_record')
asig.remove_poly(poly_fit=1)  # Remove any trend
sps[0].plot(asig.time, asig.values, label='Original')
sps[1].plot(asig.time, asig.velocity)
sps[2].plot(asig.time, asig.displacement)

# Add a trend
asig.add_series(np.linspace(0, 0.1, asig.npts))
sps[0].plot(asig.time, asig.values, ls='--', label='with trend')
sps[1].plot(asig.time, asig.velocity, ls='--')
sps[2].plot(asig.time, asig.displacement, ls='--')

# remove the trend
asig.remove_poly(poly_fit=1)
sps[0].plot(asig.time, asig.values, ls=':', label='trend removed')
sps[1].plot(asig.time, asig.velocity, ls=':')
sps[2].plot(asig.time, asig.displacement, ls=':')
sps[0].set_ylabel('Accel. [$m/s^2$]')
sps[1].set_ylabel('Velo. [$m/s$]')
sps[2].set_ylabel('Disp. [$m$]')
sps[-1].set_xlabel('Time [s]')
sps[0].legend(prop={'size': 6}, ncol=3)
save_fig('Detrend_DataMid_viejo')

##
plt.figure(figsize=(15,8))
bf, sps = plt.subplots(nrows=3, sharex='col')
acc = data_left_viejo['Linear Acceleration z (m/s^2)'][12000:60000] #min 2 a min 10
asig = eqsig.AccSignal(acc, dt, label='name_of_record')
asig.remove_poly(poly_fit=1)  # Remove any trend
sps[0].plot(asig.time, asig.values, label='Original')
sps[1].plot(asig.time, asig.velocity)
sps[2].plot(asig.time, asig.displacement)

# Add a trend
asig.add_series(np.linspace(0, 0.1, asig.npts))
sps[0].plot(asig.time, asig.values, ls='--', label='with trend')
sps[1].plot(asig.time, asig.velocity, ls='--')
sps[2].plot(asig.time, asig.displacement, ls='--')

# remove the trend
asig.remove_poly(poly_fit=1)
sps[0].plot(asig.time, asig.values, ls=':', label='trend removed')
sps[1].plot(asig.time, asig.velocity, ls=':')
sps[2].plot(asig.time, asig.displacement, ls=':')
sps[0].set_ylabel('Accel. [$m/s^2$]')
sps[1].set_ylabel('Velo. [$m/s$]')
sps[2].set_ylabel('Disp. [$m$]')
sps[-1].set_xlabel('Time [s]')
sps[0].legend(prop={'size': 6}, ncol=3)
save_fig('Detrend_DataLeft_viejo')

##
plt.figure(figsize=(15,8))
bf, sps = plt.subplots(nrows=3, sharex='col')
acc = data_right_viejo['Linear Acceleration z (m/s^2)'][12000:60000] #min 2 a min 10
asig = eqsig.AccSignal(acc, dt, label='name_of_record')
asig.remove_poly(poly_fit=1)  # Remove any trend
sps[0].plot(asig.time, asig.values, label='Original')
sps[1].plot(asig.time, asig.velocity)
sps[2].plot(asig.time, asig.displacement)

# Add a trend
asig.add_series(np.linspace(0, 0.1, asig.npts))
sps[0].plot(asig.time, asig.values, ls='--', label='with trend')
sps[1].plot(asig.time, asig.velocity, ls='--')
sps[2].plot(asig.time, asig.displacement, ls='--')

# remove the trend
asig.remove_poly(poly_fit=1)
sps[0].plot(asig.time, asig.values, ls=':', label='trend removed')
sps[1].plot(asig.time, asig.velocity, ls=':')
sps[2].plot(asig.time, asig.displacement, ls=':')
sps[0].set_ylabel('Accel. [$m/s^2$]')
sps[1].set_ylabel('Velo. [$m/s$]')
sps[2].set_ylabel('Disp. [$m$]')
sps[-1].set_xlabel('Time [s]')
sps[0].legend(prop={'size': 6}, ncol=3)
save_fig('Detrend_DataRight_viejo')
#=================== Calculate SDOF response spectra (https://github.com/eng-tools/eqsig/blob/master/examples/example_response_spectra.ipynb)

acc = data_mid_viejo['Linear Acceleration z (m/s^2)'][12000:60000] #min 2 a min 10
dt = 0.01

periods = np.linspace(0.01, 5, 40)
spectral_disp, spectral_velo, spectral_acc = eqsig.sdof.pseudo_response_spectra(acc, dt, periods, xi=0.05)

bf, sps = plt.subplots(nrows=3, figsize=(6, 6))

sps[0].plot(periods, spectral_acc / 9.8)
sps[1].plot(periods, spectral_velo)
sps[2].plot(periods, spectral_disp)

sps[0].set_ylabel('$S_a$ [g]')
sps[1].set_ylabel('$S_v$ [m/s]')
sps[2].set_ylabel('$S_d$ [m]')
sps[2].set_xlabel('Period [s]')
save_fig('SDOF_DataMid_viejo')

#
acc = data_left_viejo['Linear Acceleration z (m/s^2)'][12000:60000] #min 2 a min 10
dt = 0.01

periods = np.linspace(0.01, 5, 40)
spectral_disp, spectral_velo, spectral_acc = eqsig.sdof.pseudo_response_spectra(acc, dt, periods, xi=0.05)

bf, sps = plt.subplots(nrows=3, figsize=(6, 6))
sps[0].plot(periods, spectral_acc / 9.8)
sps[1].plot(periods, spectral_velo)
sps[2].plot(periods, spectral_disp)

sps[0].set_ylabel('$S_a$ [g]')
sps[1].set_ylabel('$S_v$ [m/s]')
sps[2].set_ylabel('$S_d$ [m]')
sps[2].set_xlabel('Period [s]')
save_fig('SDOF_DataLeft_viejo')

#
acc = data_right_viejo['Linear Acceleration z (m/s^2)'][12000:60000] #min 2 a min 10
dt = 0.01

periods = np.linspace(0.01, 5, 40)
spectral_disp, spectral_velo, spectral_acc = eqsig.sdof.pseudo_response_spectra(acc, dt, periods, xi=0.05)

bf, sps = plt.subplots(nrows=3, figsize=(6, 6))

sps[0].plot(periods, spectral_acc / 9.8)
sps[1].plot(periods, spectral_velo)
sps[2].plot(periods, spectral_disp)

sps[0].set_ylabel('$S_a$ [g]')
sps[1].set_ylabel('$S_v$ [m/s]')
sps[2].set_ylabel('$S_d$ [m]')
sps[2].set_xlabel('Period [s]')
save_fig('SDOF_DataRight_viejo')

#==============0000Elastic response time series
periods = np.array([0.5, 2.0])
acc = data_mid_viejo['Linear Acceleration z (m/s^2)'][12000:60000] #min 2 a min 10
dt = 0.01
response_disp, response_velo, response_accel = eqsig.sdof.response_series(acc, dt, periods, xi=0.05)

time = np.arange(0, len(acc)) * dt

bf, sps = plt.subplots(nrows=3, figsize=(6, 6))
sps[0].plot(time, response_accel[0] / 9.8, label='T=%.2fs' % periods[0])
sps[1].plot(time, response_velo[0])
sps[2].plot(time, response_disp[0])
sps[0].plot(time, response_accel[1] / 9.8, label='T=%.2fs' % periods[1])
sps[1].plot(time, response_velo[1])
sps[2].plot(time, response_disp[1])

sps[0].set_ylabel('Resp. Accel. [g]')
sps[1].set_ylabel('Resp. Velo [m/s]')
sps[2].set_ylabel('Resp Disp [m]')
sps[2].set_xlabel('Time [s]')
sps[0].legend()
save_fig('ElasticResponse_DataMid_viejo')

##
periods = np.array([0.5, 2.0])
acc = data_left_viejo['Linear Acceleration z (m/s^2)'][12000:60000] #min 2 a min 10
dt = 0.01
response_disp, response_velo, response_accel = eqsig.sdof.response_series(acc, dt, periods, xi=0.05)

time = np.arange(0, len(acc)) * dt

bf, sps = plt.subplots(nrows=3, figsize=(6, 6))
sps[0].plot(time, response_accel[0] / 9.8, label='T=%.2fs' % periods[0])
sps[1].plot(time, response_velo[0])
sps[2].plot(time, response_disp[0])
sps[0].plot(time, response_accel[1] / 9.8, label='T=%.2fs' % periods[1])
sps[1].plot(time, response_velo[1])
sps[2].plot(time, response_disp[1])

sps[0].set_ylabel('Resp. Accel. [g]')
sps[1].set_ylabel('Resp. Velo [m/s]')
sps[2].set_ylabel('Resp Disp [m]')
sps[2].set_xlabel('Time [s]')
sps[0].legend()
save_fig('ElasticResponse_DataLeft_viejo')

##
periods = np.array([0.5, 2.0])
acc = data_right_viejo['Linear Acceleration z (m/s^2)'][12000:60000] #min 2 a min 10 #total:127004
dt = 0.01
response_disp, response_velo, response_accel = eqsig.sdof.response_series(acc, dt, periods, xi=0.05)

time = np.arange(0, len(acc)) * dt

bf, sps = plt.subplots(nrows=3, figsize=(6, 6))
sps[0].plot(time, response_accel[0] / 9.8, label='T=%.2fs' % periods[0])
sps[1].plot(time, response_velo[0])
sps[2].plot(time, response_disp[0])
sps[0].plot(time, response_accel[1] / 9.8, label='T=%.2fs' % periods[1])
sps[1].plot(time, response_velo[1])
sps[2].plot(time, response_disp[1])

sps[0].set_ylabel('Resp. Accel. [g]')
sps[1].set_ylabel('Resp. Velo [m/s]')
sps[2].set_ylabel('Resp Disp [m]')
sps[2].set_xlabel('Time [s]')
sps[0].legend()
save_fig('ElasticResponse_DataRight_viejo')





#======================= FFT

# Python example - Fourier transform using numpy.fft method
import numpy as np
import matplotlib.pyplot as plotter

# How many time points are needed i,e., Sampling Frequency
samplingFrequency   = 100;
# At what intervals time points are sampled
samplingInterval       = 1 / samplingFrequency;
# Begin time period of the signals
beginTime           = 0;
# End time period of the signals
endTime             = 10; 
# Frequency of the signals
signal1Frequency     = 4;
signal2Frequency     = 7;
# Time points
time        = np.arange(beginTime, endTime, samplingInterval);
# Create two sine waves
amplitude1 = np.sin(2*np.pi*signal1Frequency*time)
amplitude2 = np.sin(2*np.pi*signal2Frequency*time)
# Create subplot
figure, axis = plotter.subplots(4, 1)
plotter.subplots_adjust(hspace=1)
# Time domain representation for sine wave 1
axis[0].set_title('Sine wave with a frequency of 4 Hz')
axis[0].plot(time, amplitude1)
axis[0].set_xlabel('Time')
axis[0].set_ylabel('Amplitude')
 # Time domain representation for sine wave 2
axis[1].set_title('Sine wave with a frequency of 7 Hz')
axis[1].plot(time, amplitude2)
axis[1].set_xlabel('Time')
axis[1].set_ylabel('Amplitude')
# Add the sine waves
amplitude = amplitude1 + amplitude2
# Time domain representation of the resultant sine wave
axis[2].set_title('Sine wave with multiple frequencies')
axis[2].plot(time, amplitude)
axis[2].set_xlabel('Time')
axis[2].set_ylabel('Amplitude')

# Frequency domain representation
fourierTransform = np.fft.fft(data_right_viejo['Linear Acceleration z (m/s^2)'][12000:60000])/len(data_right_viejo['Linear Acceleration z (m/s^2)'][12000:60000])           # Normalize amplitude
fourierTransform = fourierTransform[range(int(len(data_right_viejo['Linear Acceleration z (m/s^2)'][12000:60000])/2))] # Exclude sampling frequency

tpCount     = len(data_right_viejo['Linear Acceleration z (m/s^2)'][12000:60000])
values      = np.arange(int(tpCount/2))
timePeriod  = tpCount/0.01
frequencies = values/timePeriod

# Frequency domain representation
plt.set_title('Fourier transform depicting the frequency components')
plt.plot(frequencies, abs(fourierTransform))
plt.xlabel('Frequency')
plt.ylabel('Acceleration')
save_fig('FFT_right_viejo')