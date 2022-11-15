
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

fname1 = 'Ace_1.5luz.csv'  # 'main_PC_database.xlsx', 'main_PC_database_v2.xlsx'   # <-------
fname_t1 = fpath + fname1
fname2 = 'Ace_1.25luz.csv'  # 'main_PC_database.xlsx', 'main_PC_database_v2.xlsx'   # <-------
fname_t2 = fpath + fname2
fname3 = 'Ace_1.75luz.csv'  # 'main_PC_database.xlsx', 'main_PC_database_v2.xlsx'   # <-------
fname_t3 = fpath + fname3
#
# --- Reading the data
df_15luz = pd.read_csv(fname_t1)
df_125luz = pd.read_csv(fname_t2)
df_175luz = pd.read_csv(fname_t3)
# --- Initial look of the data
print(df_15luz.head(20))
print(df_15luz.info())  # data types and general info
#
# ==== Part 2: Preprocessing the data
#
data_mid = df_15luz.copy()
data_left = df_125luz.copy()
data_right = df_175luz.copy()

#===== Ploteando data
plt.figure(figsize=(15,8))
plt.plot(data_right['Time (s)'], data_right['Linear Acceleration z (m/s^2)'],linewidth=0.7)
plt.xlabel('Time (s)')
plt.ylabel('Linear Acceleration Z (m/s^2)')
plt.xlim([60, 600]) # 60 es a 1 min del inicio y 600 es a 10 min
plt.ylim([-0.6, 0.1])
save_fig("Linear Acceleration Z_right_total")

np.max(data_mid['Linear Acceleration y (m/s^2)'])
plt.figure(figsize=(15,8))
plt.plot(data_right['Time (s)'], data_right['Linear Acceleration z (m/s^2)'],linewidth=0.7)
plt.xlabel('Time (s)')
plt.ylabel('Linear Acceleration Z (m/s^2)')
plt.xlim([60, 80]) # 60 es a 1 min del inicio y 600 es a 10 min
plt.ylim([-0.5, 0])
save_fig("Linear Acceleration Z_right_incial")

np.max(data_mid['Linear Acceleration z (m/s^2)'])
plt.figure(figsize=(15,8))
plt.plot(data_right['Time (s)'], data_right['Linear Acceleration z (m/s^2)'],linewidth=0.7)
plt.xlabel('Time (s)')
plt.ylabel('Linear Acceleration Z_inicial (m/s^2)')
plt.xlim([345, 365]) # 60 es a 1 min del inicio y 600 es a 10 min
plt.ylim([-0.45, -0.05])
save_fig("Linear Acceleration Z_right_midtime")


plt.figure(figsize=(15,8))
plt.plot(data_right['Time (s)'], data_right['Linear Acceleration z (m/s^2)'],linewidth=0.7)
plt.xlabel('Time (s)')
plt.ylabel('Linear Acceleration Z (m/s^2)')
plt.xlim([580, 600]) # 60 es a 1 min del inicio y 600 es a 10 min
plt.ylim([-0.5, 0])
save_fig("Linear Acceleration Z_right_final")


Fs = 150.0;                 # tasa de muestreo
Ts = 1.0/Fs;                # intervalo de muestreo Intervalo de muestreo
t = np.arange(0,1,Ts)       # vector de tiempo, aquí Ts también es el tamaño del paso

ff = 25;                    # frecuencia de la señal
y = np.sin(2*np.pi*ff*t)

n = len(y)                  # length of the signal
k = np.arange(n)
T = n/Fs
frq = k/T                   # two sides frequency range
frq1 = frq[range(int(n/2))] # one side frequency range

YY = np.fft.fft(y)          # Sin normalizar
Y = np.fft.fft(y)/n         # normalización de normalización y computación fft
Y1 = Y[range(int(n/2))]

fig, ax = plt.subplots(4, 1)

ax[0].plot(t,y)
ax[0].set_xlabel('Time')
ax[0].set_ylabel('Amplitude')

ax[1].plot(frq,abs(YY),'r') # plotting the spectrum
ax[1].set_xlabel('Freq (Hz)')
ax[1].set_ylabel('|Y(freq)|')

ax[2].plot(frq,abs(Y),'G')  # plotting the spectrum
ax[2].set_xlabel('Freq (Hz)')
ax[2].set_ylabel('|Y(freq)|')

ax[3].plot(frq1,abs(Y1),'B') # plotting the spectrum
ax[3].set_xlabel('Freq (Hz)')
ax[3].set_ylabel('|Y(freq)|')

plt.show()

data_mid1=data_mid[,]

data_mid['Linear Acceleration z (m/s^2)'].shape[0]

def get_circular_terms(N):
    """
    N: int 
    """

    terms =  np.exp(-1j *2*np.pi * np.arange(N)/N)

    return terms

def discrete_fourier_transform(data):

    N=data.shape[0]
    n=np.arange(N)
    k=n.reshape((N,1))
    M=np.exp(-1j*2*np.pi*k*n/N)
    
    return np.dot(M,data)

def fast_fourier_transform(data):
    """
    data: np.array  
        data as 1D array
    return discrete fourier transform of data
    """

    # len of data
    N = data.shape[0]

    # Must be a power of 2
    assert   N % 2 == 0, 'len of data: {} must be a power of 2'.format(N)

    if N<= 2:
        return discrete_fourier_transform(data)

    else:
        data_even = fast_fourier_transform(data[::2])
        data_odd = fast_fourier_transform(data[1::2])
        terms = get_circular_terms(N)

        return np.concatenate(
            [
            data_even + terms[:N//2] * data_odd,
            data_even + terms[N//2:] * data_odd 
            ])


M_dot=discrete_fourier_transform(data_mid['Linear Acceleration z (m/s^2)'][60:92])
Z_dot=fast_fourier_transform(data_mid['Absolute acceleration (m/s^2)'][60:92])

plt.figure(figsize=(15,8))
plt.plot(data_mid['Time (s)'][60:92], M_dot,linewidth=0.7)
plt.xlabel('Time (s)')
plt.ylabel('Absolute acceleration (m/s^2)')
#plt.xlim([60, 600]) # 60 es a 1 min del inicio y 600 es a 10 min
plt.ylim([-0.1, 0.1])
save_fig("Fourier Abs")

import scipy.fftpack as fourier


intervalo=(np.max(data_mid['Time (s)'])-np.min(data_mid['Time (s)']))/len(data_mid['Time (s)'])

Ts=data_mid['Time (s)'][65] - data_mid['Time (s)'][64] #Tiempo entre muestreo
Fs=1/Ts 
print(Ts)
print(Fs)
# TRANFORMADA RAPIDA DE FOURIER
f = np.fft.fft(data_left['Linear Acceleration z (m/s^2)'][60:60000])
freq = np.fft.fftfreq(len(data_mid['Linear Acceleration z (m/s^2)'][60:92]), d = data_mid['Time (s)'][61] - data_mid['Time (s)'][60])

np.min(f)
np.max(f)
np.mean(f)
np.std(f)

plt.figure(figsize=(15,8))
plt.plot(data_left['Time (s)'][60:60000], f,linewidth=0.7)
#plt.plot(data_mid['Time (s)'][60:92], freq,linewidth=0.7,color='red')
plt.xlabel('Time (s)')
#plt.ylabel('Absolute acceleration (m/s^2)')
#plt.xlim([60, 300]) # 60 es a 1 min del inicio y 600 es a 10 min
plt.ylim([-1, 1])
save_fig("Fourier left_total_60000")


area_acum=[]
for i in range(60,600):
    ancho=data_mid['Time (s)'][i+1]-data_mid['Time (s)'][i]
    Acel=np.abs(data_mid['Linear Acceleration z (m/s^2)'][i])
    area=ancho*Acel
    area_acum.append(area)


plt.figure(figsize=(15,8))
plt.scatter(data_mid['Time (s)'][60:600], area_acum,linewidth=0.7)
#plt.plot(data_mid['Time (s)'][60:92], freq,linewidth=0.7,color='red')
plt.xlabel('Time (s)')
#plt.ylabel('Absolute acceleration (m/s^2)')
#plt.xlim([60, 300]) # 60 es a 1 min del inicio y 600 es a 10 min
#plt.ylim([-1, 1])
save_fig("CAV")


i=6000
ancho=data_mid['Time (s)'][i+1]-data_mid['Time (s)'][i]
Acel=np.abs(data_mid['Linear Acceleration z (m/s^2)'][i])
area=ancho*Acel
area_acum.append(area)