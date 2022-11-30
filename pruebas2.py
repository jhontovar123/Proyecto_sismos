import numpy as np
import matplotlib.pyplot as plt
import eqsig.sdof
from scipy import integrate
from scipy import fft
from scipy import signal
from scipy.signal import butter, lfilter
from scipy.signal import freqz
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import collections
import seaborn as sns
import csv
import os
import pandas as pd


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

# ==== Part 1: Preprocessing the data

#
data_mid = df_15luz.copy()
data_left = df_125luz.copy()
data_right = df_175luz.copy()

data_mid_viejo = df_15luz_viejo.copy()
data_left_viejo = df_125luz_viejo.copy()
data_right_viejo = df_175luz_viejo.copy()

dt=0.01 #intervalo de tiempo
fm=2000 #Frecuencia de muestreo
t= np.arange (0, 480, dt)
#MEDIO
VEW=integrate.cumtrapz (data_mid['Linear Acceleration z (m/s^2)'][12000:60000], t, initial=0)
DEW=integrate.cumtrapz (VEW, t, initial=0)
VEW2=integrate.cumtrapz (data_mid_viejo['Linear Acceleration z (m/s^2)'][12000:60000], t, initial=0)
DEW2=integrate.cumtrapz (VEW2, t, initial=0)
plt.figure()
plt.figure(figsize=(15,8))
plt.subplot (3,2,1); plt.plot (t, data_mid['Linear Acceleration z (m/s^2)'][12000:60000])
plt.title (['SIN TRATAMIENTO COMPONENTE Puente 1'])
plt.xlabel ('tiempo(seg)')
plt.ylabel ('Aceleracion (cm/seg2)')
plt.subplot (3,2,2); plt.plot (t, data_mid_viejo['Linear Acceleration z (m/s^2)'][12000:60000])
plt.title (['SIN TRATAMIENTO COMPONENTE Puente 2'])
plt.xlabel ('tiempo(seg)')
plt.ylabel ('Aceleracion (cm/seg2)')
plt.subplot (3,2,3); plt.plot (t, VEW)
plt.xlabel ('tiempo(seg)')
plt.ylabel ('Velocidad (cm/seg)')
plt.subplot (3,2,4); plt.plot (t, VEW2)
plt.xlabel ('tiempo(seg)')
plt.ylabel ('Velocidad (cm/seg)')
plt.subplot(3,2,5); plt.plot (t, DEW)
plt.xlabel ('tiempo (seg)')
plt.ylabel ('Desplazamiento (cm)')
plt.subplot(3,2,6); plt.plot (t, DEW2)
plt.xlabel ('tiempo (seg)')
plt.ylabel ('Desplazamiento (cm)')
save_fig("Acel_Vel_Desp_medio")

#Izquierda
VEW=integrate.cumtrapz (data_left['Linear Acceleration z (m/s^2)'][12000:60000], t, initial=0)
DEW=integrate.cumtrapz (VEW, t, initial=0)
VEW2=integrate.cumtrapz (data_left_viejo['Linear Acceleration z (m/s^2)'][12000:60000], t, initial=0)
DEW2=integrate.cumtrapz (VEW2, t, initial=0)
plt.figure()
plt.figure(figsize=(15,8))
plt.subplot (3,2,1); plt.plot (t, data_left['Linear Acceleration z (m/s^2)'][12000:60000])
plt.title (['SIN TRATAMIENTO COMPONENTE Puente 1'])
plt.xlabel ('tiempo(seg)')
plt.ylabel ('Aceleracion (cm/seg2)')
plt.subplot (3,2,2); plt.plot (t, data_left_viejo['Linear Acceleration z (m/s^2)'][12000:60000])
plt.title (['SIN TRATAMIENTO COMPONENTE Puente 2'])
plt.xlabel ('tiempo(seg)')
plt.ylabel ('Aceleracion (cm/seg2)')
plt.subplot (3,2,3); plt.plot (t, VEW)
plt.xlabel ('tiempo(seg)')
plt.ylabel ('Velocidad (cm/seg)')
plt.subplot (3,2,4); plt.plot (t, VEW2)
plt.xlabel ('tiempo(seg)')
plt.ylabel ('Velocidad (cm/seg)')
plt.subplot(3,2,5); plt.plot (t, DEW)
plt.xlabel ('tiempo (seg)')
plt.ylabel ('Desplazamiento (cm)')
plt.subplot(3,2,6); plt.plot (t, DEW2)
plt.xlabel ('tiempo (seg)')
plt.ylabel ('Desplazamiento (cm)')
save_fig("Acel_Vel_Desp_izquierda")

#Derecha
VEW=integrate.cumtrapz (data_right['Linear Acceleration z (m/s^2)'][12000:60000], t, initial=0)
DEW=integrate.cumtrapz (VEW, t, initial=0)
VEW2=integrate.cumtrapz (data_right_viejo['Linear Acceleration z (m/s^2)'][12000:60000], t, initial=0)
DEW2=integrate.cumtrapz (VEW2, t, initial=0)
plt.figure()
plt.figure(figsize=(15,8))
plt.subplot (3,2,1); plt.plot (t, data_right['Linear Acceleration z (m/s^2)'][12000:60000])
plt.title (['SIN TRATAMIENTO COMPONENTE Puente 1'])
plt.xlabel ('tiempo(seg)')
plt.ylabel ('Aceleracion (cm/seg2)')
plt.subplot (3,2,2); plt.plot (t, data_right_viejo['Linear Acceleration z (m/s^2)'][12000:60000])
plt.title (['SIN TRATAMIENTO COMPONENTE Puente 2'])
plt.xlabel ('tiempo(seg)')
plt.ylabel ('Aceleracion (cm/seg2)')
plt.subplot (3,2,3); plt.plot (t, VEW)
plt.xlabel ('tiempo(seg)')
plt.ylabel ('Velocidad (cm/seg)')
plt.subplot (3,2,4); plt.plot (t, VEW2)
plt.xlabel ('tiempo(seg)')
plt.ylabel ('Velocidad (cm/seg)')
plt.subplot(3,2,5); plt.plot (t, DEW)
plt.xlabel ('tiempo (seg)')
plt.ylabel ('Desplazamiento (cm)')
plt.subplot(3,2,6); plt.plot (t, DEW2)
plt.xlabel ('tiempo (seg)')
plt.ylabel ('Desplazamiento (cm)')
save_fig("Acel_Vel_Desp_derecha")

n=len(t)
fr=(fm/2) * np.linspace (0,1, n//2)
x1=np.fft.fft(data_mid['Linear Acceleration z (m/s^2)'][12000:14000], n)
xm1=x1* np.conjugate (x1) /n
x_m1=xm1 [0: np.size (fr)]
x2=np.fft.fft (data_mid_viejo['Linear Acceleration z (m/s^2)'][12000:14000], n)
xm2=x2 * np.conjugate (x2) /n
x_m2=xm2 [0: np.size (fr)]

plt.figure()
plt.figure(figsize=(15,8))
plt.subplot (1,2,1); plt.plot (fr, x_m1)
plt.title ('SIN TRATAMIENTO COMPONENTE PN')
plt.xlabel ('Frecuencia (Hz)')
plt.ylabel ('Magnitud')
plt.subplot (1,2,2); plt.plot (fr,x_m2)
plt.title ('SIN TRATAMIENTO COMPONENTE PV')
plt.xlabel ('Frecuencia (Hz)')
plt.ylabel ('Magnitud')
save_fig("pruebitaaaaas data_mid222")



#con tratamiento
#Corregir línea base
#medio
adet =signal.detrend (data_mid['Linear Acceleration z (m/s^2)'][12000:60000])
vdet=integrate.cumtrapz (adet, t, initial=0)
ddet=integrate.cumtrapz (vdet, t, initial=0)
adet2=signal.detrend (data_mid_viejo['Linear Acceleration z (m/s^2)'][12000:60000])
vdet2=integrate.cumtrapz(adet2, t, initial=0)
ddet2=integrate.cumtrapz(vdet2, t, initial=0)

adett=pd.DataFrame()
adett['Aceleracion cm/seg2']=adet
adett['Velocidad cm/vel']=vdet
adett['Desplazamiento cm']=ddet
adett.to_excel(".\\Salidas\\Acel_Puente1_medio.xlsx", index=False)

adett=pd.DataFrame()
adett['Aceleracion cm/seg2']=adet2
adett['Velocidad cm/vel']=vdet2
adett['Desplazamiento cm']=ddet2
adett.to_excel(".\\Salidas\\Acel_Puente2_medio.xlsx", index=False)

plt.figure()
plt.figure(figsize=(15,8))
plt.subplot (3,2,1);plt.title ('Corrección de línea base COMPONENTE Puente 1')
plt.plot (t, adet)
plt.xlabel ('Tiempo(seg)')
plt.ylabel ('Aceleracion (cm/seg2)')
plt.subplot (3,2,3); plt.plot (t, vdet)
plt.xlabel ('Tiempo(seg)')
plt.ylabel ('Velocidad (cm/seg)')
plt.subplot (3,2,5); plt.plot (t, ddet)
plt.xlabel ('Tiempo(seg)')
plt.ylabel ('Desplazamiento (cm)')
plt.subplot (3,2,2); plt.title ('Corrección de línea base COMPONENTE Puente 2')
plt.plot (t,adet2)
plt.xlabel ('Tiempo(seg)')
plt.ylabel ('Aceleracion (cm/seg2)')
plt.subplot (3,2,4);plt.plot(t,vdet2)
plt.xlabel ('Tiempo(seg)')
plt.ylabel ('Velocidad (cm/seg)')
plt.subplot (3,2,6);plt.plot(t,ddet2)
plt.xlabel ('Tiempo(seg)')
plt.ylabel ('Desplazamiento (cm)')
save_fig("Acel_Vel_Desp_CORREGIDO_medio")

#Izquierda
adet =signal.detrend (data_left['Linear Acceleration z (m/s^2)'][12000:60000])
vdet=integrate.cumtrapz (adet, t, initial=0)
ddet=integrate.cumtrapz (vdet, t, initial=0)
adet2=signal.detrend (data_left_viejo['Linear Acceleration z (m/s^2)'][12000:60000])
vdet2=integrate.cumtrapz(adet2, t, initial=0)
ddet2=integrate.cumtrapz(vdet2, t, initial=0)

adett=pd.DataFrame()
adett['Aceleracion cm/seg2']=adet
adett['Velocidad cm/vel']=vdet
adett['Desplazamiento cm']=ddet
adett.to_excel(".\\Salidas\\Acel_Puente1_izquierda.xlsx", index=False)

adett=pd.DataFrame()
adett['Aceleracion cm/seg2']=adet2
adett['Velocidad cm/vel']=vdet2
adett['Desplazamiento cm']=ddet2
adett.to_excel(".\\Salidas\\Acel_Puente2_izquierda.xlsx", index=False)

plt.figure()
plt.figure(figsize=(15,8))
plt.subplot (3,2,1);plt.title ('Corrección de línea base COMPONENTE Puente 1')
plt.plot (t, adet)
plt.xlabel ('Tiempo(seg)')
plt.ylabel ('Aceleracion (cm/seg2)')
plt.subplot (3,2,3); plt.plot (t, vdet)
plt.xlabel ('Tiempo(seg)')
plt.ylabel ('Velocidad (cm/seg)')
plt.subplot (3,2,5); plt.plot (t, ddet)
plt.xlabel ('Tiempo(seg)')
plt.ylabel ('Desplazamiento (cm)')
plt.subplot (3,2,2); plt.title ('Corrección de línea base COMPONENTE Puente 2')
plt.plot (t,adet2)
plt.xlabel ('Tiempo(seg)')
plt.ylabel ('Aceleracion (cm/seg2)')
plt.subplot (3,2,4);plt.plot(t,vdet2)
plt.xlabel ('Tiempo(seg)')
plt.ylabel ('Velocidad (cm/seg)')
plt.subplot (3,2,6);plt.plot(t,ddet2)
plt.xlabel ('Tiempo(seg)')
plt.ylabel ('Desplazamiento (cm)')
save_fig("Acel_Vel_Desp_CORREGIDO_izquierda")

#Derecha
adet =signal.detrend (data_right['Linear Acceleration z (m/s^2)'][12000:60000])
vdet=integrate.cumtrapz (adet, t, initial=0)
ddet=integrate.cumtrapz (vdet, t, initial=0)
adet2=signal.detrend (data_right_viejo['Linear Acceleration z (m/s^2)'][12000:60000])
vdet2=integrate.cumtrapz(adet2, t, initial=0)
ddet2=integrate.cumtrapz(vdet2, t, initial=0)

adett=pd.DataFrame()
adett['Aceleracion cm/seg2']=adet
adett['Velocidad cm/vel']=vdet
adett['Desplazamiento cm']=ddet
adett.to_excel(".\\Salidas\\Acel_Puente1_derecha.xlsx", index=False)

adett=pd.DataFrame()
adett['Aceleracion cm/seg2']=adet2
adett['Velocidad cm/vel']=vdet2
adett['Desplazamiento cm']=ddet2
adett.to_excel(".\\Salidas\\Acel_Puente2_derecha.xlsx", index=False)


plt.figure()
plt.figure(figsize=(15,8))
plt.subplot (3,2,1);plt.title ('Corrección de línea base COMPONENTE Puente 1')
plt.plot (t, adet)
plt.xlabel ('Tiempo(seg)')
plt.ylabel ('Aceleracion (cm/seg2)')
plt.subplot (3,2,3); plt.plot (t, vdet)
plt.xlabel ('Tiempo(seg)')
plt.ylabel ('Velocidad (cm/seg)')
plt.subplot (3,2,5); plt.plot (t, ddet)
plt.xlabel ('Tiempo(seg)')
plt.ylabel ('Desplazamiento (cm)')
plt.subplot (3,2,2); plt.title ('Corrección de línea base COMPONENTE Puente 2')
plt.plot (t,adet2)
plt.xlabel ('Tiempo(seg)')
plt.ylabel ('Aceleracion (cm/seg2)')
plt.subplot (3,2,4);plt.plot(t,vdet2)
plt.xlabel ('Tiempo(seg)')
plt.ylabel ('Velocidad (cm/seg)')
plt.subplot (3,2,6);plt.plot(t,ddet2)
plt.xlabel ('Tiempo(seg)')
plt.ylabel ('Desplazamiento (cm)')
save_fig("Acel_Vel_Desp_CORREGIDO_derecha")