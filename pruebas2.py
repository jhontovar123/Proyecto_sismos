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
adett['Tiempo (s)']=data_mid['Time (s)'][12000:60000]
adett['Aceleracion cm/seg2']=adet
adett['Velocidad cm/seg']=vdet
adett['Desplazamiento cm']=ddet
adett.to_excel(".\\Salidas\\Acel_Puente1_medio.xlsx", index=False)

adett_viejo=pd.DataFrame()
adett_viejo['Tiempo (s)']=data_mid_viejo['Time (s)'][12000:60000]
adett_viejo['Aceleracion cm/seg2']=adet2
adett_viejo['Velocidad cm/seg']=vdet2
adett_viejo['Desplazamiento cm']=ddet2
adett_viejo.to_excel(".\\Salidas\\Acel_Puente2_medio.xlsx", index=False)

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
adett['Velocidad cm/seg']=vdet
adett['Desplazamiento cm']=ddet
adett.to_excel(".\\Salidas\\Acel_Puente1_izquierda.xlsx", index=False)

adett=pd.DataFrame()
adett['Aceleracion cm/seg2']=adet2
adett['Velocidad cm/seg']=vdet2
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
adett['Velocidad cm/seg']=vdet
adett['Desplazamiento cm']=ddet
adett.to_excel(".\\Salidas\\Acel_Puente1_derecha.xlsx", index=False)

adett=pd.DataFrame()
adett['Aceleracion cm/seg2']=adet2
adett['Velocidad cm/seg']=vdet2
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

#=====================EXPLORATION WITH UNSUPERVISED LEARNIING METHODS
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from collections import Counter
from sklearn.metrics import silhouette_samples, silhouette_score
from yellowbrick.cluster import SilhouetteVisualizer
from mpl_toolkits.mplot3d import Axes3D
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering
import plotly.graph_objects as go
from sklearn.pipeline import make_pipeline
import textwrap
from sklearn import tree
from sklearn.decomposition import PCA
from collections import defaultdict
from sklearn.metrics.pairwise import euclidean_distances

distortions = []
inertias = []
mapping1 = {}
mapping2 = {}
K = range(1, 10)


var_in=['Aceleracion cm/seg2','Velocidad cm/seg','Desplazamiento cm']
data_mod=adett
for k in K:
    # Building and fitting the model
    kmeanModel = KMeans(n_clusters=k).fit(data_mod[var_in])
    kmeanModel.fit(data_mod[var_in])
  
    distortions.append(sum(np.min(cdist(data_mod[var_in], kmeanModel.cluster_centers_,
                                        'euclidean'), axis=1)) / data_mod[var_in].shape[0])
    inertias.append(kmeanModel.inertia_)
  
    mapping1[k] = sum(np.min(cdist(data_mod[var_in], kmeanModel.cluster_centers_,
                                   'euclidean'), axis=1)) / data_mod[var_in].shape[0]
    mapping2[k] = kmeanModel.inertia_

for key, val in mapping1.items():
    print(f'{key} : {val}')
plt.figure(figsize=(10, 8))
plt.plot(K, distortions, 'bx-')
plt.xlabel('Values of K')
plt.ylabel('Distortion')
plt.title('The Elbow Method using Distortion')
save_fig("elbow method")

plt.figure(figsize=(10, 8))
plt.plot(K, inertias, 'bx-')
plt.xlabel('Values of K')
plt.ylabel('Inertia')
plt.title('The Elbow Method using Inertia')
save_fig("elbow method_inertia")

#Kmeans
kmeans = KMeans(n_clusters=4).fit(data_mod[var_in])
data_mod["cluster"] = kmeans.labels_
data_mod['Time (s)']=data_mid['Time (s)'][12000:60000]
kmeans.cluster_centers_

colores = ["red", "blue", "orange", "black"]
plt.figure(figsize=(10, 8))
for cluster in range(kmeans.n_clusters):
    plt.scatter(data_mod[data_mod["cluster"] == cluster]["Time (s)"],
                data_mod[data_mod["cluster"] == cluster]["Velocidad cm/seg"],
                marker="o", s=20, color=colores[cluster], alpha=0.5)
    plt.scatter(kmeans.cluster_centers_[cluster][0], 
                kmeans.cluster_centers_[cluster][1], 
                marker="P", s=50, color=colores[cluster])
plt.title("Puente 1 - MEDIO", fontsize=20)
plt.xlabel("Tiempo (s)", fontsize=15)
plt.ylabel("Velocidad cm/seg", fontsize=15)
plt.text(1.15, 0.2, "K = %i" % kmeans.n_clusters, fontsize=25)
#plt.text(1.15, 0, "Inercia = %0.2f" % kmeans.inertia_, fontsize=25)
#plt.xlim(-0.1, 1.1)
#plt.ylim(-0.1, 1.1)    
save_fig("Kmeans_medio_P1_111")

plt.figure()
plt.figure(figsize = (20,15))
plt.subplot (3,1,1);plt.title ('KMEANS PUENTE NUEVO - MEDIO')
plt.scatter(data_mod[data_mod["cluster"] ==0]["Time (s)"],data_mod[data_mod["cluster"]==0]["Aceleracion cm/seg2"],s = 50, color=colores[0], label = "Cluster 1")
plt.scatter(data_mod[data_mod["cluster"] ==1]["Time (s)"],data_mod[data_mod["cluster"]==1]["Aceleracion cm/seg2"],s = 50, color=colores[1], label = "Cluster 2")
plt.scatter(data_mod[data_mod["cluster"] ==2]["Time (s)"],data_mod[data_mod["cluster"]==2]["Aceleracion cm/seg2"],s = 50, color=colores[2], label = "Cluster 3")
plt.scatter(data_mod[data_mod["cluster"] ==3]["Time (s)"],data_mod[data_mod["cluster"]==3]["Aceleracion cm/seg2"],s = 50, color=colores[3], label = "Cluster 4")
plt.scatter(kmeans.cluster_centers_[:,3],kmeans.cluster_centers_[:,17],marker="P", s = 150, color = colores, label = "centroids")
plt.xlabel("Time (s)")
plt.ylabel("Aceleracion cm/seg2")
plt.legend()

plt.subplot (3,1,2)
plt.scatter(data_mod[data_mod["cluster"] ==0]["Time (s)"],data_mod[data_mod["cluster"]==0]["Velocidad cm/seg"],s = 50, color=colores[0], label = "Cluster 1")
plt.scatter(data_mod[data_mod["cluster"] ==1]["Time (s)"],data_mod[data_mod["cluster"]==1]["Velocidad cm/seg"],s = 50, color=colores[1], label = "Cluster 2")
plt.scatter(data_mod[data_mod["cluster"] ==2]["Time (s)"],data_mod[data_mod["cluster"]==2]["Velocidad cm/seg"],s = 50, color=colores[2], label = "Cluster 3")
plt.scatter(data_mod[data_mod["cluster"] ==3]["Time (s)"],data_mod[data_mod["cluster"]==3]["Velocidad cm/seg"],s = 50, color=colores[3], label = "Cluster 4")
plt.scatter(kmeans.cluster_centers_[:,3],kmeans.cluster_centers_[:,17],marker="P", s = 150, color = colores, label = "centroids")
plt.xlabel("Time (s)")
plt.ylabel("Velocidad cm/seg")

plt.subplot (3,1,3)
plt.scatter(data_mod[data_mod["cluster"] ==0]["Time (s)"],data_mod[data_mod["cluster"]==0]["Desplazamiento cm"],s = 50, color=colores[0], label = "Cluster 1")
plt.scatter(data_mod[data_mod["cluster"] ==1]["Time (s)"],data_mod[data_mod["cluster"]==1]["Desplazamiento cm"],s = 50, color=colores[1], label = "Cluster 2")
plt.scatter(data_mod[data_mod["cluster"] ==2]["Time (s)"],data_mod[data_mod["cluster"]==2]["Desplazamiento cm"],s = 50, color=colores[2], label = "Cluster 3")
plt.scatter(data_mod[data_mod["cluster"] ==3]["Time (s)"],data_mod[data_mod["cluster"]==3]["Desplazamiento cm"],s = 50, color=colores[3], label = "Cluster 4")
plt.scatter(kmeans.cluster_centers_[:,3],kmeans.cluster_centers_[:,17],marker="P", s = 150, color = colores, label = "centroids")
plt.xlabel("Time (s)")
plt.ylabel("Desplazamiento cm")
save_fig("Kmeans_P1_medio")

plt.figure()
plt.figure(figsize = (20,10))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = np.array(data_mod['Desplazamiento cm'])
y = np.array(data_mod['Velocidad cm/seg'])
z = np.array(data_mod['Aceleracion cm/seg2'])
ax.scatter(data_mod[data_mod["cluster"] ==0]["Desplazamiento cm"],data_mod[data_mod["cluster"] ==0]["Velocidad cm/seg"],data_mod[data_mod["cluster"]==0]["Aceleracion cm/seg2"], s=20, color=colores[0], label = "Cluster 1")
ax.scatter(data_mod[data_mod["cluster"] ==1]["Desplazamiento cm"],data_mod[data_mod["cluster"] ==1]["Velocidad cm/seg"],data_mod[data_mod["cluster"]==1]["Aceleracion cm/seg2"], s=20, color=colores[1], label = "Cluster 2")
ax.scatter(data_mod[data_mod["cluster"] ==2]["Desplazamiento cm"],data_mod[data_mod["cluster"] ==2]["Velocidad cm/seg"],data_mod[data_mod["cluster"]==2]["Aceleracion cm/seg2"], s=20, color=colores[2], label = "Cluster 3")
ax.scatter(data_mod[data_mod["cluster"] ==3]["Desplazamiento cm"],data_mod[data_mod["cluster"] ==3]["Velocidad cm/seg"],data_mod[data_mod["cluster"]==3]["Aceleracion cm/seg2"], s=20, color=colores[3], label = "Cluster 4")
ax.scatter(kmeans.cluster_centers_[:,9],kmeans.cluster_centers_[:,3],kmeans.cluster_centers_[:,17],marker="P", s = 50, color = colores, label = "centroids")
plt.xlabel("Desplazamiento cm")
plt.ylabel("Velocidad cm/seg")
plt.zlabel("Aceleracion cm/seg2")
plt.legend()
plt.show()
save_fig("3D_clustering_Kmeans_P1_medio")

###==
data_mod2=adett_viejo
kmeans2 = KMeans(n_clusters=4).fit(data_mod2[var_in])
data_mod2["cluster"] = kmeans2.labels_
data_mod2['Time (s)']=data_mid_viejo['Time (s)'][12000:60000]
kmeans2.cluster_centers_
colores = ["red", "blue", "orange", "black"]
plt.figure()
plt.figure(figsize = (20,15))
plt.subplot (3,1,1);plt.title ('KMEANS PUENTE VIEJO - MEDIO')
plt.scatter(data_mod2[data_mod2["cluster"] ==0]["Time (s)"],data_mod2[data_mod2["cluster"]==0]["Aceleracion cm/seg2"],s = 50, color=colores[0], label = "Cluster 1")
plt.scatter(data_mod2[data_mod2["cluster"] ==1]["Time (s)"],data_mod2[data_mod2["cluster"]==1]["Aceleracion cm/seg2"],s = 50, color=colores[1], label = "Cluster 2")
plt.scatter(data_mod2[data_mod2["cluster"] ==2]["Time (s)"],data_mod2[data_mod2["cluster"]==2]["Aceleracion cm/seg2"],s = 50, color=colores[2], label = "Cluster 3")
plt.scatter(data_mod2[data_mod2["cluster"] ==3]["Time (s)"],data_mod2[data_mod2["cluster"]==3]["Aceleracion cm/seg2"],s = 50, color=colores[3], label = "Cluster 4")
plt.scatter(kmeans2.cluster_centers_[:,3],kmeans2.cluster_centers_[:,17],marker="P", s = 150, color = colores, label = "centroids")
plt.xlabel("Time (s)")
plt.ylabel("Aceleracion cm/seg2")
plt.legend()

plt.subplot (3,1,2)
plt.scatter(data_mod2[data_mod2["cluster"] ==0]["Time (s)"],data_mod2[data_mod2["cluster"]==0]["Velocidad cm/seg"],s = 50, color=colores[0], label = "Cluster 1")
plt.scatter(data_mod2[data_mod2["cluster"] ==1]["Time (s)"],data_mod2[data_mod2["cluster"]==1]["Velocidad cm/seg"],s = 50, color=colores[1], label = "Cluster 2")
plt.scatter(data_mod2[data_mod2["cluster"] ==2]["Time (s)"],data_mod2[data_mod2["cluster"]==2]["Velocidad cm/seg"],s = 50, color=colores[2], label = "Cluster 3")
plt.scatter(data_mod2[data_mod2["cluster"] ==3]["Time (s)"],data_mod2[data_mod2["cluster"]==3]["Velocidad cm/seg"],s = 50, color=colores[3], label = "Cluster 4")
plt.scatter(kmeans2.cluster_centers_[:,3],kmeans2.cluster_centers_[:,17],marker="P", s = 150, color = colores, label = "centroids")
plt.xlabel("Time (s)")
plt.ylabel("Velocidad cm/seg")

plt.subplot (3,1,3)
plt.scatter(data_mod2[data_mod2["cluster"] ==0]["Time (s)"],data_mod2[data_mod2["cluster"]==0]["Desplazamiento cm"],s = 50, color=colores[0], label = "Cluster 1")
plt.scatter(data_mod2[data_mod2["cluster"] ==1]["Time (s)"],data_mod2[data_mod2["cluster"]==1]["Desplazamiento cm"],s = 50, color=colores[1], label = "Cluster 2")
plt.scatter(data_mod2[data_mod2["cluster"] ==2]["Time (s)"],data_mod2[data_mod2["cluster"]==2]["Desplazamiento cm"],s = 50, color=colores[2], label = "Cluster 3")
plt.scatter(data_mod2[data_mod2["cluster"] ==3]["Time (s)"],data_mod2[data_mod2["cluster"]==3]["Desplazamiento cm"],s = 50, color=colores[3], label = "Cluster 4")
plt.scatter(kmeans2.cluster_centers_[:,3],kmeans2.cluster_centers_[:,17],marker="P", s = 150, color = colores, label = "centroids")
plt.xlabel("Time (s)")
plt.ylabel("Desplazamiento cm")
save_fig("Kmeans_P2_medio")

plt.figure()
plt.figure(figsize = (20,10))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = np.array(data_mod2['Desplazamiento cm'])
y = np.array(data_mod2['Velocidad cm/seg'])
z = np.array(data_mod2['Aceleracion cm/seg2'])
ax.scatter(data_mod2[data_mod2["cluster"] ==0]["Desplazamiento cm"],data_mod2[data_mod2["cluster"] ==0]["Velocidad cm/seg"],data_mod2[data_mod2["cluster"]==0]["Aceleracion cm/seg2"], s=20, color=colores[0], label = "Cluster 1")
ax.scatter(data_mod2[data_mod2["cluster"] ==1]["Desplazamiento cm"],data_mod2[data_mod2["cluster"] ==1]["Velocidad cm/seg"],data_mod2[data_mod2["cluster"]==1]["Aceleracion cm/seg2"], s=20, color=colores[1], label = "Cluster 2")
ax.scatter(data_mod2[data_mod2["cluster"] ==2]["Desplazamiento cm"],data_mod2[data_mod2["cluster"] ==2]["Velocidad cm/seg"],data_mod2[data_mod2["cluster"]==2]["Aceleracion cm/seg2"], s=20, color=colores[2], label = "Cluster 3")
ax.scatter(data_mod2[data_mod2["cluster"] ==3]["Desplazamiento cm"],data_mod2[data_mod2["cluster"] ==3]["Velocidad cm/seg"],data_mod2[data_mod2["cluster"]==3]["Aceleracion cm/seg2"], s=20, color=colores[3], label = "Cluster 4")
ax.scatter(kmeans2.cluster_centers_[:,9],kmeans2.cluster_centers_[:,3],kmeans2.cluster_centers_[:,17],marker="P", s = 50, color = colores, label = "centroids")
plt.xlabel("Desplazamiento cm")
plt.ylabel("Velocidad cm/seg")
plt.zlabel("Aceleracion cm/seg2")
plt.legend()
save_fig("3D_clustering_Kmeans_P2_medio")