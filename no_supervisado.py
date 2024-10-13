import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

#tomando lineas del ejemplo anterior.
num_muestras = 100
rutas = ['R' + str(i) for i in np.random.randint(1, 21, num_muestras)]  
num_pasajeros = np.random.randint(10, 100, num_muestras)  

data = pd.DataFrame({
    'Ruta': rutas,
    'Num_Pasajeros': num_pasajeros
})

data_dummies = pd.get_dummies(data, columns=['Ruta'])

#k-means aprendizaje no supervisado.
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(data_dummies)

# Agregar la etiqueta del cluster 
data['Cluster'] = kmeans.labels_

# representaicon grafica delos resultados
plt.figure(figsize=(8,5)) # coordenadas para dibujar el plano, x=8, y=5
sns.scatterplot(x='Num_Pasajeros', y='Ruta', data=data, hue='Cluster', palette='Set1', s=100)
plt.title('Agrupamiento de Viajes (K-means)')
plt.show()
