import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

missing_values_deleted = pd.read_csv("step2/missing_values_deleted.csv", sep=",", low_memory=False)

hrmn=pd.cut(missing_values_deleted['hrmn'],24,labels=[str(i) for i in range(0,24)])

missing_values_deleted['hrmn']=hrmn.values

# On extrait du tableau la latitude et la longitude

X_lat = missing_values_deleted['lat']
X_long = missing_values_deleted['long']

# On définit tous nos points à classifier

X_cluster = np.array((list(zip(X_lat, X_long))))

# Kmeans nous donne pour chaque point la catégorie associée

clustering = KMeans(n_clusters=15, random_state=0)
clustering.fit(X_cluster)

# Enfin on ajoute les catégories dans la base d'entraînement

geo = pd.Series(clustering.labels_)
missing_values_deleted['geo'] = geo

missing_values_deleted.to_csv("step3/time_encoding.csv", index=False)