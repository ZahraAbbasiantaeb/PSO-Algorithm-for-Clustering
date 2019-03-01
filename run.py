import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from question4.PSO_algorithm import PSO_algorithm
from sklearn.metrics.cluster import normalized_mutual_info_score

dataPath = '/Users/zahra_abasiyan/Desktop/Data_Cortex_.xls'

data = pd.read_excel(open(dataPath, 'rb'), sheet_name='Hoja1')

classes = set(data.loc[:,'class'])

classes = list(classes)

actual_lablel = []

cluster_count = 8

for i in range (0, 1080):

    index = 0
    for j in range (0, len(classes)):

        if(data.loc[i,'class']==classes[j]):
            index = j

    actual_lablel.append(index)

data2 = data.as_matrix();

data = data2[:,1:78]

for i in range(0, np.shape(data)[1]):

    mu = np.nanmean(data[:,i], 0, keepdims=1)
    for j in range (0, np.shape(data)[0]):

        if np.isnan(data[j,i]):
            data[j,i]=mu


cls = KMeans(cluster_count)

cls.fit(data)


max = np.zeros(np.shape(data)[1])
min = np.zeros(np.shape(data)[1])

cols = (np.shape(data)[1])

# find min and max of each column

for i in range (0, cols):
    max[i] = np.nanmax(data[:,i])
    min[i] = np.nanmin(data[:, i])

cls = KMeans(cluster_count)

cls.fit(data)

pso = PSO_algorithm(data, cluster_count, 77, 10 ,15, min, max, actual_lablel)

pso.initialPopulation(cls.cluster_centers_)

pred_lables= pso.run()


print("MI of Kmeans is:")

print(normalized_mutual_info_score(actual_lablel, cls.labels_))

print("MI of PSO is:")

print(normalized_mutual_info_score(actual_lablel, pred_lables))
