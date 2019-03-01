import time
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score


start = time. time()

dataPath = '/Users/zahra_abasiyan/Desktop/Data_Cortex_.xls'

data = pd.read_excel(open(dataPath, 'rb'),sheet_name='Hoja1')

classes = set(data.loc[:,'class'])

classes = list(classes)

actual_lablel = []

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

cls = KMeans(8)

cls.fit(data)

print(normalized_mutual_info_score(actual_lablel, cls.labels_))

end = time. time()

print(end-start)

print(cls.cluster_centers_)