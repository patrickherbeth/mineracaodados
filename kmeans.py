import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
import seaborn as sb

from sklearn.cluster import KMeans

df = pd.read_csv('iris.csv')

X = np.array(df.drop('target', axis=1))

kmeans = KMeans(n_clusters=3, random_state=0)

kmeans.fit(X)

# print X

df['K-classes'] = kmeans.labels_

# print X print df
# sb.pairplot(df, hue='target')
sb.pairplot(df, 'K-classes')

pl.show()



