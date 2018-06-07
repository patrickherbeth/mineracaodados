import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_mldata

# load data set Iris
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# load data set into Pandas DataFrame
df = pd.read_csv(url, names=['sepal length', 'sepal width', 'petal length', 'petal width', 'target'])



# ------------------------------------- Padronize os dados --------------------------------------------


features = ['sepal length', 'sepal width', 'petal length', 'petal width']

# Separando os recursos
x = df.loc[:, features].values

print x

# Padronizando os recursos
x = StandardScaler().fit_transform(x)

# ----------------------------------------- PCA de 4 dimensoes para 2 dimensoes ----------------------------------------

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)

principalDf = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2'])

# print principalDf

# -------------Concatenando DataFrame ao longo do eixo 1 finalDf e o DataFrame final antes de plotar os dados ----------

finalDf = pd.concat([principalDf, df[['target']]], axis=1)

# ------------------------ Utilizando o Kmeans --------------------------


X = np.array(finalDf.drop('target', axis=1))

kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(X)

# print kmeans.labels_
finalDf['K-classes'] = kmeans.labels_
sb.pairplot(finalDf, hue='target')

# print finalDf

# -------------------------------- PLOT ----------------------------------
fig = plt.figure(figsize=(8, 8))

ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('Principal Component 1', fontsize=15)
ax.set_ylabel('Principal Component 2', fontsize=15)
ax.set_title('2 component PCA', fontsize=20)

targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
colors = ['r', 'g', 'b']

for target, color in zip(targets, colors):
    indicesToKeep = finalDf['target'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c=color
               , s=50)
ax.legend(targets)
ax.grid()






# outros graficos


pca.explained_variance_ratio_
mnist = fetch_mldata('MNIST original')

from sklearn.model_selection import train_test_split

# test_size: what proportion of original data is used for test set
train_img, test_img, train_lbl, test_lbl = train_test_split(mnist.data, mnist.target, test_size=1/7.0, random_state=0)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# Fit on training set only.
scaler.fit(train_img)

# Apply transform to both the training set and the test set.
train_img = scaler.transform(train_img)
test_img = scaler.transform(test_img)


plt.show()
