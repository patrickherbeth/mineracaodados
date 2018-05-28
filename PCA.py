import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# load data set Iris
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# load data set into Pandas DataFrame
df = pd.read_csv(url, names=['sepal length', 'sepal width', 'petal length', 'petal width', 'target'])

# print df

# ------------------------------------- Padronize os dados --------------------------------------------


features = ['sepal length', 'sepal width', 'petal length', 'petal width']

# Separando os recursos
x = df.loc[:, features].values

# Padronizando os recursos
x = StandardScaler().fit_transform(x)

# ----------------------------------------- PCA de 4 dimensões para 2 dimwnsões ----------------------------------------

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)

principalDf = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2'])

print principalDf

# ---------------------------------------------------------------------------------

finalDf = pd.concat([principalDf, df[['target']]], axis = 1)
