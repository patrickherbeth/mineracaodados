import pandas

from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from matplotlib import pyplot

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# carregando iris.csv sem a coluna que contem
# os rotulos das classes
dados = pandas.read_csv(url, names=['sepal length', 'sepal width', 'petal length', 'petal width', 'target'])
dados = dados.drop('target', axis=1)


# O metodo 'pdist' recebe como entrada um numpy.array.
# O atributo 'values' de um pandas.DataFrame retorna
# seus valores em tal formato.

# Medidas de (dis)similaridade
distancias = pdist(dados, metric='euclidean')

# o metodo 'squareform'.
distancias = squareform(distancias)


h = linkage(dados, method='complete', metric='euclidean')

dendrogram(h)

pyplot.show()






