import matplotlib.pyplot as plt
import nltk as nltk
import re
import csv
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


# nltk.download ('stopwords')

reader = csv.reader(open('tweets.csv', 'rb'))

X = []
Y = []



# --------------------- Metodo de Pre-processamento --------------------

def RemoviStopWords(instancia):
    instancia = instancia.lower ()
    stopwords = set(nltk.corpus.stopwords.words('english'))
    palavras = [i for i in instancia.split() if not i in stopwords]
    return " ".join(palavras)


def Character(instancia):
  # texto = re.sub('[-|0-9]', ' ', instancia)
    texto = re.sub('[-|%-./?!,":;_*&@!#+^~|"()\'=]', ' ', instancia)
    return texto


# --------------------------------------------------------------------

# text = RemoviStopWords(Character(texto))

# print df


for linha in reader:
        dado = [linha[0], linha[8]]
        X.append(dado)

for x in X:
    print x


# Pre processamento

# ------------------------------------- Padronize os dados --------------------------------------------


df = pd.read_csv('tweets.csv', names=['Id',
                                       'Created At',
                                       'Geo Coordinates.latitude',
                                       'Geo Coordinates'
                                       'longitude',
                                       'User Location',
                                       'Username',
                                       'User Screen Name',
                                       'Retweet Count',
                                       'Classificacao'])






#print tabela

# Separando os recursos
#x = df.loc[:, X].values

# Padronizando os recursos
# x = StandardScaler().fit_transform(x)


# ----------------------------------------- PCA de 4 dimensoes para 2 dimensoes ----------------------------------------

#pca = PCA(n_components=2)
#principalComponents = pca.fit_transform(x)

#principalDf = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2'])

# print principalDf

# -------------Concatenando DataFrame ao longo do eixo 1 finalDf e o DataFrame final antes de plotar os dados ----------

#finalDf = pd.concat([principalDf, df[['target']]], axis=1)


# -------------Concatenando DataFrame ao longo do eixo 1 finalDf e o DataFrame final antes de plotar os dados ----------


# ------------------------ Utilizando o Kmeans --------------------------

# kmeans = KMeans(n_clusters=3, random_state=0)
# kmeans.fit(X)

# print kmeans.labels_
#df['text'] = kmeans.labels_
# sb.pairplot(df, hue='target')




#kmeans = KMeans(n_clusters=3, random_state=0)
#kmeans.fit(X)

# print finalDf

# ----------------------- PLOT ------------------------
# fig = plt.figure(figsize=(8, 8))
# plt.show()