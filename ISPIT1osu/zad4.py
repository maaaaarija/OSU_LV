'''
IRIS

Iris Dataset sastoji se od informacija o laticama i ˇcašicama tri razliˇcita cvijeta
irisa (Setosa, Versicolour i Virginica). Dostupan je u sklopu bibilioteke scikitlearn:
from sklearn import datasets
iris = datasets.load_iris()
Upoznajte se s datasetom i dodajte programski kod u skriptu pomo´cu kojeg možete odgovoriti na
sljede´ca pitanja:
a) Prikažite odnos duljine latice i ˇcašice svih pripadnika klase Virginica pomo´cu scatter
dijagrama zelenom bojom. Dodajte na isti dijagram odnos duljine latice i ˇcašice svih
pripadnika klase Setosa, sivom bojom. Dodajte naziv dijagrama i nazive osi te legendu.
Komentirajte prikazani dijagram.
b) Pomo´cu stupˇcastog dijagrama prikažite najve´cu vrijednost širine ˇcašice za sve tri klase
cvijeta. Dodajte naziv dijagrama i nazive osi. Komentirajte prikazani dijagram.
c) Koliko jedinki pripadnika klase Setosa ima ve´cu širinu ˇcašice od prosjeˇcne širine ˇcašice te
klase?

'''


from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

iris = datasets.load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target
print(df.head(5))
print(df.tail(5))

virginica = df[df['target'] == 2]
laticaV = virginica['petal length (cm)']
casicaV = virginica['sepal length (cm)']
setosa = df[df['target'] == 0]
laticaS = setosa['petal length (cm)']
casicaS = setosa['sepal length (cm)']

plt.figure()
plt.scatter(laticaV, casicaV, color = 'green', label = 'virginica')
plt.scatter(laticaS, casicaS, color = 'gray', label = 'setosa')
plt.xlabel('duljina latice')
plt.ylabel('duljina casice')
plt.title('zadatak')
plt.legend()


versicolour = df[df['target'] == 1]

max0 = setosa['sepal width (cm)'].max()
max1 = virginica['sepal width (cm)'].max()
max2 = versicolour['sepal width (cm)'].max()

lista_imena = ['setosa', 'virginica', 'versicolor']
lista_maxVr = [max0, max1, max2]

plt.figure()
plt.bar(lista_imena, lista_maxVr)
plt.xlabel('ime cvijeta')
plt.ylabel('sirina casice')
plt.title('b zadatak')
plt.show()


prosjek = setosa['sepal width (cm)'].mean()

veciOdProsjeka = setosa[setosa['sepal width (cm)'] > prosjek]

rezultat = len(veciOdProsjeka)
print(rezultat)



plt.show()
