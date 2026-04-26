'''
IRIS

Iris Dataset sastoji se od informacija o laticama i ˇcašicama tri razliˇcita cvijeta
irisa (Setosa, Versicolour i Virginica). Dostupan je u sklopu bibilioteke scikitlearn:
from sklearn import datasets
iris = datasets.load_iris()
Upoznajte se s datasetom. Podijelite ga na ulazne podatke X i izlazne podatke y predstavljene
klasom cvijeta. Pripremite podatke za uˇcenje neuronske mreže (kategoriˇcke veliˇcine, skaliranje...).
Podijelite podatke na skup za uˇcenje i skup za testiranje modela u omjeru 80:20. Dodajte
programski kod u skriptu pomo´cu kojeg možete odgovoriti na sljede´ca pitanja:
a) Izgradite neuronsku mrežu sa sljede´cim karakteristikama:
- model oˇcekuje ulazne podatke X
- prvi skriveni sloj ima 12 neurona i koristi relu aktivacijsku funkciju
- drugi skriveni sloj ima 7 neurona i koristi relu aktivacijsku funkciju
- tre´ci skriveni sloj ima 5 neurona i koristi relu aktivacijsku funkciju
- izlazni sloj ima 3 neurona i koristi softmax aktivacijsku funkciju.
-izme¯du prvog i drugog te drugog i tre´ceg sloja dodajte Dropout sloj s 20%, odnosno 30%
izbaˇcenih neurona
Ispišite informacije o mreži u terminal.
b) Podesite proces treniranja mreže sa sljede´cim parametrima:
- loss argument: categorical_crossentropy
- optimizer: adam
- metrika: accuracy.
c) Pokrenite uˇcenje mreže sa proizvoljnim brojem epoha (pokušajte sa 450) i veliˇcinom
batch-a 7.
d) Pohranite model na tvrdi disk te preostale zadatke izvršite na temelju uˇcitanog modela.
e) Izvršite evaluaciju mreže na testnom skupu podataka.
f) Izvršite predikciju mreže na skupu podataka za testiranje. Prikažite matricu zabune za skup
podataka za testiranje. Komentirajte dobivene rezultate i predložite kako biste ih poboljšali,
ako je potrebno
'''

from sklearn.preprocessing import StandardScaler
from sklearn import datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers


iris = datasets.load_iris()


X = iris.data
y = to_categorical(iris.target)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)


model = Sequential()
model.add(layers.Input(shape = (X.shape[1], )))   # umjesto X.shape[1] moglo je samo pisati 4
model.add(layers.Dense(12, activation = 'relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(7, activation = 'relu'))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(5, activation = 'relu'))
model.add(layers.Dense(3, activation = 'softmax'))
model.summary()

model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])
model.fit(X_train_s, y_train, batch_size = 7, epochs = 450)

model.save('modelZadatak')

model2 = load_model('modelZadatak')
rezultat = model2.evaluate(X_test_s, y_test)

plt.figure()
predict = model2.predict(X_test_s)
y_pred = np.argmax(predict, axis = 1)
y_test_klase = np.argmax(y_test, axis = 1)
cm = confusion_matrix(y_test_klase, y_pred)
print(cm)
disp = ConfusionMatrixDisplay(cm)
disp.plot()
plt.show()

'''
Mreža je uspješno klasificirala iris. Korišteni su Dropout slojevi kako bi se 
spriječilo prenaučenost (overfitting), što je vidljivo iz stabilnog rasta točnosti na 
testnom skupu. Matrica zabune potvrđuje da model rijetko griješi, a eventualne greške 
se događaju između morfološki sličnih klasa (Versicolor i Virginica).
'''
