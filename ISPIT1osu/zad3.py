'''
Datoteka titanic.csv sadrži podatke o putnicima broda Titanic, koji je potonuo
1912. godine. Upoznajte se s datasetom. Uˇcitajte dane podatke. Podijelite ih na ulazne podatke X
predstavljene stupcima Pclass, Sex, Fare i Embarked i izlazne podatke y predstavljene stupcem
Survived. Podijelite podatke na skup za uˇcenje i skup za testiranje modela u omjeru 75:25.
Izbacite izostale i null vrijednosti. Skalirajte podatke. Dodajte programski kod u skriptu pomo´cu
kojeg možete odgovoriti na sljede´ca pitanja:
a) Izgradite neuronsku mrežu sa sljede´cim karakteristikama:
- model oˇcekuje ulazne podatke X
- prvi skriveni sloj ima 12 neurona i koristi relu aktivacijsku funkciju
- drugi skriveni sloj ima 8 neurona i koristi relu aktivacijsku funkciju
- tre´ci skriveni sloj ima 4 neurona i koristi relu aktivacijsku funkciju
- izlazni sloj ima jedan neuron i koristi sigmoid aktivacijsku funkciju.
Ispišite informacije o mreži u terminal.
b) Podesite proces treniranja mreže sa sljede´cim parametrima:
- loss argument: binary_crossentropy
- optimizer: adam
- metrika: accuracy.
c) Pokrenite uˇcenje mreže sa proizvoljnim brojem epoha (pokušajte sa 100) i veliˇcinom
batch-a 5.
d) Pohranite model na tvrdi disk te preostale zadatke izvršite na temelju uˇcitanog modela.
e) Izvršite evaluaciju mreže na testnom skupu podataka.
f) Izvršite predikciju mreže na skupu podataka za testiranje. Prikažite matricu zabune za skup
podataka za testiranje. Komentirajte dobivene rezultate i predložite kako biste ih poboljšali,
ako je potrebno
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

data = pd.read_csv('titanic.csv')
data = data.dropna(axis = 0)
X = data[['Pclass', 'Sex', 'Fare', 'Embarked']]
y = data['Survived']
X = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 1)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

model = keras.Sequential()
model.add(layers.Input((X_train_s.shape[1], )))
model.add(layers.Dense(12, activation ='relu'))
model.add(layers.Dense(8, activation ='relu'))
model.add(layers.Dense(4, activation ='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ["accuracy"])
model.fit(X_train_s, y_train, batch_size = 5, epochs = 100)
model.summary()
model.save('model.keras')

model2 = load_model('model.keras')

results = model2.evaluate(X_test_s, y_test)



predictions = model2.predict(X_test_s)
y_pred = (predictions > 0.5).astype("int32")
cm = confusion_matrix(y_test, y_pred)
print(cm)
disp = ConfusionMatrixDisplay(cm)
disp.plot()
plt.show()


