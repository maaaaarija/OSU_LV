'''
Datoteka titanic.csv sadrži podatke o putnicima broda Titanic, koji je potonuo
1912. godine. Upoznajte se s datasetom. Uˇcitajte dane podatke. Podijelite ih na ulazne podatke X
predstavljene stupcima Pclass, Sex, Fare i Embarked i izlazne podatke y predstavljene stupcem
Survived. Podijelite podatke na skup za uˇcenje i skup za testiranje modela u omjeru 70:30.
Izbacite izostale i null vrijednosti. Skalirajte podatke. Dodajte programski kod u skriptu pomo´cu
kojeg možete odgovoriti na sljede´ca pitanja:
a) Izradite algoritam KNN na skupu podataka za uˇcenje (uz K=5). Vizualizirajte podatkovne
primjere i granicu odluke.
b) Izraˇcunajte toˇcnost klasifikacije na skupu podataka za uˇcenje i skupu podataka za testiranje.
Komentirajte dobivene rezultate.
c) Pomo´cu unakrsne validacije odredite optimalnu vrijednost hiperparametra K algoritma
KNN.
d) Izraˇcunajte toˇcnost klasifikacije na skupu podataka za uˇcenje i skupu podataka za testiranje
za dobiveni K. Usporedite dobivene rezultate s rezultatima kada je K=5.
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score 


data = pd.read_csv('titanic.csv')
data = data.dropna(axis = 0)
X = data[['Pclass', 'Sex', 'Fare', 'Embarked']]
y = data['Survived']
X = pd.get_dummies(X)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)

scaler = MinMaxScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

model = KNeighborsClassifier(n_neighbors = 5)
model.fit(X_train_s, y_train)
y_pred = model.predict(X_test_s)




y_pred_train = model.predict(X_train_s)
print('tocnost klasifikacije ucenje ', accuracy_score(y_train, y_pred_train))
print('tocnost klasifikacije test', accuracy_score(y_test, y_pred))

prosjeci = []

for k in range(1, 21):
    model_cv = KNeighborsClassifier(n_neighbors = k)
    scores = cross_val_score(model_cv, X_train_s, y_train, cv = 10)
    prosjeci.append(scores.mean())
    
optimalniK = np.argmax(prosjeci) + 1
print('k je ', optimalniK)

model2 = KNeighborsClassifier(n_neighbors = optimalniK)
model2.fit(X_train_s, y_train)
y_test_pred2 = model2.predict(X_test_s)
y_train_pred2 = model2.predict(X_train_s)

print('tocnost ucenje ', accuracy_score(y_train, y_train_pred2))
print('tocnost klasifikacija ', accuracy_score(y_test, y_test_pred2))


