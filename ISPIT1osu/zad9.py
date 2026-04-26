'''
Datoteka pima-indians-diabetes.csv sadrži mjerenja provedena u svrhu
otkrivanja dijabetesa, pri ˇ cemu je prvih 8 stupaca ulazna veliˇcina, a u devetom stupcu se nalazi
izlazna veliˇcina: klasa 0 (nema dijabetes) ili klasa 1 (ima dijabetes).
Uˇcitajte dane podatke. Podijelite ih na ulazne podatke X i izlazne podatke y. Podijelite podatke
na skup za uˇcenje i skup za testiranje modela u omjeru 80:20.
a) Izgradite neuronsku mrežu sa sljede´cim karakteristikama:
- model oˇcekuje ulazne podatke s 8 varijabli
- prvi skriveni sloj ima 12 neurona i koristi relu aktivacijsku funkciju
- drugi skriveni sloj ima 8 neurona i koristi relu aktivacijsku funkciju
- izlasni sloj ima jedan neuron i koristi sigmoid aktivacijsku funkciju.
Ispišite informacije o mreži u terminal.
b) Podesite proces treniranja mreže sa sljede´cim parametrima:
- loss argument: cross entropy
- optimizer: adam
- metrika: accuracy.
c) Pokrenite uˇcenje mreže sa proizvoljnim brojem epoha (pokušajte sa 150) i veliˇcinom
batch-a 10.
d) Pohranite model na tvrdi disk te preostale zadatke izvršite na temelju uˇcitanog modela.
e) Izvršite evaluaciju mreže na testnom skupu podataka.
f) Izvršite predikciju mreže na skupu podataka za testiranje. Prikažite matricu zabune za skup
podataka za testiranje. Komentirajte dobivene rezultate.
'''


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


data = np.genfromtxt('pima-indians-diabetes.csv', delimiter=',')
columns = [
    'Number of times pregnant', 'Plasma glucose concentration a 2 hours in an oral glucose tolerance test', 'Diastolic blood pressure (mm Hg)', 'Triceps skin fold thickness (mm)',
    '2-Hour serum insulin (mu U/ml)', 'Body mass index (weight in kg/(height in m)^2)', 'DiabetesPedigreeFunction', 'Age (years)', 'Class variable (0 or 1)'
]

data = pd.DataFrame(data, columns=columns)
X = data[['Number of times pregnant', 'Plasma glucose concentration a 2 hours in an oral glucose tolerance test', 'Diastolic blood pressure (mm Hg)', 'Triceps skin fold thickness (mm)',
    '2-Hour serum insulin (mu U/ml)', 'Body mass index (weight in kg/(height in m)^2)', 'DiabetesPedigreeFunction', 'Age (years)']]
y = data['Class variable (0 or 1)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

model = Sequential()
model.add(layers.Input(shape=(8,)))
model.add(layers.Dense(12, activation = 'relu'))
model.add(layers.Dense(8, activation = 'relu'))
model.add(layers.Dense(1, activation = 'sigmoid'))
model.summary()

model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ["accuracy"])
model.fit(X_train_s, y_train, epochs = 150, batch_size = 10)

model.save('modelZadatak9')

model2 = load_model('modelZadatak9')
rezultat = model2.evaluate(X_test_s, y_test)

predict = model2.predict(X_test_s)
y_pred = (predict > 0.5).astype("int32")

cm = confusion_matrix(y_test, y_pred)
print(cm)
plt.figure()
disp = ConfusionMatrixDisplay(cm)
disp.plot()
plt.show()

'''
Model neuronske mreže pokazuje stabilne rezultate na testnom skupu. Matrica zabune nam 
omogućuje uvid u to koliko smo osoba s dijabetesom (klasa 1) ispravno detektirali 
(True Positives). Budući da je riječ o medicinskoj dijagnozi, cilj nam je minimizirati 
lažno negativne rezultate (donji lijevi kut matrice)
'''

