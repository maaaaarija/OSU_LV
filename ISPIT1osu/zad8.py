'''
 Datoteka pima-indians-diabetes.csv sadrži mjerenja provedena u svrhu
otkrivanja dijabetesa, pri ˇcemu se u devetom stupcu nalazi izlazna veliˇcina, predstavljena klasom
0 (nema dijabetes) ili klasom 1 (ima dijabetes).
Uˇcitajte dane podatke u obliku numpy polja data. Podijelite ih na ulazne podatke X i izlazne
podatke y. Podijelite podatke na skup za uˇcenje i skup za testiranje modela u omjeru 80:20.
Dodajte programski kod u skriptu pomo´cu kojeg možete odgovoriti na sljede´ca pitanja:
a) Izgradite model logistiˇcke regresije pomo´cu scikit-learn biblioteke na temelju skupa poda
taka za uˇcenje.
b) Provedite klasifikaciju skupa podataka za testiranje pomo´cu izgra¯denog modela logistiˇcke
regresije.
c) Izraˇcunajte i prikažite matricu zabune na testnim podacima. Komentirajte dobivene rezul
tate.
d) Izraˇcunajte toˇcnost, preciznost i odziv na skupu podataka za testiranje. Komentirajte
dobivene rezultate.
'''

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, recall_score, precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

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

model = LogisticRegression()
model.fit(X_train_s, y_train)
y_pred = model.predict(X_test_s)

cm = confusion_matrix(y_test, y_pred)
print(cm)
plt.figure()
disp = ConfusionMatrixDisplay(cm)
disp.plot()
plt.show()

accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
print(accuracy, recall, precision)
