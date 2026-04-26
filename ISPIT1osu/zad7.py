'''
Datoteka pima-indians-diabetes.csv sadrži mjerenja provedena u svrhu
otkrivanja dijabetesa, pri ˇcemu se u devetom stupcu nalazi klasa 0 (nema dijabetes) ili klasa 1
(ima dijabetes). Uˇcitajte dane podatke u obliku numpy polja data. Dodajte programski kod u
skriptu pomo´cu kojeg možete odgovoriti na sljede´ca pitanja:
a) Na temelju veliˇcine numpy polja data, na koliko osoba su izvršena mjerenja?
b) Postoje li izostale ili duplicirane vrijednosti u stupcima s mjerenjima dobi i indeksa tjelesne
mase (BMI)? Obrišite ih ako postoje. Koliko je sada uzoraka mjerenja preostalo?
c) Prikažite odnos dobi i indeksa tjelesne mase (BMI) osobe pomo´cu scatter dijagrama.
Dodajte naziv dijagrama i nazive osi s pripadaju´cim mjernim jedinicama. Komentirajte
odnos dobi i BMI prikazan dijagramom.
d) Izraˇcunajte i ispišite u terminal minimalnu, maksimalnu i srednju vrijednost indeksa tjelesne
mase (BMI) u ovom podatkovnom skupu.
e) Ponovite zadatak pod d), ali posebno za osobe kojima je dijagnosticiran dijabetes i za one
kojima nije. Kolikom je broju ljudi dijagonosticiran dijabetes? Komentirajte dobivene
vrijednosti.
'''

from sklearn import datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = np.genfromtxt('pima-indians-diabetes.csv', delimiter=',')
brOsoba = len(data)

columns = [
    'Number of times pregnant', 'Plasma glucose concentration a 2 hours in an oral glucose tolerance test', 'Diastolic blood pressure (mm Hg)', 'Triceps skin fold thickness (mm)',
    '2-Hour serum insulin (mu U/ml)', 'Body mass index (weight in kg/(height in m)^2)', 'DiabetesPedigreeFunction', 'Age (years)', 'Class variable (0 or 1)'
]

data = pd.DataFrame(data, columns=columns)


data['Body mass index (weight in kg/(height in m)^2)'].isnull().sum()
data['Age (years)'].isnull().sum()
data['Age (years)'].unique()
data['Body mass index (weight in kg/(height in m)^2)'].unique()
data.dropna(subset = ['Age (years)', 'Body mass index (weight in kg/(height in m)^2)'], inplace = True)
data.drop_duplicates(subset = ['Body mass index (weight in kg/(height in m)^2)', 'Age (years)'], inplace = True)
data.reset_index(drop = True, inplace = True)

noviBrOsoba = len(data)

plt.figure()
plt.scatter(data['Age (years)'], data['Body mass index (weight in kg/(height in m)^2)'])
plt.xlabel('dob u godinama')
plt.ylabel('bmi u kilogramima po metrima kvadratnim')
plt.title('graf za zadatak')


min = data['Body mass index (weight in kg/(height in m)^2)'].min()
max = data['Body mass index (weight in kg/(height in m)^2)'].max()
mean = data['Body mass index (weight in kg/(height in m)^2)'].mean()
print(min, max, mean)

dijabetes = data[data['Class variable (0 or 1)'] == 1]
neDijabetes = data[data['Class variable (0 or 1)'] == 0]

brDijabetes = len(dijabetes)
print(brDijabetes)

minD = dijabetes['Body mass index (weight in kg/(height in m)^2)'].min()
maxD = dijabetes['Body mass index (weight in kg/(height in m)^2)'].max()
meanD = dijabetes['Body mass index (weight in kg/(height in m)^2)'].mean()
print(minD, maxD, meanD)

minN = neDijabetes['Body mass index (weight in kg/(height in m)^2)'].min()
maxN = neDijabetes['Body mass index (weight in kg/(height in m)^2)'].max()
meanN = neDijabetes['Body mass index (weight in kg/(height in m)^2)'].mean()
print(minN, maxN, meanN)


