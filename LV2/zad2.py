'''
Datoteka data.csv sadrži mjerenja visine i mase provedena na muškarcima i ženama. 
Skripta zadatak_2.py ucitava dane podatke u obliku numpy polja data pri cemu je u 
prvom stupcu polja oznaka spola (1 muško, 0 žensko), drugi stupac polja je visina u cm, 
a treci stupac polja je masa u kg.
a) Na temelju velicine numpy polja data, na koliko osoba su izvršena mjerenja? 
b) Prikažite odnos visine i mase osobe pomocu naredbe ´ matplotlib.pyplot.scatter.
c) Ponovite prethodni zadatak, ali prikažite mjerenja za svaku pedesetu osobu na slici.
d) Izracunajte i ispišite u terminal minimalnu, maksimalnu i srednju vrijednost visine u ovom podatkovnom skupu.
e) Ponovite zadatak pod d), ali samo za muškarce, odnosno žene. Npr. kako biste izdvojili
muškarce, stvorite polje koje zadrži bool vrijednosti i njega koristite kao indeks retka ind = (data[:,0] == 1)
'''

import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('data.csv', delimiter = ',', skiprows = 1)
rows, cols = np.shape(data)

print('Number of people measured: ', rows)

height = data[:,1]
weight = data[:,2]
plt.scatter(height, weight, s = 1)
plt.xlabel('height')
plt.ylabel('weight')
plt.show()

height50 = height[: : 50]
weight50 = weight[: : 50]
plt.scatter(height50, weight50, s = 5)
plt.xlabel('height')
plt.ylabel('weight')
plt.title('Every 50th person')
plt.show()

print('Minimal height', np.min(height))
print('Maximal height', np.max(height))
print('Average height', np.mean(height))

men = data[np.where(data[:, 0] == 1)]
women = data[np.where(data[:, 0] == 0)]
print('Minimal men height', np.min(men[:, 1]))
print('Maximal men height', np.max(men[:, 1]))
print('Average men height', np.mean(men[:, 1]))
print('Minimal women height', np.min(women[:, 1]))
print('Maximal women height', np.max(women[:, 1]))
print('Average women height', np.mean(women[:, 1]))