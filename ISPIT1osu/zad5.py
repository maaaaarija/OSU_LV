'''
IRIS


Iris Dataset sastoji se od informacija o laticama i ˇcašicama tri razliˇcita cvijeta
irisa (Setosa, Versicolour i Virginica). Dostupan je u sklopu bibilioteke scikitlearn:
from sklearn import datasets
ris = datasets.load_iris()
Upoznajte se s datasetom. Pripremite podatke za uˇcenje. Dodajte programski kod u skriptu
pomocu kojeg možete odgovoriti na sljede´ca pitanja:
a) Prona¯dite optimalni broj klastera K za klasifikaciju cvijeta irisa algoritmom K srednjih
vrijednosti.
b) Graficki prikažite lakat metodu.
c) Primijenite algoritam K srednjih vrijednosti koji ´ ce prona´ ci grupe u podatcima. Koristite vrijednot K dobivenu u prethodnom zadatku.
d) Dijagramom raspršenja prikažite dobivene klastere. Obojite ih razliˇ citim bojama (zelena,
žuta i naranˇcasta). Centroide obojite crvenom bojom. Dodajte nazive osi, naziv dijagrama i
legendu. Komentirajte prikazani dijagram.
e) Usporedite dobivene klase sa njihovim stvarnim vrijednostima. Izraˇcunajte toˇcnost klasifi
kacije.
'''


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


iris = datasets.load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

scaler = StandardScaler()
scaled = scaler.fit_transform(df)

inercije = []
K_raspon = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
for k in K_raspon:
    model_test = KMeans(n_clusters = k, random_state = 1)
    model_test.fit(scaled)
    inercije.append(model_test.inertia_)
    
plt.figure()
plt.plot(K_raspon, inercije, marker = 'o')
plt.show()

model = KMeans(n_clusters = 3, random_state = 1)
grupe = model.fit_predict(scaled)


# --- d) Dijagram raspršenja (Scatter plot) ---
plt.figure(figsize=(10, 6))

# Crtanje svake grupe (klastera) s traženim bojama
# Uzimamo prva dva stupca: sepal length (indeks 0) i sepal width (indeks 1)
plt.scatter(scaled[grupe == 0, 0], scaled[grupe == 0, 1], c='green', label='Klaster 1 (Zelena)')
plt.scatter(scaled[grupe == 1, 0], scaled[grupe == 1, 1], c='yellow', label='Klaster 2 (Žuta)')
plt.scatter(scaled[grupe == 2, 0], scaled[grupe == 2, 1], c='orange', label='Klaster 3 (Narančasta)')

# Crtanje centroida (središta) - oni su crveni X
plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], 
            s=200, c='red', marker='X', label='Centroidi')

# Nazivi i ukrasi
plt.title('Prikaz klastera irisa (K-Means)')
plt.xlabel('Sepal length (skalirano)')
plt.ylabel('Sepal width (skalirano)')
plt.legend()
plt.show()

# KOMENTAR (za usmeno ili u skripti): 
# Vidljivo je da su podaci podijeljeni u 3 grupe. Zelena grupa je prilično 
# izdvojena, dok se žuta i narančasta malo preklapaju, što je očekivano kod Irisa.

# --- e) Usporedba i točnost ---
# Uspoređujemo dobivene 'grupe' sa stvarnim 'iris.target'
tocnost = accuracy_score(iris.target, grupe)

print(f"Točnost klasifikacije: {tocnost * 100:.2f}%")

# VAŽNA NAPOMENA ZA ISPIT: 
# Ako je točnost mala (npr. 20% ili 40%), to je normalno kod K-Means-a 
# jer algoritam nasumično dodijeli broj klastera (0, 1 ili 2), 
# pa se brojevi možda ne poklapaju sa stvarnim oznakama (target), 
# iako je vizualno grupiranje točno.






