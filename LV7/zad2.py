'''
Kvantizacija boje je proces smanjivanja broja razlicitih boja u digitalnoj slici, ali
uzimajuci u obzir da rezultantna slika vizualno bude što slicnija originalnoj slici. Jednostavan
nacin kvantizacije boje može se postici primjenom algoritma K srednjih vrijednosti na RGB
vrijednosti elemenata originalne slike. Kvantizacija se tada postiže zamjenom vrijednosti svakog
elementa originalne slike s njemu najbližim centrom. Na slici 7.3a dan je primjer originalne
slike koja sadrži ukupno 106,276 boja, dok je na slici 7.3b prikazana rezultantna slika nakon
kvantizacije i koja sadrži samo 5 boja koje su odredene algoritmom K srednjih vrijednosti.
1. Otvorite skriptu zadatak_2.py. Ova skripta ucitava originalnu RGB sliku test_1.jpg
te ju transformira u podatkovni skup koji dimenzijama odgovara izrazu (7.2) pri cemu je n
broj elemenata slike, a m je jednak 3. Koliko je razlicitih boja prisutno u ovoj slici?
2. Primijenite algoritam K srednjih vrijednosti koji ce pronaci grupe u RGB vrijednostima
elemenata originalne slike.
3. Vrijednost svakog elementa slike originalne slike zamijeni s njemu pripadajucim centrom.
4. Usporedite dobivenu sliku s originalnom. Mijenjate broj grupa K. Komentirajte dobivene
rezultate.
5. Primijenite postupak i na ostale dostupne slike.
6. Graficki prikažite ovisnost J o broju grupa K. Koristite atribut inertia objekta klase
KMeans. Možete li uociti lakat koji upucuje na optimalni broj grupa?
7. Elemente slike koji pripadaju jednoj grupi prikažite kao zasebnu binarnu sliku. Što
primjecujete?
'''

# druga verzija (ima dio i u boji)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as Image
from sklearn.cluster import KMeans

# 1. Učitavanje slike
img = Image.imread("imgs/imgs/test_1.jpg")

# Prikaz originalne slike
plt.figure()
plt.title("Originalna slika")
plt.imshow(img)
plt.axis('off')
plt.show()

# Pretvaranje u float i skaliranje [0,1]
img = img.astype(np.float64) / 255

# 2. Transformacija u (n, 3)
w, h, d = img.shape
img_array = np.reshape(img, (w * h, d))

# Broj različitih boja (ISPRAVNO)
colors_num = np.unique(img_array, axis=0)
print('Broj boja u originalnoj slici:', len(colors_num))

# 3. K-means
K = 15
km = KMeans(n_clusters=K, init='random', n_init=5, random_state=0)
km.fit(img_array)

# Zamjena vrijednosti centrima
labels = km.labels_
centers = km.cluster_centers_
img_array_aprox = centers[labels]

# Povratak u oblik slike
img_finished = np.reshape(img_array_aprox, (w, h, d))

# Prikaz kvantizirane slike
plt.figure()
plt.title(f"Kvantizirana slika (K={K})")
plt.imshow(img_finished)
plt.axis('off')
plt.show()

# 4. Lakat metoda
J_values = []
K_range = range(1, 10)

for i in K_range:
    km_test = KMeans(n_clusters=i, n_init=5, random_state=0)
    km_test.fit(img_array)
    J_values.append(km_test.inertia_)

# Graf lakat metode
plt.figure()
plt.plot(K_range, J_values, marker='o')
plt.xlabel('Broj grupa K')
plt.ylabel('J (inertia)')
plt.title('Lakat metoda')
plt.show()

# 5. Prikaz binarnih slika za svaku grupu
for i in range(K):
    img_bin = (labels == i).reshape((w, h))
    plt.figure()
    plt.imshow(img_bin, cmap='gray')
    plt.title(f'Grupa {i+1}')
    plt.axis('off')
    plt.show()