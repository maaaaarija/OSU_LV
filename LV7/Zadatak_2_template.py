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

# prva verzija (fali jedan dio koji ostane u boji)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as Image
from sklearn.cluster import KMeans

# ucitaj sliku
img = Image.imread("imgs\imgs\\test_1.jpg")

# prikazi originalnu sliku
plt.figure()
plt.title("Originalna slika")
plt.imshow(img)
plt.tight_layout()
plt.show()

# pretvori vrijednosti elemenata slike u raspon 0 do 1
img = img.astype(np.float64) / 255

# transfromiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
w,h,d = img.shape
img_array = np.reshape(img, (w*h, d))

# rezultatna slika
img_array_aprox = img_array.copy()

colors_num = np.unique(img_array)
print('Broj boja u originalnoj slici: ', len(colors_num))

# ... (učitavanje i transformacija ostaju isti) ...

# 2. Primjena K-means
km = KMeans(n_clusters=15, init='random', n_init=5, random_state=0)
km.fit(img_array) # Fitaj na originalu

# 3. Zamjena vrijednosti (Vektorizirano)
labels = km.predict(img_array)
centers = km.cluster_centers_
img_array_aprox = centers[labels] # Mnogo brže od for petlje

# Povratak u format slike
img_finished = np.reshape(img_array_aprox, (w, h, d))

# 6. Lakat metoda (na originalu!)
J_values = []
for i in range(1, 10):
    km_test = KMeans(n_clusters=i, n_init=5)
    km_test.fit(img_array)
    J_values.append(km_test.inertia_)

# 7. Prikaz grupa
for i in range(km.n_clusters):
    img_bin = (labels == i).reshape((w, h)) # Binarna maska
    plt.figure()
    plt.imshow(img_bin, cmap='gray')
    plt.title(f'Grupa {i+1}')
    plt.show()

#vec nakon 2 grupe boja jasno se vide razlike i tekst se moze procitati

colors_num = np.unique(img_array)
print('broj boja u originalnoj slici')
