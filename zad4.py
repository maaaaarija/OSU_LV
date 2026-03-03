'''
Napišite Python skriptu koja ce ucitati tekstualnu datoteku naziva song.txt.
Potrebno je napraviti rjecnik koji kao kljuceve koristi sve razlicite rijeci koje se pojavljuju u 
datoteci, dok su vrijednosti jednake broju puta koliko se svaka rijec (kljuc) pojavljuje u datoteci. 
Koliko je rijeci koje se pojavljuju samo jednom u datoteci? Ispišite ih.
'''

import string

fhand = open('Song.txt')
sadrzaj = fhand.read()
rijeci = sadrzaj.split()

rjecnik = {}

for r in rijeci:
    rjecnik[r] = rjecnik.get(r, 0) + 1

jedinstveneRijeci = []

for kljuc in rjecnik:
    if rjecnik[kljuc] == 1:
        jedinstveneRijeci.append(kljuc)

brojJedinstvenihRijeci = len(jedinstveneRijeci)
print(f"Broj jedinstvenih rijeci: {brojJedinstvenihRijeci}")
print(f"{jedinstveneRijeci}")

fhand.close()

