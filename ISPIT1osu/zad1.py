'''
Datoteka titanic.csv sadrži podatke o putnicima broda Titanic, koji je potonuo
1912. godine. Upoznajte se s datasetom i dodajte programski kod u skriptu pomo´cu kojeg možete
odgovoriti na sljede´ca pitanja:
a) Za koliko žena postoje podatci u ovom skupu podataka?
b) Koliki postotak osoba nije preživio potonu´ce broda?
c) Pomo´cu stupˇcastog dijagrama prikažite postotke preživjelih muškaraca (zelena boja) i žena
(žuta boja). Dodajte nazive osi i naziv dijagrama. Komentirajte korelaciju spola i postotka
preživljavanja.
d) Kolika je prosjeˇcna dob svih preživjelih žena, a kolika je prosjeˇcna dob svih preživjelih
muškaraca?
e) Koliko godina ima najstariji preživjeli muškarac u svakoj od klasa? Komentirajte.
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


data = pd.read_csv('titanic.csv')

zene = data[data['Sex'] == 'female']
brZena = len(zene)

passenger = len(data['PassengerId'])
nisuPrezivjeli = data[data['Survived'] == 0]
postotakB = ( len(nisuPrezivjeli) / passenger ) * 100

muskarci = data[data['Sex'] == 'male']
brMuskaraca = len(muskarci)

prezivjeleZene = data[(data['Sex'] == 'female') & (data['Survived'] == 1)]
postotakPrezivjelihZena = (len(prezivjeleZene) / brZena) * 100
prezivjeliMuskarci = data[(data['Sex'] == 'male') & (data['Survived'] == 1)]
postotakPrezivjelihMuskaraca = ( len(prezivjeliMuskarci) / brMuskaraca ) * 100

plt.figure()
kategorije = ['zene', 'muskarci']
boje = ['yellow', 'green']
postotci = [postotakPrezivjelihZena, postotakPrezivjelihMuskaraca]
plt.bar(kategorije, postotci, color = boje)
plt.xlabel('spol')
plt.ylabel('postotak prezivjelih')
plt.show()


dobZena = prezivjeleZene['Age'].mean()
dobMuskaraca = prezivjeliMuskarci['Age'].mean()

grupirani = prezivjeliMuskarci.groupby('Pclass')
grupiraniGodine = grupirani['Age']
godine = grupiraniGodine.max()
