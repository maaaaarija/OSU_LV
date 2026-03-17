'''
Skripta zadatak_1.py ucitava podatkovni skup iz data_C02_emission.csv.
Dodajte programski kod u skriptu pomocu kojeg možete odgovoriti na sljedeca pitanja:
a) Koliko mjerenja sadrži DataFrame? Kojeg je tipa svaka velicina? Postoje li izostale ili
duplicirane vrijednosti? Obrišite ih ako postoje. Kategoricke velicine konvertirajte u tip
category.
b) Koja tri automobila ima najvecu odnosno najmanju gradsku potrošnju? Ispišite u terminal:
ime proizvodaca, model vozila i kolika je gradska potrošnja.
c) Koliko vozila ima velicinu motora izmedu 2.5 i 3.5 L? Kolika je prosjecna C02 emisija
plinova za ova vozila?
d) Koliko mjerenja se odnosi na vozila proizvodaca Audi? Kolika je prosjecna emisija C02
plinova automobila proizvodaca Audi koji imaju 4 cilindara?
e) Koliko je vozila s 4,6,8. . . cilindara? Kolika je prosjecna emisija C02 plinova s obzirom na
broj cilindara?
f) Kolika je prosjecna gradska potrošnja u slucaju vozila koja koriste dizel, a kolika za vozila
koja koriste regularni benzin? Koliko iznose medijalne vrijednosti?
g) Koje vozilo s 4 cilindra koje koristi dizelski motor ima najvecu gradsku potrošnju goriva?
h) Koliko ima vozila ima rucni tip mjenjaca (bez obzira na broj brzina)?
i) Izracunajte korelaciju izmedu numerickih velicina. Komentirajte dobiveni rezultat.
'''

import pandas as pd
import numpy as np

data = pd.read_csv('data_C02_emission.csv')

data = data.dropna(axis = 0)
data = data.drop_duplicates()
data = data.reset_index(drop = True)

data[['Make', 'Model', 'Vehicle Class', 'Transmission', 'Fuel Type']] = data[['Make', 'Model', 'Vehicle Class', 'Transmission', 'Fuel Type']].astype('category')

data_sortedbyconsumption = data.sort_values(by = 'Fuel Consumption City (L/100km)', ascending = True)
print("Najmanja gradska potrošnja:\n", data_sortedbyconsumption[['Make', 'Model', 'Fuel Consumption City (L/100km)']].head(3))
print("Najveća gradska potrošnja:\n", data_sortedbyconsumption[['Make', 'Model', 'Fuel Consumption City (L/100km)']].tail(3))

data_motorsize = data[data['Engine Size (L)'].between(2.5, 3.5)]
print('Broj vozila (2.5-3.5 L): ', len(data_motorsize))
print('Prosjecna CO2 emisija: ', data_motorsize['CO2 Emissions (g/km)'].mean())

audi_data = data[data['Make'] == 'Audi']
print('Broj Audi mjerenja: ', len(audi_data)) 
audi4cylinders = audi_data[audi_data['Cylinders'] == 4]
print('Prosječna CO2 Audi 4 cil: ', audi4cylinders['CO2 Emissions (g/km)'].mean().round(2))

data_groupedbycylinders = data.groupby('Cylinders')
print('Prosječna CO2 emisija po cilindrima:\n', data_groupedbycylinders['CO2 Emissions (g/km)'].mean().round(2))

diesel_cars = data[data['Fuel Type'] == 'D']
gasoline_cars = data[data['Fuel Type'] == 'X']
print('Diesel prosjek:', diesel_cars['Fuel Consumption City (L/100km)'].mean().round(2))
print('Benzin prosjek:', gasoline_cars['Fuel Consumption City (L/100km)'].mean().round(2))

cars_manual = data[data['Transmission'].astype(str).str.startswith('M')]
print('Broj ručnih mjenjača: ', len(cars_manual))

print(data.corr(numeric_only = True))

