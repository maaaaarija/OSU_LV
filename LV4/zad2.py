'''
Na temelju rješenja prethodnog zadatka izradite model koji koristi i kategoriˇcku
varijable „Fuel Type“ kao ulaznu veliˇcinu. Pri tome koristite 1-od-K kodiranje kategoriˇckih
veliˇcina. Radi jednostavnosti nemojte skalirati ulazne veliˇcine. Komentirajte dobivene rezultate.
Kolika je maksimalna pogreška u procjeni emisije C02 plinova u g/km? O kojem se modelu
vozila radi?
'''

import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

data = pd.read_csv('data_C02_emission.csv')

ohe = OneHotEncoder()
X_encoded = ohe.fit_transform(data[['Fuel Type']]).toarray()
X_encoded = pd.DataFrame(X_encoded, columns=ohe.get_feature_names_out(['Fuel Type']))


X_num = data[['Engine Size (L)', 'Cylinders', 'Fuel Consumption City (L/100km)', 'Fuel Consumption Hwy (L/100km)', 'Fuel Consumption Comb (L/100km)', 'Fuel Consumption Comb (mpg)']]

X = pd.concat([X_num, X_encoded], axis=1)
y = data['CO2 Emissions (g/km)']



