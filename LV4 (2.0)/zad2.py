'''
Na temelju rješenja prethodnog zadatka izradite model koji koristi i kategoricku
varijable „Fuel Type“ kao ulaznu velicinu. Pri tome koristite 1-od-K kodiranje kategorickih
velicina. Radi jednostavnosti nemojte skalirati ulazne velicine. Komentirajte dobivene rezultate.
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
ohe_columns = ohe.get_feature_names_out(['Fuel Type'])
X_encoded_df = pd.DataFrame(X_encoded, columns=ohe_columns, index=data.index)

numerical_features = data.select_dtypes(include='number')
full_data = pd.concat([numerical_features, X_encoded_df], axis=1)

X = full_data.drop(['CO2 Emissions (g/km)'], axis=1)
y = full_data['CO2 Emissions (g/km)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

linearModel = lm.LinearRegression()
linearModel.fit(X_train, y_train)

y_test_prediction = linearModel.predict(X_test)

plt.figure()
plt.scatter(y_test, y_test_prediction, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Real Values')
plt.ylabel('Predicted Values')
plt.title('Real vs Predicted CO2 (with Fuel Type)')
plt.show()

absolute_errors = abs(y_test - y_test_prediction)
max_error_index = absolute_errors.idxmax()
max_error = absolute_errors[max_error_index]
vehicle_model = data.loc[max_error_index, 'Model']

print(f'Model coefficients: {linearModel.coef_}')
print(f'Maximum absolute error: {max_error:.2f} g/km')
print(f'Model of the vehicle associated with maximum error: {vehicle_model}')