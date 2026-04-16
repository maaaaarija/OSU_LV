'''
MNIST podatkovni skup za izgradnju klasifikatora rukom pisanih znamenki
dostupan je u okviru Keras-a. Skripta zadatak_1.py uˇcitava MNIST podatkovni skup te podatke
priprema za uˇcenje potpuno povezane mreže.
1. Upoznajte se s uˇcitanim podacima. Koliko primjera sadrži skup za uˇcenje, a koliko skup za
testiranje? Kako su skalirani ulazni podaci tj. slike? Kako je kodirana izlazne veliˇcina?
2. Pomocu matplotlib biblioteke prikažite jednu sliku iz skupa podataka za uˇcenje te ispišite
njezinu oznaku u terminal.
3. Pomocu klase Sequential izgradite mrežu prikazanu na slici 8.5. Pomocu metode
.summary ispišite informacije o mreži u terminal.
4. Pomocu metode .compile podesite proces treniranja mreže.
5. Pokrenite uˇcenje mreže (samostalno definirajte broj epoha i veliˇcinu serije). Pratite tijek
uˇcenja u terminalu.
6. Izvršite evaluaciju mreže na testnom skupu podataka pomocu metode .evaluate.
7. Izraˇcunajte predikciju mreže za skup podataka za testiranje. Pomocu scikit-learn biblioteke
prikažite matricu zabune za skup podataka za testiranje.
8. Pohranite model na tvrdi disk.
'''


import numpy as np
import keras
from keras import layers
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Postavke
num_classes = 10
input_shape = (28, 28, 1)

# 1. Učitavanje podataka 
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# 2. Prikaz slike i oznake 
plt.imshow(x_train[0], cmap='gray')
plt.title(f'Oznaka: {y_train[0]}')
plt.show()

# 3. Skaliranje i priprema 
x_train_s = x_train.astype("float32") / 255
x_test_s = x_test.astype("float32") / 255
x_train_s = np.expand_dims(x_train_s, -1)
x_test_s = np.expand_dims(x_test_s, -1)

# 1-od-K kodiranje oznaka
y_train_s = keras.utils.to_categorical(y_train, num_classes)
y_test_s = keras.utils.to_categorical(y_test, num_classes)

# 4. Izgradnja mreže 
model = keras.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Flatten(),
    layers.Dense(100, activation='relu'),
    layers.Dense(50, activation='relu'),
    layers.Dense(10, activation='softmax')
])
model.summary() 

# 5. Kompajliranje 
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 6. Učenje (Batch size 32, Epochs 20 prema tekstu) 
model.fit(x_train_s, y_train_s, batch_size=32, epochs=20, validation_split=0.1)

# 7. Evaluacija i matrica zabune 
score = model.evaluate(x_test_s, y_test_s, verbose=0)
print('Test accuracy:', score[1])

predictions = model.predict(x_test_s)
y_test_p = np.argmax(predictions, axis=1) # Pretvaramo vjerojatnosti natrag u znamenke (0-9)

cm = confusion_matrix(y_test, y_test_p)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

# 8. Spremanje 
model.save('model_zadatak_1.keras')