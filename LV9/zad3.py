'''
Dodajte funkciju povratnog poziva za rano zaustavljanje koja ce zaustaviti proces
ucenja nakon što se 5 uzastopnih epoha ne smanji prosjecna vrijednost funkcije gubitka na
validacijskom skupu.
'''


import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

# 1. Priprema podataka (isto kao i do sada)
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train_n = X_train.astype('float32') / 255.0
X_test_n = X_test.astype('float32') / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 2. Arhitektura mreže (zadržavamo Dropout iz Zadatka 2)
model = keras.Sequential([
    layers.Input(shape=(32, 32, 3)),
    
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.2),
    
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.2),
    
    layers.Flatten(),
    layers.Dense(500, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(10, activation='softmax')
])

# 3. Definiranje EarlyStopping i TensorBoard callback funkcija
my_callbacks = [
    # Rano zaustavljanje: prati se val_loss (gubitak na validacijskom skupu) 
    # patience=5 znači da se učenje prekida nakon 5 epoha bez poboljšanja 
    keras.callbacks.EarlyStopping(monitor='val_loss', 
                                  patience=5, 
                                  verbose=1,
                                  restore_best_weights=True),
    
    keras.callbacks.TensorBoard(log_dir='logs/cnn_early_stopping', update_freq=100)
]

# 4. Prevođenje i učenje
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Postavljamo veći broj epoha (npr. 100), jer će EarlyStopping prekinuti učenje ranije
model.fit(X_train_n, y_train,
          epochs=100, 
          batch_size=64,
          callbacks=my_callbacks,
          validation_split=0.1)

# 5. Evaluacija
score = model.evaluate(X_test_n, y_test, verbose=0)
print(f'Točnost s Early Stoppingom: {100.0*score[1]:.2f}%')




'''
Kako radi EarlyStopping u ovom kodu?U klasi EarlyStopping koristimo sljedeće ključne 
argumente iz dokumentacije:monitor='val_loss': Pratimo vrijednost funkcije gubitka na 
validacijskom skupu jer ona najbolje pokazuje generalizaciju modela.patience=5: 
Ovo je uvjet iz tvog zadatka. Ako gubitak na validaciji ne padne ni jednom tijekom 5 
uzastopnih epoha, algoritam se zaustavlja.restore_best_weights=True: (Dodatno) Ova opcija 
osigurava da model na kraju učenja zadrži težine iz epohe u kojoj je imao najbolji 
rezultat, a ne iz zadnje (koja je vjerojatno lošija zbog overfittinga).

Što ćeš vidjeti u konzoli/logovima?Iako smo postavili epochs=100, mreža vjerojatno 
neće doći do stote epohe. Čim krivulja validacijskog gubitka postane ravna ili 
počne rasti (što ukazuje na početak pretjeranog usklađivanja), Keras će ispisati 
poruku: "Epoch X: early stopping".
'''
