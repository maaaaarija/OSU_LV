'''
Skripta Zadatak_1.py ucitava CIFAR-10 skup podataka. Ovaj skup sadrži
50000 slika u skupu za ucenje i 10000 slika za testiranje. Slike su RGB i rezolucije su 32x32.
Svakoj slici je pridružena jedna od 10 klasa ovisno koji je objekt prikazan na slici. Potrebno je:
1. Proucite dostupni kod. Od kojih se slojeva sastoji CNN mreža? Koliko ima parametara
mreža?
2. Pokrenite ucenje mreže. Pratite proces ucenja pomocu alata Tensorboard na sljedeci nacin.
Pokrenite Tensorboard u terminalu pomocu naredbe:
tensorboard–logdir=logs
i zatim otvorite adresu http://localhost:6006/ pomo´cu web preglednika.
3. Proucite krivulje koje prikazuju tocnost klasifikacije i prosjecnu vrijednost funkcije gubitka
na skupu podataka za ucenje i skupu podataka za validaciju. Što se dogodilo tijekom ucenja
mreže? Zapišite tocnost koju ste postigli na skupu podataka za testiranje.
'''


import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from matplotlib import pyplot as plt

# 1. Učitavanje CIFAR-10 skupa podataka 
# Skup sadrži 50.000 slika za učenje i 10.000 za testiranje 
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Prikaži 9 slika iz skupa za učenje radi vizualizacije
plt.figure(figsize=(8,8))
for i in range(9):
    plt.subplot(330 + 1 + i)
    plt.xticks([]), plt.yticks([])
    plt.imshow(X_train[i])
plt.show()

# 2. Priprema podataka
# Skaliranje na raspon [0,1] pretvorbom u float32 
X_train_n = X_train.astype('float32') / 255.0
X_test_n = X_test.astype('float32') / 255.0

# 1-od-K kodiranje labela (One-hot encoding)
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 3. Izgradnja CNN mreže 
model = keras.Sequential()

# Ulazni sloj: Slike su 32x32 piksela s 3 kanala (RGB) 
model.add(layers.Input(shape=(32, 32, 3)))

# Prvi blok: Konvolucija (32 filtra) + Sažimanje 
model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

# Drugi blok: Konvolucija (64 filtra) + Sažimanje
model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

# Treći blok: Konvolucija (128 filtra) + Sažimanje 
model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

# Ravnanje i potpuno povezani slojevi 
model.add(layers.Flatten())
model.add(layers.Dense(500, activation='relu'))
model.add(layers.Dense(10, activation='softmax')) # Izlazni sloj za 10 klasa

# Ispis strukture mreže i broja parametara 
model.summary()

# 4. Definiranje callback funkcija 
# TensorBoard bilježi proces učenja za vizualizaciju 
my_callbacks = [
    keras.callbacks.TensorBoard(log_dir='logs/zadatak_1', update_freq=100)
]

# 5. Prevođenje i učenje modela 
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Učenje s 10% validacijskog skupa 
history = model.fit(X_train_n,
                    y_train,
                    epochs=40,
                    batch_size=64,
                    callbacks=my_callbacks,
                    validation_split=0.1)

# 6. Evaluacija modela na testnom skupu 
score = model.evaluate(X_test_n, y_test, verbose=0)
print(f'Točnost na testnom skupu podataka: {100.0*score[1]:.2f}%')
print(f'Gubitak (Loss) na testnom skupu: {score[0]:.4f}')


'''
1. Od kojih se slojeva sastoji mreža? 
Mreža se sastoji od:Conv2D slojeva: Izvlače značajke (rubove, teksture) pomoću 32, 64 i 128 
filtara.MaxPooling2D slojeva: Smanjuju dimenzije i broj parametara, čineći mrežu robusnijom.
Flatten sloja: Pretvara 2D mape u 1D vektor za ulaz u klasifikator.Dense slojeva: Potpuno 
povezani slojevi koji donose konačnu odluku o klasi.
2. Koliko parametara ima mreža? 
Broj parametara možeš vidjeti pokretanjem naredbe model.summary(). U ovoj konkretnoj 
arhitekturi, najveći broj parametara dolazi iz prvog Dense sloja nakon operacije Flatten. 
Točan broj će biti ispisan u konzoli (bit će ih preko 1.000.000 zbog Dense(500) sloja).
3. Što se dogodilo tijekom učenja? Nakon što pokreneš TensorBoard naredbom 
tensorboard --logdir logs, primijetit ćeš:Overfitting (Pretjerano usklađivanje): 
Vjerojatno ćeš vidjeti da točnost na trening skupu nastavlja rasti prema 95%+, 
dok se točnost na validacijskom skupu stabilizira ili čak počne padati nakon određenog 
broja epoha.Gubitak (Loss): Gubitak na treningu će padati, ali na validaciji bi mogao 
početi rasti, što je jasan znak da su potrebni Dropout ili Early Stopping (što se 
rješava u sljedećim zadacima).
'''
