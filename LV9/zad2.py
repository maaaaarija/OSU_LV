'''
Modificirajte skriptu iz prethodnog zadatka na nacin da na odgovarajuca mjesta u
mrežu dodate droput slojeve. Prije pokretanja ucenja promijenite Tensorboard funkciju povratnog
poziva na nacin da informacije zapisuje u novi direktorij (npr. =/log/cnn_droput). Pratite tijek
ucenja. Kako komentirate utjecaj dropout slojeva na performanse mreže?
'''


import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

# 1. Učitavanje i priprema podataka (isto kao u prvom zadatku)
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train_n = X_train.astype('float32') / 255.0
X_test_n = X_test.astype('float32') / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 2. Izgradnja CNN mreže s Dropout slojevima
model = keras.Sequential()
model.add(layers.Input(shape=(32, 32, 3)))

# Prvi konvolucijski blok
model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.2)) # Dodano: isključuje 20% neurona 

# Drugi konvolucijski blok
model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.2)) # Dodano

# Treći konvolucijski blok
model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.2)) # Dodano

# Klasifikacijski dio
model.add(layers.Flatten())
model.add(layers.Dense(500, activation='relu'))
model.add(layers.Dropout(0.3)) # Dodano: obično veći postotak na Dense slojevima 
model.add(layers.Dense(10, activation='softmax'))

# 3. Promjena log direktorija za TensorBoard (prema uputi zadatka) 
my_callbacks = [
    keras.callbacks.TensorBoard(log_dir='logs/cnn_dropout', update_freq=100)
]

# 4. Prevođenje i učenje
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train_n, y_train,
          epochs=40,
          batch_size=64,
          callbacks=my_callbacks,
          validation_split=0.1)

# 5. Evaluacija
score = model.evaluate(X_test_n, y_test, verbose=0)
print(f'Točnost s Dropoutom na testnom skupu: {100.0*score[1]:.2f}%')




'''
Što smo točno promijenili?Dodavanje Dropout slojeva: Ubacili smo layers.Dropout nakon 
svakog sloja sažimanja i nakon glavnog gusto povezanog (Dense) sloja. Postotak (rate) 
smo postavili na 0.2 (20%) i 0.3 (30%), što su tipične preporučene vrijednosti.Novi log 
direktorij: Putanja u TensorBoard callbacku promijenjena je u 'logs/cnn_dropout' kako 
bi u alatu mogli usporediti krivulje ovog modela s onim iz prvog zadatka.


Kako Dropout utječe na performanse? (Komentar za izvještaj) Smanjenje razmaka (Gap): 
U prvom zadatku si vjerojatno primijetio da točnost na treningu ide do 99%, a na 
validaciji stane na npr. 75%. S dropoutom će se taj razmak smanjiti.Sporije učenje: 
Mreži će trebati malo više epoha da postigne visoku točnost jer stalno "otežavamo" 
posao neuronima gaseći ih.Bolja generalizacija: Konačna točnost na testnom skupu 
(podacima koje mreža nikad nije vidjela) trebala bi biti veća nego u prvom zadatku 
jer je model prisiljen naučiti robusnije značajke umjesto da "pamti" trening slike.


Što ćeš vidjeti u Tensorboardu?

Vidjet ćeš dvije krivulje (npr. plavu za prvi zadatak i crvenu za ovaj). 
Primijetit ćeš da je krivulja gubitka (loss) na validacijskom skupu kod dropout modela 
puno stabilnija i rjeđe počinje rasti (što je znak da nema toliko izraženog overfittinga).
'''
