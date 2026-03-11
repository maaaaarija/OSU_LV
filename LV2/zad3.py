'''
Skripta zadatak_3.py ucitava sliku road.jpg. Manipulacijom odgovarajuce numpy matrice pokušajte:
a) posvijetliti sliku,
b) prikazati samo drugu cetvrtinu slike po širini, 
c) zarotirati sliku za 90 stupnjeva u smjeru kazaljke na satu,
d) zrcaliti sliku.
'''

import numpy as np
import matplotlib.pyplot as plt

img = plt.imread('road.jpg')
img = img[:, :, 0].copy()

brightness = 50
brightened_image = np.clip(img.astype(np.uint16) + brightness, 0, 255).astype(np.uint8)
plt.figure()
plt.imshow(brightened_image, cmap='gray')
plt.show()

width = img.shape[1]
q = width // 4
second_quarter_img = img[:, q : 2*q]
plt.figure()
plt.imshow(second_quarter_img, cmap='gray')
plt.show()

rotated_img = np.rot90(img, k=-1)
plt.figure()
plt.imshow(rotated_img, cmap='gray')
plt.show()

mirrored_img = np.flip(img, axis=1)
plt.figure()
plt.imshow(mirrored_img, cmap='gray')
plt.show()