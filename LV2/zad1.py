'''
Pomocu funkcija numpy.array i matplotlib.pyplot pokušajte nacrtati sliku2.3 u okviru skripte zadatak_1.py. 
Igrajte se sa slikom, promijenite boju linija, debljinu linije i sl.
'''

import numpy as np
import matplotlib.pyplot as plt

x = np.array([1, 2, 3, 3, 1], float)
y = np.array([1, 2, 2, 1, 1], float)

plt.plot(x, y, 'r', linewidth = 1, marker = '*', markersize = 5)
plt.axis([0.0, 4.0, 0.0, 4.0])
plt.xlabel('x os')
plt.ylabel('y os')
plt.title('Primjer')
plt.show()