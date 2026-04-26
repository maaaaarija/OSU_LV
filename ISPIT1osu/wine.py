import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_wine

wine = load_wine()

print(wine.feature_names)

data = pd.DataFrame(wine.data, columns = wine.feature_names)
data['target'] = wine.target 

print(data.target)
print(data.head(10))
print(data.tail(10))

sns.pairplot(data.iloc[:, [0, 1, 2, 13]], hue = 'target')
plt.show()

print(data.columns)

