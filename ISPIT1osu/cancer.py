import seaborn as sns 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_breast_cancer


cancer = load_breast_cancer()
print(cancer.feature_names)

data = pd.DataFrame(cancer.data, columns = cancer.feature_names)
data['target'] = cancer.target 
print(data.target)

print(data.head(10))
print(data.tail(10))


sns.pairplot(data.iloc[:, [0, 1, 2, 3, 29, 30]], hue = 'target')
plt.show()