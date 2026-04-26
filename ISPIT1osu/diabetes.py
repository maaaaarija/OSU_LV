import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_diabetes


diabetes = load_diabetes()

print(diabetes.feature_names)

data = pd.DataFrame(diabetes.data, columns = diabetes.feature_names)
data['target'] = diabetes.target 

print(diabetes.target)

print(data.head(10))
print(data.tail(10))

sns.pairplot(data.iloc[:, [1, 2, 4, 5, 8, 10]], hue = 'target')
plt.show()