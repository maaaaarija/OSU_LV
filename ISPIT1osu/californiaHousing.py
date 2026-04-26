import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns 

from sklearn.datasets import fetch_california_housing


housing = fetch_california_housing()
 
print(housing.feature_names)
print(housing.target)

data = pd.DataFrame(housing.data, columns = housing.feature_names)
data['target'] = housing.target 

print(data.head(10))
print(data.tail(10))


sns.pairplot(data.sample(200), hue = 'target')
plt.show()