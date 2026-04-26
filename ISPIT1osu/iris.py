from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = load_iris()

df = pd.DataFrame(data.data, columns = data.feature_names)
df['target'] = data.target
print(data.target)


print(df.head(10))

print(df.tail(10))


plt.figure()
pd.plotting.scatter_matrix(df, c=df['target'])
sns.pairplot(df, hue='target', palette='viridis')
plt.show()
