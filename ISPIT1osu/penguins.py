import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = sns.load_dataset('penguins')
print(df.columns)
print(df.head(10))
print(df.tail(10))

print(df['species'].unique())
sns.pairplot(df.sample(200), hue = 'species')
plt.show()