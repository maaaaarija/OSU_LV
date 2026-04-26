import seaborn as sns 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = sns.load_dataset('titanic')

print(df.columns)

print(df.head(10))
print(df.tail(10))


sns.pairplot(df.sample(300), hue = 'survived')
plt.show()