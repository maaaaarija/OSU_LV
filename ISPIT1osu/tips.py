import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


df_tips = sns.load_dataset('tips')

print(df_tips.columns)
print(df_tips.head(10))
print(df_tips.tail(10))


sns.pairplot(df_tips, hue = 'tip')
plt.show()