import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

df = pd.read_csv('Pokemon.csv', index_col=0)

df.head()
# Recommended
sns.lmplot(x='Attack', y='Defense', data=df)
