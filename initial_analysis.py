import seaborn as sns; sns.set_theme()
import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn import preprocessing

pd.set_option('display.max_columns', None)
df = pd.read_csv('fuel_stations_csv.csv')
fuel_stations = df.drop(['EV Pricing (French)', 'Intersection Directions (French)', 'Access Days Time (French)', 'BD Blends (French)', 'Groups With Access Code (French)'], axis = 1)
def stripp(x):
  if str(x) != 'nan':
      return int(str(x)[:4])
  else:
      return x
fuel_stations['Open Year'] = fuel_stations['Open Date'].apply(stripp)

sns.countplot(data = fuel_stations, x = 'Open Year')
plt.show()
sns.countplot(data = fuel_stations, hue = 'State', x = 'Open Year')
plt.show()
sns.countplot(data = fuel_stations, x = 'ZIP')
plt.show()
