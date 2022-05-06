import seaborn as sns; sns.set_theme()
import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn import preprocessing, model_selection, linear_model, metrics

pd.set_option('display.max_columns', None)
df = pd.read_csv('fuel_stations_csv.csv')
fuel_stations = df.drop(['EV Pricing (French)', 'Intersection Directions (French)', 'Access Days Time (French)', 'BD Blends (French)', 'Groups With Access Code (French)',
                         'Station Name', 'Street Address', 'Intersection Directions', 'Plus4', 'Station Phone', 'Groups With Access Code', 'Access Days Time',
                         'Cards Accepted', 'BD Blends', 'NG Fill Type Code', 'NG PSI', 'EV Level1 EVSE Num', 'EV Level1 EVSE Num', 'EV DC Fast Count', 'EV Other Info',
                         'EV Network', 'EV Network Web', 'ID', 'Owner Type Code', 'Federal Agency ID', 'Federal Agency Name', 'Hydrogen Status Link', 'NG Vehicle Class',
                         'LPG Primary', 'E85 Blender Pump', 'EV Connector Types', 'Country', 'Hydrogen Is Retail', 'Access Code', 'Access Detail Code', 'Federal Agency Code',
                         'Facility Type', 'CNG Dispenser Num', 'CNG On-Site Renewable Source', 'CNG Total Compression Capacity', 'CNG Storage Capacity', 'LNG On-Site Renewable Source',
                         'E85 Other Ethanol Blends', 'EV Pricing', 'LPG Nozzle Types', 'Hydrogen Pressures', 'Hydrogen Standards', 'CNG Fill Type Code', 'CNG PSI',
                         'CNG Vehicle Class', 'LNG Vehicle Class', 'EV On-Site Renewable Source', 'Restricted Access'], axis = 1)
def stripp(x):
  if str(x) != 'nan':
      return int(str(x)[:4])
  else:
      return x

fuel_stations['Open Date'] = np.where(fuel_stations['Open Date'].isnull(), fuel_stations['Expected Date'], fuel_stations['Open Date'])
fuel_stations['Open Year'] = fuel_stations['Open Date'].apply(stripp)
fuel_stations = fuel_stations[fuel_stations['Open Year'] > 500]
fuel_stations = pd.concat([fuel_stations, pd.get_dummies(fuel_stations['State'])], axis = 1)
fuel_stations['State Count'] = fuel_stations.groupby('State')['State'].transform('count')
fuel_stations['Year Count'] = fuel_stations.groupby('Open Year')['Open Year'].transform('count')
fuel_stations['Lat STD'] = fuel_stations.groupby('Open Year')['Latitude'].transform('std')
fuel_stations['Long STD'] = fuel_stations.groupby('Open Year')['Longitude'].transform('std')

scale = preprocessing.StandardScaler()
X = scale.fit_transform(fuel_stations[['Latitude', 'Longitude']])
fuel_stations_train_X = X[:int(len(X) * .9)]
fuel_stations_test_X = X[int(len(X) * .9):]
fuel_stations_train_Y = fuel_stations['Open Year'][:int(len(fuel_stations) * .9)]
fuel_stations_test_Y = fuel_stations['Open Year'][int(len(fuel_stations) * .9):]

lin_reg = linear_model.LinearRegression().fit(fuel_stations_train_X, fuel_stations_train_Y)
print("Linear regression score:", lin_reg.score(fuel_stations_train_X, fuel_stations_train_Y))
log_reg = linear_model.LogisticRegression().fit(fuel_stations_train_X, fuel_stations_train_Y)
print("Logistic regression score:", log_reg.score(fuel_stations_train_X, fuel_stations_train_Y))
residual = (fuel_stations_test_Y - lin_reg.predict(fuel_stations_test_X))
sns.scatterplot(x = lin_reg.predict(fuel_stations_test_X), y = residual)
plt.show()

#sns.color_palette("viridis", as_cmap = True)
#sns.scatterplot(data = fuel_stations, x = 'Open Year', y = 'Latitude', hue = 'Open Year')
#plt.show()
#sns.scatterplot(data = fuel_stations, x = 'Open Year', y = 'Longitude', hue = 'Open Year')
#plt.show()
