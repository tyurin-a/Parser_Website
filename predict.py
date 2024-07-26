import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
import joblib

# Display in terminal all columns for viewing
pd.set_option('display.max_columns', None)

file = r'C:\Users\Артем\Desktop\Здания\Results\data_24.07.xlsx'

xl = pd.ExcelFile(file)
df = xl.parse()
#print(df.dtypes)

address = df[['shortname_region', 'formalname_region', 'shortname_city', 'formalname_city', 'address']]

# df['energy_efficiency'] = df['energy_efficiency'].astype('category')
# classes = ['G', 'F', 'E', 'D', 'C', 'B', 'B+', 'B++', 'A', 'A+', 'A++']
# encoder = OrdinalEncoder(categories = [classes])
# df['energy_efficiency_encoded'] = encoder.fit_transform(df[['energy_efficiency']])
# #print(df[['energy_efficiency','energy_efficiency_encoded']].head())
# target = df['energy_efficiency_encoded']

df['project_type'] = df['project_type'].astype('category')
project = df['project_type']
target = df['energy_efficiency']
df.drop(['shortname_region', 'formalname_region', 'shortname_city', 'formalname_city', 'energy_efficiency', 'address', 'project_type'], axis=1, inplace=True)

# Change value type from text columns (from object to categorial)
df[['house_type', 'is_alarm', 'foundation_type', 'floor_type', 'wall_material', 
    'chute_type', 'electrical_type', 'heating_type', 'hot_water_type', 'cold_water_type', 'sewerage_type', 
    'gas_type', 'ventilation_type', 'firefighting_type', 'drainage_type']] = df[['house_type', 'is_alarm', 'foundation_type', 'floor_type', 'wall_material', 
    'chute_type', 'electrical_type', 'heating_type', 'hot_water_type', 'cold_water_type', 'sewerage_type', 
    'gas_type', 'ventilation_type', 'firefighting_type', 'drainage_type']].apply(lambda x: x.astype('category'))

# Factorize columns (drop_first=True in pd.get_dummies need to emit N-1 variables to avoid collinearity
df['is_alarm'] = [1 if i == 'Нет' else 0 for i in df['is_alarm']]

df = pd.get_dummies(df, columns=['house_type', 'foundation_type', 'floor_type', 'wall_material', 
    'chute_type', 'electrical_type', 'heating_type', 'hot_water_type', 'cold_water_type', 'sewerage_type', 
    'gas_type', 'ventilation_type', 'firefighting_type', 'drainage_type'], drop_first=True)

#print(df.head())
col = df.columns
print(col)

# loaded_tree_grid = joblib.load("model_tree_with_cv.joblib")
# result = loaded_tree_grid.predict(df)

# print(result)