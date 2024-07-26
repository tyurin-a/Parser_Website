from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
#from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
import openpyxl
import joblib

# Display in terminal all columns for viewing
pd.set_option('display.max_columns', None)

file = r'C:\Users\Артем\Desktop\Здания\Results\test_data_12.07.xlsx'

xl = pd.ExcelFile(file)
df = xl.parse()
#print(df.dtypes)

address = df[['shortname_region', 'formalname_region', 'shortname_city', 'formalname_city', 'address']]

df['energy_efficiency'] = df['energy_efficiency'].astype('category')
classes = ['G', 'F', 'E', 'D', 'C', 'B', 'B+', 'B++', 'A', 'A+', 'A++']
encoder = OrdinalEncoder(categories = [classes])
df['energy_efficiency_encoded'] = encoder.fit_transform(df[['energy_efficiency']])
#print(df[['energy_efficiency','energy_efficiency_encoded']].head())
target = df['energy_efficiency_encoded']

df['project_type'] = df['project_type'].astype('category')
project = df['project_type']
df.drop(['shortname_region', 'formalname_region', 'shortname_city', 'formalname_city', 'address', 'energy_efficiency', 'project_type', 'energy_efficiency_encoded'], axis=1, inplace=True)

# Change value type from text columns (from object to categorial)
df[['house_type', 'is_alarm', 'foundation_type', 'floor_type', 'wall_material', 
    'chute_type', 'electrical_type', 'heating_type', 'hot_water_type', 'cold_water_type', 'sewerage_type', 
    'gas_type', 'ventilation_type', 'firefighting_type', 'drainage_type']] = df[['house_type', 'is_alarm', 'foundation_type', 'floor_type', 'wall_material', 
    'chute_type', 'electrical_type', 'heating_type', 'hot_water_type', 'cold_water_type', 'sewerage_type', 
    'gas_type', 'ventilation_type', 'firefighting_type', 'drainage_type']].apply(lambda x: x.astype('category'))

# Factorize columns (drop_first=True in pd.get_dummies need to emit N-1 variables to avoid collinearity
df['is_alarm'] = [1 if i == 'Нет' else 0 for i in df['is_alarm']]

# # Alternative way to factorize with OneHotEncoding algorythm: it works correct, but cannot rename all columns for further analisys
# encoder = OneHotEncoder(sparse_output = False)
# onehot_columns = encoder.fit_transform(df[['house_type', 'foundation_type', 'floor_type', 'wall_material', 
#     'chute_type', 'electrical_type', 'heating_type', 'hot_water_type', 'cold_water_type', 'sewerage_type', 
#     'gas_type', 'ventilation_type', 'firefighting_type', 'drainage_type']])
# df = pd.concat([df, pd.DataFrame(onehot_columns)], axis=1)

df = pd.get_dummies(df, columns=['house_type', 'foundation_type', 'floor_type', 'wall_material', 
    'chute_type', 'electrical_type', 'heating_type', 'hot_water_type', 'cold_water_type', 'sewerage_type', 
    'gas_type', 'ventilation_type', 'firefighting_type', 'drainage_type'], drop_first=True)

col = df.columns
print(col)
print(df.head())

# Define train and test datasets
X_train, X_holdout, y_train, y_holdout = train_test_split(df.values, target, test_size=0.2,random_state=17)

# print(df.dtypes)
# Decision Tree
# tree = DecisionTreeClassifier(max_depth=19, max_features = 20, random_state=17)
# # KNN: doesn't work with NaN values
# #knn = KNeighborsClassifier(n_neighbors=10)

# tree.fit(X_train, y_train)
# #knn.fit(X_train, y_train)

# tree_pred = tree.predict(X_holdout)
# t = accuracy_score(y_holdout, tree_pred)
# #knn_pred = knn.predict(X_holdout)
# #k = accuracy_score(y_holdout, knn_pred)

# print(t)
# #print(k)

# joblib.dump(tree, "model_tree.joblib")

#Decision Tree with cross-validation
tree = DecisionTreeClassifier(max_depth=19, max_features = 20, random_state=17)
tree_params = {'max_depth': range(1,20), 'max_features': range(4,27)}
tree_grid = GridSearchCV(tree, tree_params, cv=5, n_jobs=-1, verbose=True)
tree_grid.fit(X_train, y_train)

tg_bp = tree_grid.best_params_
tg_bs = tree_grid.best_score_
tg = accuracy_score(y_holdout, tree_grid.predict(X_holdout))

#print(tg_bp)
#print(tg_bs)
print(tg)

joblib.dump(tree_grid, "model_tree_with_cv.joblib")

#Random forest with cross-validation
# forest = RandomForestClassifier(n_estimators=100, max_depth = 10, max_features = 8, n_jobs=-1, random_state=17)

# #print(np.mean(cross_val_score(forest, X_train, y_train, cv=5)))
# forest_params = {'max_depth': range(1,20), 'max_features': range(4,27)}
# forest_grid = GridSearchCV(forest, forest_params, cv=5, n_jobs=-1, verbose=True)
# forest_grid.fit(X_train, y_train)

# fg_bp = forest_grid.best_params_
# fg_bs = forest_grid.best_score_
# fg = accuracy_score(y_holdout, forest_grid.predict(X_holdout))

# #print(fg_bp)
# #print(fg_bs)
# print(fg)

# joblib.dump(forest_grid, "model_random_forest.joblib")