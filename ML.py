from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
#from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import openpyxl

file = r'C:\Users\Артем\Desktop\Здания\Results\test_data_12.07.xlsx'

xl = pd.ExcelFile(file)
df = xl.parse()

address = df[['shortname_region', 'formalname_region', 'shortname_city', 'formalname_city', 'address']]
project = df['project_type']
df['energy_efficiency'] = pd.factorize(df['energy_efficiency'])[0]
target = df['energy_efficiency']
df.drop(['shortname_region', 'formalname_region', 'shortname_city', 'formalname_city', 'address', 'energy_efficiency'], axis=1, inplace=True)

df['project_type'] = pd.factorize(df['project_type'])[0]
df['house_type'] = pd.factorize(df['house_type'])[0]
df['is_alarm'] = pd.factorize(df['is_alarm'])[0]
df['foundation_type'] = pd.factorize(df['foundation_type'])[0]
df['floor_type'] = pd.factorize(df['floor_type'])[0]
df['wall_material'] = pd.factorize(df['wall_material'])[0]
df['chute_type'] = pd.factorize(df['chute_type'])[0]
df['electrical_type'] = pd.factorize(df['electrical_type'])[0]
df['heating_type'] = pd.factorize(df['heating_type'])[0]
df['hot_water_type'] = pd.factorize(df['hot_water_type'])[0]
df['cold_water_type'] = pd.factorize(df['cold_water_type'])[0]
df['sewerage_type'] = pd.factorize(df['sewerage_type'])[0]
df['gas_type'] = pd.factorize(df['gas_type'])[0]
df['ventilation_type'] = pd.factorize(df['ventilation_type'])[0]
df['firefighting_type'] = pd.factorize(df['firefighting_type'])[0]
df['drainage_type'] = pd.factorize(df['drainage_type'])[0]

X_train, X_holdout, y_train, y_holdout = train_test_split(df.values, target, test_size=0.5,
random_state=17)

tree = DecisionTreeClassifier(max_depth=19, max_features = 20, random_state=17)
#knn = KNeighborsClassifier(n_neighbors=10)

tree.fit(X_train, y_train)
#knn.fit(X_train, y_train)

tree_pred = tree.predict(X_holdout)
t = accuracy_score(y_holdout, tree_pred)
#knn_pred = knn.predict(X_holdout)
#k = accuracy_score(y_holdout, knn_pred)

print(t)
#print(k)

#Cross-validation
tree_params = {'max_depth': range(1,20), 'max_features': range(4,27)}
tree_grid = GridSearchCV(tree, tree_params, cv=5, n_jobs=-1, verbose=True)
tree_grid.fit(X_train, y_train)

tg_bp = tree_grid.best_params_
tg_bs = tree_grid.best_score_
tg = accuracy_score(y_holdout, tree_grid.predict(X_holdout))

#print(tg_bp)
#print(tg_bs)
print(tg)

#Random forest with cross-validation
forest = RandomForestClassifier(n_estimators=100, max_depth = 10, max_features = 8, n_jobs=-1, random_state=17)

#print(np.mean(cross_val_score(forest, X_train, y_train, cv=5)))
forest_params = {'max_depth': range(1,20), 'max_features': range(4,27)}
forest_grid = GridSearchCV(forest, forest_params, cv=5, n_jobs=-1, verbose=True)
forest_grid.fit(X_train, y_train)

fg_bp = forest_grid.best_params_
fg_bs = forest_grid.best_score_
fg = accuracy_score(y_holdout, forest_grid.predict(X_holdout))

#print(fg_bp)
#print(fg_bs)
print(fg)