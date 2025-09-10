import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
import joblib
import os.path
import openpyxl
from openpyxl import load_workbook

tree = joblib.load("model_tree.joblib")
tree_grid = joblib.load("model_tree_with_cv.joblib")
forest_grid = joblib.load("model_random_forest.joblib")