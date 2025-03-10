"""
Utility script to create transformed dataset in Pickle format

by Jake Kolster, Alex João Peterson Santos
March 6, 2025
for CS 453 Project
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
import numpy as np
import pickle
import os

CSV_DATASET_FILEPATH = '../data/car_price_dataset.csv'
SAVE_TO_PKL_DIRPATH = '../data/transformed/'

# load dataset from CSV
dataset = 'car_price_dataset.csv'
data = pd.read_csv(CSV_DATASET_FILEPATH)

# Define how to transform each feature type
numerical_cols = ['Year', 'Engine_Size', 'Mileage', 'Doors', 'Owner_Count']
numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
categorical_cols = ['Brand', 'Model', 'Fuel_Type', 'Transmission']
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(drop='first', sparse_output=False))])

# Fit and transform features into a (samples, features) numpy array
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numerical_cols),
    ('cat', categorical_transformer, categorical_cols)
])
transformed_input_data = preprocessor.fit_transform(data[numerical_cols + categorical_cols]).astype(np.float32)

num_samples, num_features = transformed_input_data.shape
print(f"Transformed input data contains {num_samples} samples and {num_features} features")

# Concatenate untransformed target as the last column of the array, now (samples, features+target)
transformed_concatenated_data = np.concatenate((transformed_input_data, data['Price'].to_numpy().reshape(-1, 1)), axis=1)

#Split Data
train_data, unseen_data = train_test_split(np.concatenate((transformed_input_data,data['Price'].to_numpy().reshape(-1,1)), axis=1), test_size=.3, random_state=42)
val_data, test_data = train_test_split(unseen_data, test_size=1/3, random_state=42)

print(f"Training data (samples, features+target): {train_data.shape}")
print(f"Validation data (samples, features+target): {val_data.shape}")
print(f"Test data (samples, features+target): {test_data.shape}")

# erase existing pickles, if any
if os.path.exists(SAVE_TO_PKL_DIRPATH):
    if len(os.listdir(SAVE_TO_PKL_DIRPATH)) > 0:
        print("Erasing existing pickles...")
    for file in os.listdir(SAVE_TO_PKL_DIRPATH):
        os.remove(os.path.join(SAVE_TO_PKL_DIRPATH, file))
else:
    raise ValueError("Directory for saving pickles does not exist")

#Save Data to pickles
with open(os.path.join(SAVE_TO_PKL_DIRPATH, "train_t_car_price_dataset.pkl"), 'wb') as f:
    pickle.dump(train_data, f)

with open(os.path.join(SAVE_TO_PKL_DIRPATH, "val_t_car_price_dataset.pkl"), 'wb') as f:
    pickle.dump(val_data, f)

with open(os.path.join(SAVE_TO_PKL_DIRPATH, "test_t_car_price_dataset.pkl"), 'wb') as f:
    pickle.dump(test_data, f)


print("Saved data to pickle files")