import torch.nn
from torch.utils.data import Dataset, DataLoader
import torch.functional as F
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt

# Neural Net Hyperparameters.
USE_RELU = True
DEPTH = 1       #How many layers?
WIDTH = 50      #Width of layers
L2_NORM = 0.01

EPOCHS = 150
STOCHASTIC = False
LEARNING_RATE = 0.001
BATCH_NUM = 5
BATCH_SIZE = 32

# Dataset class to help create Torch DataLoader
class CSV_Dataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame):
        super().__init__()
        self.data = dataframe

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index][:-1]
        label = self.data[index][-1]
        return torch.tensor(sample.astype(float), dtype=torch.float32), torch.tensor(label.astype(float), dtype=torch.float32).view(-1)

# Load & Split Data into train, validation, test
dataset = 'car_price_dataset.csv'
data = pd.read_csv(f'./data/{dataset}')

# Preprocessing pipelines for features
numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(drop='first', sparse_output=False))])

# Define categorical and numerical features
categorical_cols = ['Brand', 'Model', 'Fuel_Type', 'Transmission']
numerical_cols = ['Year', 'Engine_Size', 'Mileage', 'Doors', 'Owner_Count']

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numerical_cols),
    ('cat', categorical_transformer, categorical_cols)
])

# Fit and transform features
processed_data = preprocessor.fit_transform(data[numerical_cols + categorical_cols]).astype(np.float32)
length, n_attrs = processed_data.shape



#### NEW STUFF
# Extract and scale the target variable (Price)
price = data['Price'].to_numpy().reshape(-1, 1).astype(np.float32)
target_scaler = StandardScaler()
scaled_price = target_scaler.fit_transform(price)

# Concatenate processed features with the scaled target
concatenated_data = np.concatenate((processed_data, scaled_price), axis=1)
#### NO MORE NEW STUFF


# Split Data
train_data, temp_data = train_test_split(concatenated_data, test_size=0.3, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=1/3, random_state=42)

print(f"# Attributes\t{n_attrs}")
print(train_data.shape)
print(val_data.shape)
print(test_data.shape)

# Create Data Loaders
train_dl = DataLoader(CSV_Dataset(train_data), batch_size=BATCH_SIZE, shuffle=True)
val_dl = DataLoader(CSV_Dataset(val_data), batch_size=BATCH_SIZE, shuffle=True)
test_dl = DataLoader(CSV_Dataset(test_data), batch_size=BATCH_SIZE, shuffle=True)
print("Datasets saved")

class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Linear(n_attrs, 1)

    def forward(self, x):
        return self.layers(x)

model = MLP()
criterion = torch.nn.MSELoss()
optim = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# Training
loss_tracker = []
model.train()
for epoch in range(EPOCHS):
    epoch_loss = 0
    for batch_idx, (features, target) in enumerate(train_dl):
        optim.zero_grad()
        output = model(features)
        loss = criterion(output, target)
        epoch_loss += loss.item()
        if batch_idx == 0:
            print(f"Epoch {epoch}\tLoss:\t{loss}")
        loss.backward()
        optim.step()
    loss_tracker.append(epoch_loss / len(train_dl))

x_axis = range(len(loss_tracker))
plt.plot(x_axis, loss_tracker)
plt.show()

# Testing
model.eval()
predictions_scaled = np.array([])
targets_scaled = np.array([])

with torch.no_grad():
    for features, target in test_dl:
        output = model(features)
        predictions_scaled = np.concatenate((predictions_scaled, output.flatten()))
        targets_scaled = np.concatenate((targets_scaled, target.flatten()))

mse = mean_squared_error(targets_scaled, predictions_scaled)
mae = mean_absolute_error(targets_scaled, predictions_scaled)
print(f"Scaled Test MSE: {mse}")
print(f"Scaled Test MAE: {mae}")

# Unscale the predictions and targets back to original
predictions_original = target_scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
targets_original = target_scaler.inverse_transform(targets_scaled.reshape(-1, 1)).flatten()

mse_original = mean_squared_error(targets_original, predictions_original)
mae_original = mean_absolute_error(targets_original, predictions_original)
print(f"Original Test MSE: {mse_original}")
print(f"Original Test MAE: {mae_original}")
print(f"Shape: {targets_original.shape}")
raw_error = np.sum(np.abs(targets_original - predictions_original)) / targets_original.shape[0]
print(f"Raw Error (same as MAE): {raw_error}")