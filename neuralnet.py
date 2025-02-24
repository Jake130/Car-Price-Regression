import torch.nn
from torch.utils.data import Dataset, DataLoader
import torch.functional as F
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import numpy as np

#Nerual Net Hyperparams.
USE_RELU = True
DEPTH = 5       #How many layers?
WIDTH = 50       #Width of layers
L2_NORM = 0.01

EPOCHS = 10
STOCHASTIC = False
LEARNING_RATE = 0.001
BATCH_NUM = 5
BATCH_SIZE = 32

#Dataset class To help create Torch DataLoader
class CSV_Dataset(Dataset):
    def __init__(self, dataframe:pd.DataFrame):
        super().__init__()
        self.data = dataframe
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):                       #TODO: Need to convert categorical vars to other datatype, for NN
        sample = self.data[index][:-1]
        label = self.data[index][-1]
        return torch.tensor(sample.astype(dtype=float), dtype=torch.float32),torch.tensor(label.astype(float), dtype=torch.float32).view(-1)

#Load & Split Data into train,eval,test
dataset = 'car_price_dataset.csv'
data = pd.read_csv(f'./data/{dataset}')                 #This is all the data, but it becomes the train_data later


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
length,n_attrs = processed_data.shape

#Split Data
train_data,temp_data = train_test_split(np.concatenate((processed_data,data['Price'].to_numpy().reshape(-1,1)), axis=1), test_size=.3, random_state=42)
val_data,test_data = train_test_split(temp_data, test_size=1/3, random_state=42)

print(f"# Attributes\t{n_attrs}")
print(train_data.shape)
print(val_data.shape)
print(test_data.shape)

#Save Data to CSV
#train_data.to_csv(f'./data/train_{dataset}', index=False)
#val_data.to_csv(f'./data/val_{dataset}', index=False)
#test_data.to_csv(f'./data/test_{dataset}', index=False)

#Create Data Loaders        (Torch's way of selecting batches)
train_dl = DataLoader(CSV_Dataset(train_data), batch_size=BATCH_SIZE, shuffle=True)
val_dl = DataLoader(CSV_Dataset(val_data), batch_size=BATCH_SIZE, shuffle=True)
test_dl = DataLoader(CSV_Dataset(test_data), batch_size=BATCH_SIZE, shuffle=True)
print("Datasets saved")

class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(n_attrs, WIDTH),
            torch.nn.ReLU() if USE_RELU	else torch.nn.Sigmoid(),
            *[torch.nn.Linear(WIDTH,WIDTH) if (i%2==0) else torch.nn.ReLU() if USE_RELU else torch.nn.Sigmoid() for i in range(DEPTH*2)],
            torch.nn.Linear(WIDTH, 1)
        )
        #print(self.layers)

    def forward(self, x):
        return self.layers(x)

model = MLP()
criterion = torch.nn.MSELoss()
optim = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

#Training
model.train()
for epoch in range(EPOCHS):
    for batch_idx,(features,target) in enumerate(train_dl):     #Used for mini-batching
        optim.zero_grad()                                       #Set gradient to zero
        output = model(features)                                #Generate our prediction
        loss = criterion(output, target)                        #Calculate Loss
        if batch_idx==0:
            print(f"Epoch {epoch}\tLoss:\t{loss}")
        loss.backward()                                         #Compute Gradient
        optim.step()                                            #Update Model Params"""

#Validation

#Testing
model.eval()
predictions = np.array([])
targets = np.array([])
with torch.no_grad():
    for features,target in test_dl:
        output = model(features)
        predictions = np.concatenate((predictions,output.flatten()))        #[32]
        targets = np.concatenate((targets,target.flatten()))                #[32]

predictions = np.array(predictions)
targets = np.array(targets)


mse = mean_squared_error(targets, predictions)
print(f"Shape: {targets.shape}")
raw_error = np.sum(np.abs(targets - predictions)) / targets.shape[0]
print(f"Raw Error: {raw_error}")
print(f"Test MSE: {mse}")

