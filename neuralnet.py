import torch.nn
from torch.utils.data import Dataset, DataLoader
import torch.functional as F
import pandas as pd
from sklearn.model_selection import train_test_split
from math import floor, ceil

n_attrs = 10

#Nerual Net Hyperparams.
USE_RELU = True
DEPTH = 5       #How many layers?
WIDTH = 4       #Width of layers
L2_NORM = 0.01

EPOCHS = 10
STOCHASTIC = False
LEARNING_RATE = 0.1
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
        sample = self.data.iloc[index, 7:9].values
        label = self.data.iloc[index, -1]
        return torch.tensor(sample.astype(dtype=float), dtype=torch.float32),torch.tensor(label.astype(float), dtype=torch.float32)

#Load & Split Data into train,eval,test
dataset = 'car_price_dataset.csv'
data = pd.read_csv(f'./data/{dataset}')                 #This is all the data, but it becomes the train_data later
train_data,temp_data = train_test_split(data, test_size=.3, random_state=42)
val_data,test_data = train_test_split(temp_data, test_size=1/3, random_state=42)
length,n_attrs = train_data.shape
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
        self.layers(x)

model = MLP()
optim = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

#Training
for epoch in range(EPOCHS):
    print(f"Epoch {epoch}")
    for batch_idx,(data,target) in enumerate(train_dl):         #Used for mini-batching
        print(f"\tBatch {batch_idx}")
        optim.zero_grad()                                       #Set gradient to zero
        output = model(data)                                    #Generate our prediction
        loss = torch.nn.MSELoss(output, target)                 #TODO: Is the loss correct?
        loss.backward()
        optim.step()