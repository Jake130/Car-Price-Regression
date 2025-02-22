import torch.nn
import torch.functional as F

n_attrs = 10

#Nerual Net Hyperparams.
USE_RELU = True
DEPTH = 5       #How many layers?
WIDTH = 4       #Width of layers
L2_NORM = 0.01


STOCHASTIC = False
LEARNING_RATE = 0.1
BATCH_SIZE = 32

class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(n_attrs, WIDTH),
            torch.nn.ReLU() if USE_RELU	else torch.nn.Sigmoid(),
            *[torch.nn.Linear(WIDTH,WIDTH) if (i%2==0) else torch.nn.ReLU() if USE_RELU else torch.nn.Sigmoid() for i in range(DEPTH*2)],
            torch.nn.Linear(WIDTH, 1)
        )
        print(self.layers)

    def forward(self, x):
        self.layers(x)

model = MLP()


#Train
#for i in max_iters:
#    pass

mlp = MLP()