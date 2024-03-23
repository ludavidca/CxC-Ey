import torch
import pandas as pd
from torch.utils.data import DataLoader
from torch import nn
import math


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device}')

TestDf = pd.read_csv(r'./mlData/TestData.csv').sample(frac = 1)
TrainDf = pd.read_csv(r'./mlData/TrainData.csv').sample(frac = 1)

testTensor = []
trainTensor = []


maxValue = math.log10(max(TestDf['ex_hectares'].max(), TrainDf['ex_hectares'].max()))
minValue = math.log10(min(TestDf['ex_hectares'].min(), TrainDf['ex_hectares'].min()))

def normalize(input):
    global minValue,maxValue
    return (math.log10(input) - minValue)/(maxValue-minValue)

def unnormalize(input):
    global minValue,maxValue
    return 10**(input*(maxValue-minValue)+minValue)


def createTensor(row, tensorList):
    ntl = [x for x in tensorList]
    try:
        long = row.fire_location_longitude
        lat = row.fire_location_latitude
        datenum = row.data_value
        timenum = row.time_value
        temp = row.temperature
        relhum =  row.relative_humidity
        wSpeed = row.wind_speed
        finalSize = normalize(row.ex_hectares)
        inputtensor = torch.tensor([long, lat, datenum, timenum, temp, relhum, wSpeed]).to(device=device)
        labeltensor = torch.tensor([finalSize]).to(device=device)
        ntl.append([inputtensor, labeltensor])

    except Exception as error:
        print(error,row)
    
    return ntl


for row in TestDf.itertuples():
    testTensor = createTensor(row, testTensor)

for row in TrainDf.itertuples():
    trainTensor = createTensor(row, trainTensor)

batch_size = 64

train_dataloader = DataLoader(trainTensor, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(testTensor, batch_size=batch_size, shuffle=False)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(7, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits
    
model = NeuralNetwork().to(device)
#model.load_state_dict(torch.load('./mlData/model.pth'))

learning_rate = 1e-3

loss_fn = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):  
     # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

    
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    test_loss, meanError = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            meanError += ((((unnormalize(pred)-unnormalize(y))**2)**0.5)/unnormalize(y)).sum().item()
        
    test_loss /= size
    meanError /= size
    print(f"Test Error: Mean Error:{meanError} Avg loss: {test_loss:>8f} \n")


epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")

torch.save(model.state_dict(), "./mlData/model.pth")
print("Saved PyTorch Model State to model.pth")