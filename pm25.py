import torch
import pandas as pd
import numpy as np
import datetime
import matplotlib
import matplotlib.pyplot as plt
from sklearn import preprocessing

pm = pd.read_csv('PM25.csv')

# use get_dummies to encode data (for string type fields)
features = pd.get_dummies(pm)

# get results for labels and remove results from features
labels = np.array(features['Concentration'])
features = features.drop('Concentration', axis = 1)

# use standard scaler to normalized
features = np.array(features)
inputFeatures = preprocessing.StandardScaler().fit_transform(features)

# build network
inputSize = inputFeatures.shape[1]
hiddenSize = 1024
outputSize = 1
batchSize = 12

# using linear moodel and sigmoid function
nn = torch.nn.Sequential(
    torch.nn.Linear(inputSize, hiddenSize),
    torch.nn.Sigmoid(),
    torch.nn.Linear(hiddenSize, outputSize),
)

# get loss
cost = torch.nn.MSELoss(reduction = 'mean')

# optimization
optimizer = torch.optim.Adam(nn.parameters(), lr = 0.001)

# training
trainNum = 1024

lossList = []
for i in range(trainNum):
    batchLoss = []
    featureSize = len(inputFeatures)
    for start in range(0, featureSize, batchSize):
        end = (start + batchSize) if (start + batchSize < featureSize) else featureSize

        # set features to tensor
        xaxis = torch.tensor(inputFeatures[start:end], dtype = torch.float, requires_grad = True)

        # set results to tensor
        yaxis = torch.tensor(labels[start:end], dtype = torch.float, requires_grad = True)

        # predict by using features
        prediction = nn(xaxis)

        # compute loss
        loss = cost(prediction, yaxis)

        optimizer.zero_grad()
        loss.backward(retain_graph = True)

        optimizer.step()
        batchLoss.append(loss.data.numpy())

    if 0 == i % 100:
        lossMean = np.mean(batchLoss)
        lossList.append(lossMean)

result = torch.tensor(inputFeatures, dtype=torch.float)
predict = nn(result).data.numpy()

years = features[:, 0]
months = features[:, 1]
dates = features[:, 2]
datetimes = [datetime.datetime.strptime(("%04d" % year) + '-' + ("%02d" % month) + '-' + ("%02d" % date), '%Y-%m-%d') for year, month, date in zip(years, months, dates)]

trueData = pd.DataFrame(data = {'Date': datetimes, 'Concentration': labels})
predictData = pd.DataFrame(data = {'Date': datetimes, 'Concentration': predict.reshape(-1)})

matplotlib.rc('font', family = 'DFKai-SB')

plt.plot(trueData['Date'], trueData['Concentration'], 'b+', label = 'Real Data')

plt.plot(predictData['Date'], predictData['Concentration'], 'r+', label = 'Perdict Data')
plt.xticks(rotation = '60')
plt.legend()


plt.xlabel('Date')
plt.ylabel('Concentration')
plt.title('Predict for PM2.5')
plt.savefig('predict.png')