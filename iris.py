import torch
import torch.nn as nn
import pandas as pd
from sklearn import datasets
from sklearn import preprocessing

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
print(f"Using device: {device}")


iris = datasets.load_iris()
def namedata(data):
    return {
        'sepal_l': data[0],
        'sepal_w': data[1],
        'petal_l': data[2],
        'petal_w': data[3]
    }
table = list(map(namedata, iris.data))

tmap = {0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'}
targets = list(map(lambda item: tmap[item], iris.target))
table = [{**row, 'species': name} for row, name in zip(table, targets)]

df = pd.DataFrame(table)

le = preprocessing.LabelEncoder()

x = df[["sepal_l", "sepal_w", "petal_l", "petal_w"]].values
y = le.fit_transform(df["species"])

species = le.classes_

x = torch.tensor(x, device=device, dtype=torch.float32)
y = torch.tensor(y, device=device, dtype=torch.long)

model = nn.Sequential(
    nn.Linear(x.shape[1], 50),
    nn.ReLU(),
    nn.Linear(50, 25),
    nn.ReLU(),
    nn.Linear(25, len(species)),
    nn.LogSoftmax(dim=1),
)

model = torch.compile(model,backend="aot_eager").to(device)
criterion = nn.CrossEntropyLoss()  # cross entropy loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

model.train()
for epoch in range(10000):
    optimizer.zero_grad()
    out = model(x)
    # Note: CrossEntropyLoss combines nn.LogSoftmax() and nn.NLLLoss() so don't use Softmax in the model
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, loss: {loss.item()}")