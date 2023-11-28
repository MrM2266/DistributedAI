import torch
import torch.nn as nn
import pandas as pd
import ray
from sklearn import datasets
from sklearn import preprocessing
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig, Checkpoint, report
from ray.train.torch import prepare_model
from ray.train.torch import prepare_data_loader

def train_func(config):
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

    #model = torch.compile(model,backend="aot_eager").to(device)
    model = prepare_model(model)
    criterion = nn.CrossEntropyLoss()  # cross entropy loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    #train_loader = prepare_data_loader(train_loader)

    model.train()
    for epoch in range(1000):
        optimizer.zero_grad()
        out = model(x)
        # Note: CrossEntropyLoss combines nn.LogSoftmax() and nn.NLLLoss() so don't use Softmax in the model
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        report({"loss": loss.item()})
            
ray.init(address="auto")
scaling_config = ScalingConfig(num_workers=7, use_gpu=False) #zde si cluster žádám o resources - můj cluster má 8cpus - lze požádat o 7 workers (1 cpu je head)
trainer = TorchTrainer(train_func, scaling_config=scaling_config)
result = trainer.fit()