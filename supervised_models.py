import numpy as np
from tqdm import tqdm
import xgboost as xgb
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler

from functions import store_results
from data import load_data

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MLP(nn.Module):
    # dims as [in, inter, out]
    def __init__(self, dims):
        super(MLP, self).__init__()

        layers = []
        for i, dim in enumerate(dims):
            # Add layers until list clears
            if i < len(dims) - 1:
                layers.append(nn.Linear(dim, dims[i+1]))
            # Add a ReLU layer if there is antohter layer
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
                
        self.model = nn.Sequential(layers)
    
    def forward(self, x):
        out = self.model(x)
        return out


def mlp(train_loader, test_loader, val_loader, num_epochs, dims):
    model = MLP(dims)
    model.to(DEVICE)
    
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum').to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for _ in tqdm(num_epochs):
        model.train()
    
        for x, y in train_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            
            predict = model(x)
            
            loss = loss_fn(predict, y.squeeze().long())
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        model.eval()
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(DEVICE)
                y = y.to(DEVICE)
                
                predict = model(x)
                
                loss = loss_fn(predict, y.squeeze().long())
    
    # Test
    model.eval()
    with torch.no_grad():
        target = []
        predict = []
        for x, y in test_loader:
            input = x.to(DEVICE)
    
            target.extend(y)
            predict.extend(torch.sigmoid(model(input)).cpu()[:,1])

    predict = np.array(predict)

    store_results(target, predict, fname="MLP.json")


def xgboost(train_loader, test_loader):
    inputs = []
    targets = []
    for x, y in train_loader:
        inputs.extend(x.numpy())
        targets.extend(y.numpy())
    
    inputs = np.array(inputs)
    targets = np.array(targets)
    
    clf = xgb.XGBClassifier(tree_method="gpu_hist", enable_categorical=True)
    clf.fit(inputs, targets)
    
    inputs = []
    targets = []
    for x, y in test_loader:
        inputs.extend(x.numpy())
        targets.extend(y.numpy())
    
    inputs = np.array(inputs)
    target = np.array(targets)
    
    predict = clf.predict_proba(inputs)[:,1]

    store_results(target, predict, fname="XGB.json")


#TOETS of STEEK,
# TO run use 
def supervised_model(model, labelled_data, trainp, testp, batch_size, model_params=None):
    total_size = len(labelled_data)
    train_size = int(trainp * total_size)
    test_size = int(testp * total_size)
    val_size = total_size-train_size-test_size

    train_data, test_data, val_data = random_split(labelled_data, [train_size, test_size, val_size])
    
    class_tensor = train_data[:][1].long()
    class_counts = torch.bincount(class_tensor)
    class_weights = 1 / class_counts
    weights = [class_weights[c] for c in class_tensor]
    sampler = WeightedRandomSampler(weights, len(weights))
    
    train_loader = DataLoader(train_data, batch_size=batch_size, sampler=sampler)
    test_loader = DataLoader(test_data, batch_size=batch_size)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    if model == "XGB":
        xgboost(train_loader, test_loader)

    if model == "MLP":
        mlp(train_loader, test_loader, val_loader, 200, **model_params)


if __name__ == '__main__':
    sample, biased = load_data()
    supervised_model("XGB", sample, 0.5, 0.5, 1024)

    # params = {
    #     "epochs": 200, 
    #     "dims":   [sample[0][0].size()[0], 2]
    # }
    # supervised_model("MLP", sample, 0.8, 0.1, 256, params)
