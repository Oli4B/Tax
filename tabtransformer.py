import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from tab_transformer_pytorch import TabTransformer

from functions import store_results, scramble_data
from data import load_data, CATS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MCC(nn.Module):
    def __init__(self, transformer, num_cat, dim):
        super().__init__()
        self.transformer = transformer
        self.classifiers = nn.ModuleList()
        
        for _ in range(num_cat):
            self.classifiers.append(nn.Linear(dim, 1))

        self.act = nn.Sigmoid()
        
    def forward(self, x):
        out = self.transformer(x)
        classification = []
        for i, classifier in enumerate(self.classifiers):
            out_classification = classifier(out[:,i])
            out_classification = self.act(out_classification)
            classification.append(out_classification)
            
        return torch.cat(classification, dim=1)

def tabtransformer(utrain_loader, ltrain_loader, val_loader, test_loader, num_epochs, input_dim, p_m, depth, heads, dim, mlp):
    model_pre = TabTransformer(
        categories = tuple(CATS), 
        num_continuous = input_dim,
        dim = dim,    
        dim_out = 2,
        depth = depth,   
        heads = heads, 
        mlp_hidden_mults = mlp,
        mlp_act = nn.ReLU(),
        num_special_tokens=2)
    
    # model_pre = model_pre.to(device)
    
    model = MCC(model_pre.transformer, input_dim, dim)
    model = model.to(DEVICE)
    
    loss_fn = torch.nn.BCELoss().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    num_epochs = 5
    
    epochs = tqdm(range(num_epochs), leave=False)
    for epoch in epochs:
        model.train()
    
        epoch_loss = 0
        for input in utrain_loader:
            mask, scrambled = scramble_data(input, p_m)
            
            mask = mask.to(DEVICE)
            scrambled = scrambled.to(DEVICE)
            
            output = model(scrambled)
            loss = loss_fn(output, mask)
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            epochs.set_description(f"Epoch: {epoch}\t Batch loss: {loss.item()/len(input)}")
                
    model = model.cpu()
    
    for param in model_pre.transformer.parameters():
        param.requires_grad = False
        
    model_pre = model_pre.to(DEVICE)
    
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    optimizer = torch.optim.Adam(model_pre.parameters(), lr=0.0005)
    num_epochs = 100
    train_loss_list = []
    val_loss_list = []
    
    epochs = tqdm(range(num_epochs), leave=False)
    
    for epoch in epochs:
    
        model_pre.train()
        epoch_loss = 0
        for input_cat, input_cont, target in ltrain_loader:
            input_cat = input_cat.to(DEVICE)
            input_cont = input_cont.to(DEVICE)
            target = target.to(DEVICE)
            
            output = model_pre(input_cat.clone(), input_cont)
    
            loss = loss_fn(output, target.long())
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            epochs.set_description(f"Epoch: {epoch}\t loss: {loss.item()/len(target)}")
            epoch_loss += loss.item()

        train_loss_list.append(epoch_loss)
        

        model_pre.eval()
        with torch.no_grad():
            val_loss = 0
            for input_cat, input_cont, target in val_loader:
                input_cat = input_cat.to(DEVICE)
                input_cont = input_cont.to(DEVICE)
                target = target.to(DEVICE)
                
                output = model_pre(input_cat.clone(), input_cont)
        
                loss = loss_fn(output, target.long())
                
                val_loss += loss.item()/len(target) # Fix loss to be for each shape
            
        val_loss_list.append(val_loss)
    
    model_pre.eval()
    with torch.no_grad():
        target = []
        predict = []
        for input_cat, input_cont, y in test_loader:        
            input_cat = input_cat.clone().to(DEVICE)
            input_cont = input_cont.to(DEVICE)
            target.extend(y)
        
            predict.extend(torch.sigmoid(model_pre(input_cat, input_cont)).cpu()[:,1])
    
    predict = np.array(predict)
    
    store_results(target, predict, fname=f"tabtrans_{p_m}_{depth}_{heads}_{dim}_{mlp}.json")


if __name__ == '__main__':
    unlabelled, sample, _ = load_data(unlabelled=True)

    trainp = .9
    batch_size = 1024

    total_size = len(unlabelled)
    train_size = int(trainp * total_size)
    val_size = total_size-train_size

    train_data, val_data = random_split(unlabelled, [train_size, val_size])
    utrain_loader = DataLoader(train_data, batch_size=batch_size)
    uval_loader = DataLoader(val_data, batch_size=batch_size)


    trainp, testp = 0.8, 0.1
    total_size = len(sample)
    train_size = int(trainp * total_size)
    test_size = int(testp * total_size)
    val_size = total_size-train_size-test_size

    train_data, test_data, val_data = random_split(sample, [train_size, test_size, val_size])

    batch_size = 256
    
    class_tensor = train_data[:][1].long()
    class_counts = torch.bincount(class_tensor)
    class_weights = 1 / class_counts
    weights = [class_weights[c] for c in class_tensor]
    sampler = WeightedRandomSampler(weights, len(weights))
    
    batch_size = 256
    ltrain_loader = DataLoader(train_data, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    params = {
        "p_m":   .1,
        "depth": 4,
        "heads": 8,
        "dim":   8,
        "mlp":   (4,2)
    }
    
    tabtransformer(utrain_loader, ltrain_loader, val_loader, test_loader, 5, train_data[:][1].size()[0], **params)

