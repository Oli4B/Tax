import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler, TensorDataset

from functions import store_results, scramble_data
from data import load_data, CAT_COLS, ALL_COLS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VIME_self(nn.Module):
    def __init__(self, dim, latent_dim):
        super(VIME_self, self).__init__()
        hidden_dim1 = max(latent_dim, dim//2)
        hidden_dim2 = max(latent_dim, hidden_dim1//2)
        self.encoder = nn.Sequential(
            nn.Linear(dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(dim, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, latent_dim),
            nn.ReLU())
        self.mask_estimator = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, dim),
            nn.Sigmoid())
        self.feature_estimator = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, dim))
        

    def forward(self, x):
        h = self.encoder(x)
        out_me = self.mask_estimator(h)
        out_fe = self.feature_estimator(h)
        return out_me, out_fe


class Feature_loss(nn.Module):
    def __init__(self, cat_columns, column_order):
        super(Feature_loss, self).__init__()

        self.cat_features = []
        self.cont_features = []
        for i, name in enumerate(column_order, start=0):
            if name in cat_columns:
                self.cat_features.append(i)
            else:
                self.cont_features.append(i)

        self.cont_fn = nn.MSELoss()
        self.cat_fn = nn.BCEWithLogitsLoss()

    def forward(self, input, target):
        cont_loss = self.cont_fn(input[:, self.cont_features], target[:, self.cont_features])
        cat_loss = self.cat_fn(input[:, self.cat_features], target[:, self.cat_features])
        
        return cont_loss+cat_loss


class VIME_semi(nn.Module):
    def __init__(self, latent_dim, output_dim, encoder):
        super(VIME_semi, self).__init__()
        hidden_dim = output_dim + latent_dim-output_dim//2
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim))
        
        self.encoder = encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        
    def forward(self, x):
        x = self.encoder(x)
        out_logit = self.predictor(x)
        return out_logit


class Variational_loss(nn.Module):
    def __init__(self):
        super(Variational_loss, self).__init__()

    def forward(self, output):
        return torch.sum(torch.var(output, dim=0))


class DVE(nn.Module):
    def __init__(self, dim, hidden_dim, comb_dim):
        super(DVE, self).__init__()
        self.inter = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, comb_dim),
            nn.ReLU()
        )

        self.out = nn.Sequential(
            nn.Linear(comb_dim+1, comb_dim),
            nn.ReLU(),
            nn.Linear(comb_dim, 1),
            nn.Sigmoid()
        )
        

    def forward(self, x, y, y_hat):
        x = torch.cat((x, y.unsqueeze(1)), dim=1).float()
        x = self.inter(x)
        x = torch.cat((x,torch.abs(y-y_hat).unsqueeze(1)), dim=1).float()
        
        out = self.out(x)
        return out


class DVE_loss(nn.Module):
    def __init__(self, epsilon, threshold):
        super(DVE_loss, self).__init__()
        self.epsilon = epsilon
        self.threshold = threshold

    def forward(self, input, est_data_val, reward):
        prob = torch.sum(input * torch.log(est_data_val+self.epsilon) +
                         (1 - input) * torch.log(1-est_data_val+self.epsilon))
        
        loss = (-reward*prob) + 1e3 * (torch.clamp(torch.mean(est_data_val)-self.threshold, min=0) +
                                       torch.clamp(1 - self.threshold - torch.mean(est_data_val), min=0))

        return loss

def vime(utrain_loader, uval_loader, train_data, val_data, test_data, p_m, alpha, beta, k, input_dim, latent_dim, epochs):
    # Pre-training embeddings
    model = VIME_self(input_dim, latent_dim)
    model.to(DEVICE)

    loss_fn_m = nn.BCELoss()
    loss_fn_f = Feature_loss(CAT_COLS, ALL_COLS)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = tqdm(range(1, epochs+1), leave=False)
    for _ in epochs:
        model.train()

        for input in utrain_loader:
            mask, scrambled = scramble_data(input, p_m)
            
            input = input.to(DEVICE)
            mask = mask.to(DEVICE)
            scrambled = scrambled.to(DEVICE)

            out_m, out_f = model(scrambled)
            loss_m = loss_fn_m(out_m, mask)
            loss_f = loss_fn_f(out_f, input)

            loss = loss_m + alpha*loss_f
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_u += loss.item()/len(utrain_loader)

        model.eval()
        with torch.no_grad():
            for input in uval_loader:
                mask, scrambled = scramble_data(input, p_m)
                
                input = input.to(DEVICE)
                mask = mask.to(DEVICE)
                scrambled = scrambled.to(DEVICE)
        
                out_m, out_f = model(scrambled)
                loss_m = loss_fn_m(out_m, mask)
                loss_f = loss_fn_f(out_f, input)
        
                loss = loss_m + alpha*loss_f

                val_loss_u += loss.item()/len(uval_loader)

    if dve:
        dve(model, utrain_loader, train_data, val_data, test_data, input_dim, latent_dim, p_m, alpha, beta, k)

    else:
        vime_predict(model, utrain_loader, train_data, val_data, test_data, latent_dim, p_m, alpha, beta, k)

def vime_predict(model, utrain_loader, train_data, val_data, test_data, latent_dim, p_m, alpha, beta, k):
    class_tensor = train_data[:][1].long()
    class_counts = torch.bincount(class_tensor)
    class_weights = 1 / class_counts
    weights = [class_weights[c] for c in class_tensor]
    sampler = WeightedRandomSampler(weights, len(weights))
    
    batch_size = 256
    train_l_loader = DataLoader(train_data, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)
    
    model_semi = VIME_semi(latent_dim, 2, model.encoder)
    model_semi.to(DEVICE)
    
    loss_fn_s = torch.nn.CrossEntropyLoss(reduction='sum').to(DEVICE)
    loss_fn_f = Variational_loss().to(DEVICE)
    optimizer = torch.optim.Adam(model_semi.parameters(), lr=0.005)
    num_epochs = 200
    
    train_loss_list = []
    val_loss_list = []
    
    epochs = tqdm(range(1, num_epochs+1), leave=False)
    model_semi.train()
    
    for epoch in epochs:
        unlabeld_iter = iter(utrain_loader)
    
        epoch_loss = 0
        for input_l, target in train_l_loader:
            input_u = next(unlabeld_iter)
    
            input_u_batch = []
            for i in range(k):
                _, scrambled = scramble_data(input_u, p_m)
                input_u_batch.append(scrambled)
    
            input_u_batch = torch.cat(input_u_batch, dim=0)
            
            input_l = input_l.to(DEVICE)
            target = target.to(DEVICE)
            input_u_batch = input_u_batch.to(DEVICE)
            
            predict_super = model_semi(input_l)
            predict_unsuper = model_semi(input_u_batch)
            
            supervised_loss = loss_fn_s(predict_super, target.squeeze().long())
            unsupervised_loss = loss_fn_f(predict_unsuper)
        
            loss = supervised_loss/len(train_data) + (beta/k)*unsupervised_loss/(len(input_u_batch)*len(train_l_loader))
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            epoch_loss += loss.item()

        train_loss_list.append(epoch_loss)
        epochs.set_description(f"Epoch: {epoch}\tlabelled loss: {train_loss_list[-1]}")
    
        model_semi.eval()
        with torch.no_grad():
            val_loss = 0
            for x, y in val_loader:
                x = x.to(DEVICE)
                y = y.to(DEVICE)
                
                predict = model_semi(x)
    
                loss = loss_fn_s(predict, y.squeeze().long())/len(x)
                val_loss += loss.item()
            
        val_loss_list.append(val_loss)
        
    # Test
    model_semi.eval()
    with torch.no_grad():
        target = []
        predict = []
        for x, y in test_loader:
            input = x.to(DEVICE)
    
            target.extend(y)
            predict.extend(torch.sigmoid(model_semi(input)).cpu()[:,1])
    
    predict = np.array(predict)

    store_results(target, predict, fname=f"vime_dve_{p_m}_{alpha}_{beta}_{k}.json")


def dve(model, utrain_loader, train_data, val_data, test_data, input_dim, latent_dim, p_m, alpha, beta, k):
    batch_size = 256
    val_loader = DataLoader(val_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    model_semi = VIME_semi(latent_dim, 2, model.encoder)
    model_semi.to(DEVICE)
    
    loss_fn_s = torch.nn.CrossEntropyLoss(reduction='sum').to(DEVICE)
    loss_fn_f = Variational_loss().to(DEVICE)
    optimizer = torch.optim.Adam(model_semi.parameters(), lr=0.005)
    
    model_DVE = DVE(input_dim+1, 100, 10)#latent_dim, 2, model.encoder)
    model_DVE.to(DEVICE)
    
    loss_dve_fn = DVE_loss(1e-8, 0.9)
    optimizer_dve = torch.optim.Adam(model_DVE.parameters(), lr=0.005)
    
    num_epochs_outer = 40
    num_epochs_inner = 20
    
    dve_loss_list = []
    
    model_DVE.train()
    
    epochs_outer = tqdm(range(1, num_epochs_outer+1))
    for epoch_iter in epochs_outer:
        # DVE
        model_semi.eval()
    
        input, target = train_data[:]
        input = input.to(DEVICE)
        target = target.to(DEVICE)
        with torch.no_grad():
            predict = torch.sigmoid(model_semi(input.to(DEVICE)))
            valid_perf = loss_fn_s(predict, target.squeeze().long()).item()
    
        est_data_val = model_DVE(input, target, predict[:,1])
        sel_prob_curr = torch.bernoulli(est_data_val).long()
    
        if sel_prob_curr.sum() == 0:
            est_data_val = .5 * torch.ones_like(est_data_val)
            sel_prob_curr = torch.bernoulli(est_data_val).long()
    
        unlabeld_iter = iter(utrain_loader)
        train_l_loader = DataLoader(
            TensorDataset(input, target, sel_prob_curr),
            batch_size=batch_size)#, sampler=sel_prob_curr)
    
        model_semi_copy = VIME_semi(latent_dim, 2, model.encoder)
        model_semi_copy.load_state_dict(model_semi.state_dict())
        model_semi_copy.to(DEVICE)
        model_semi_copy.train()
        
        epochs_inner = range(1, num_epochs_inner+1)
        for _ in epochs_inner:
            epoch_loss = 0
            for input_l, target, weights in train_l_loader:
                # Transform data
                input_u = next(unlabeld_iter)
        
                input_u_batch = []
                for _ in range(k):
                    _, scrambled = scramble_data(input_u, p_m)
                    input_u_batch.append(scrambled)
        
                input_u_batch = torch.cat(input_u_batch, dim=0)
                
                input_l = input_l.to(DEVICE)
                target = target.to(DEVICE)
                input_u_batch = input_u_batch.to(DEVICE)
    
                # Predict
                predict_super = model_semi_copy(input_l)
                predict_unsuper = model_semi_copy(input_u_batch)
                
                supervised_loss = loss_fn_s(weights*predict_super, (target.unsqueeze(1)*weights).squeeze().long())
                unsupervised_loss = loss_fn_f(predict_unsuper)
    
                # VIME
                loss = supervised_loss/len(train_data) + (beta/k)*unsupervised_loss/(len(input_u_batch)*len(train_l_loader))
        
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
                epoch_loss += loss.item()
            
        # DVE
        input, target = train_data[:]
        input = input.to(DEVICE)
        target = target.to(DEVICE)
        with torch.no_grad():
            predict = torch.sigmoid(model_semi(input.to(DEVICE)))
            train_perf = loss_fn_s(predict, target.squeeze().long()).item()
    
        dve_perf = train_perf - valid_perf
    
        loss_dve = loss_dve_fn(sel_prob_curr, est_data_val, dve_perf)
    
        optimizer_dve.zero_grad()
        loss_dve.backward()
        optimizer_dve.step()
    
        dve_loss_list.append(loss_dve.item())
        epochs_outer.set_description(f"Epoch: {epoch_iter}\tDVE loss: {dve_loss_list[-1]}")

        unlabeld_iter = iter(utrain_loader)
    
        # Update base model
        epochs_inner = range(1, num_epochs_inner+1)
        for epoch in epochs_inner:
            epoch_loss = 0
            for input_l, target, weights in train_l_loader:
                # Transform data
                input_u = next(unlabeld_iter)
        
                input_u_batch = []
                for i in range(k):
                    _, scrambled = scramble_data(input_u, p_m)
                    input_u_batch.append(scrambled)
        
                input_u_batch = torch.cat(input_u_batch, dim=0)
                
                input_l = input_l.to(DEVICE)
                target = target.to(DEVICE)
                input_u_batch = input_u_batch.to(DEVICE)
    
                # Predict
                predict_super = model_semi(input_l)
                predict_unsuper = model_semi(input_u_batch)
                
                supervised_loss = loss_fn_s(weights*predict_super, (target.unsqueeze(1)*weights).squeeze().long())
                unsupervised_loss = loss_fn_f(predict_unsuper)
    
                # VIME
                loss = supervised_loss/len(train_data) + (beta/k)*unsupervised_loss/(len(input_u_batch)*len(train_l_loader))
        
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
                epoch_loss += loss.item()
    
    
    # Test
    model_semi.eval()
    with torch.no_grad():
        target = []
        predict = []
        for x, y in test_loader:
            input = x.to(DEVICE)
    
            target.extend(y)
            predict.extend(torch.sigmoid(model_semi(input)).cpu()[:,1])
    
    predict = np.array(predict)
        
    store_results(target, predict, fname=f"vime_dve_{p_m}_{alpha}_{beta}_{k}.json")
        

if __name__ == '__main__':
    unlabelled, sample, biased = load_data(unlabelled=True)
    params = {
        "p_m":   .4,
        "alpha": 1,
        "beta":  1,
        "k":     1,
        "input_dim":  unlabelled.shape[1],
        "latent_dim": unlabelled.shape[1],
        "epochs":     5
    }
    
    trainp = .9
    batch_size = 4096

    total_size = len(unlabelled)
    train_size = int(trainp * total_size)
    val_size = total_size-train_size

    train_data, val_data = random_split(unlabelled, [train_size, val_size])
    utrain_loader = DataLoader(train_data, batch_size=batch_size)
    uval_loader = DataLoader(val_data, batch_size=batch_size)

    testp = 0.5, 0.5
    total_size = len(sample)
    test_size = int(testp * total_size)
    val_size = total_size-test_size
    
    test_data, val_data = random_split(sample, [test_size, val_size])
    train_data, _ = random_split(
        # column_permute(labelled_data_toets, f_permute),
        biased,
        [3200, len(biased)-3200])

    vime(utrain_loader, uval_loader, train_data, val_data, test_data, sample, **params)
