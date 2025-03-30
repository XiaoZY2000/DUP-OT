import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch import optim
import os

overwrite = False
dtype = torch.float32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

feature_dim = 768
hidden_dim = 384
target_dim = 128

class SharedDataset(Dataset):
    def __init__(self, item_features):
        self.item_features = item_features

    def __len__(self):
        return len(self.item_features)

    def __getitem__(self, index):
        return torch.tensor(self.item_features[index]).to(device).to(dtype)

class SharedAutoEncoder(nn.Module):
    def __init__(self, feature_dim=feature_dim, hidden_dim=hidden_dim, target_dim=target_dim):
        super(SharedAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, target_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(target_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        z = F.normalize(z, p=2, dim=1)
        x_recon = self.decoder(z)
        return x_recon, z

class ReconstructLoss(nn.Module):
    def __init__(self):
        super(ReconstructLoss, self).__init__()

    def forward(self, x_recon, x):
        cos_sim = F.cosine_similarity(x_recon, x, dim=1)
        cos_loss = torch.mean(1 - cos_sim)

        return cos_loss

def train_model(dataset, model, optimizer, loss_fn, domain_pair, epochs=500, batch_size=128):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    total_loss_list = []

    for epoch in range(epochs):
        total_loss = 0.0
        model.train()
        for item_feature in dataloader:
            optimizer.zero_grad()
            x_recon, _ = model(item_feature)
            loss = loss_fn(x_recon, item_feature)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        total_loss_list.append(total_loss/len(dataloader))
        if (epoch+1)%10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader)}")

    plt.figure(figsize=(8,6))
    plt.plot(list(range(epochs)), total_loss_list, label='Train Loss', color='b', linestyle='-')

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("AutoEncoder Training Loss Curve")
    plt.legend()
    plt.savefig("autoencodertraining_" + domain_pair[0] + "_" + domain_pair[1] + ".png")

    return model

def train_autoencoder(sharedfeatures, domain_pair):
    dataset = SharedDataset(sharedfeatures)
    model = SharedAutoEncoder(feature_dim, hidden_dim, target_dim).to(device)
    loss_fn = ReconstructLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0002)
    model_path = "./model_save/sharedautoencoder_" + domain_pair[0] + "_" + domain_pair[1] + '.pth'
    if not os.path.exists(model_path) or overwrite:
        model = train_model(dataset, model, optimizer, loss_fn, domain_pair, epochs=100, batch_size=128)
        torch.save(model.state_dict(), model_path)
    else:
        model.load_state_dict(torch.load(model_path))

    return model