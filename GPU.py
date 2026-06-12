# %% [markdown]
# ## Work with GPU

# %%
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

# %%
torch.cuda.is_available()

# %%
import torch

print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device being used:',device)

# %%
device

# %%
df = pd.read_csv("fashion-mnist_train.csv")
df.shape

# %%
df.head()

# %%
features = df.iloc[:,1:]
targets = df.iloc[:,0]
targets

# %%
x_train, x_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state= 42)
x_train /= 255.0
x_test /= 255.0

# %%
x_train.shape

# %%
x_test.shape

# %%
device

# %%
# to create the dataset
class CustomDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.tensor(features.to_numpy()).to(torch.float32)
        self.targets = torch.tensor(targets.to_numpy()).to(torch.long)
    
    def __len__(self):
        return self.features.shape[0]
    
    def __getitem__(self, index):
        return (self.features[index], self.targets[index])

# %%
# dataset collection stored here
train_dataset = CustomDataset(x_train, y_train)
test_dataset = CustomDataset(x_test, y_test)

# %%
train_dataset.__len__()

# %%
test_dataset.__len__()

# %%
train_dataset.__getitem__(0)

# %%
train_dataset_loader = DataLoader(train_dataset, batch_size= 32, shuffle= True, pin_memory=True)
test_dataset_loader = DataLoader(test_dataset, batch_size= 32, shuffle=False, pin_memory=True)

# %%
for feature, target in train_dataset_loader:
    print(feature)
    print(target)
    break

# %%
class ImageClassifier(nn.Module):

    def __init__(self, number_of_features):
        super().__init__()

        # Image classifier model
        self.model = nn.Sequential(
            nn.Linear(number_of_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            nn.Dropout(p = 0.4),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p = 0.4),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(p = 0.4),
            nn.Linear(32, 10)
        )

        # optimizer
        self.optim = optim.SGD(self.model.parameters(), lr = 0.1, weight_decay= 1e-4)
    
    def forward(self, features):
        self.y_pred = self.model(features)
        return self.y_pred
    
    def loss_function(self, y_real):
        loss_fn = nn.CrossEntropyLoss()
        self.loss = loss_fn(self.y_pred, y_real)
        return self.loss.item()
    
    def optimization(self):
        self.optim.zero_grad()
        self.loss.backward()
        self.optim.step()


# %%
x_train.shape[1]

# %%
# model is defined but the model uses the "cpu"
model = ImageClassifier(x_train.shape[1])

# %%
for i in model.named_parameters():
    print(i)

# %%
device

# %%
# making model use gpu instead of cpu
model = model.to(device)

# %%
model

# %%
epochs = 30
total_loss = 0
for epoch in range(epochs):
    epoch += 1
    total_loss = 0
    for features_batch, targets_batch in train_dataset_loader:
        features_batch = features_batch.to(device)
        targets_batch = targets_batch.to(device)
        y_pred = model(features_batch)
        loss = model.loss_function(targets_batch)
        model.optimization()
        total_loss = total_loss + loss

    print("epoch:", epoch, "\tloss:", total_loss/len(train_dataset_loader))

# %%
# model ready for evaluation
model.eval()

# %% [markdown]
# ### Accuracy:
# We need to find the accurcy now. The accuracy of the data is given by the formula:
# 
# Accuracy = (number of correct predictions) / (total number of the predictions)

# %%
len(test_dataset_loader)*32

# %%
correct = 0
data_records = 0
with torch.no_grad():
    for features_batch, targets_batch in test_dataset_loader:
        features_batch = features_batch.to(device)
        targets_batch = targets_batch.to(device)
        y_pred = model(features_batch)
        correct = correct + (targets_batch == torch.max(y_pred, dim = 1)[1]).sum().item()
        data_records = data_records + features_batch.shape[0]
    print("total data_records:", data_records, "correct records:", correct)
    accuracy = correct / data_records
    print(accuracy)

# %%
correct = 0
data_records = 0
with torch.no_grad():
    for features_batch, targets_batch in train_dataset_loader:
        features_batch = features_batch.to(device)
        targets_batch = targets_batch.to(device)
        y_pred = model(features_batch)
        correct = correct + (targets_batch == torch.max(y_pred, dim = 1)[1]).sum().item()
        data_records = data_records + features_batch.shape[0]
    print("total data_records:", data_records, "correct records:", correct)
    accuracy = correct / data_records
    print(accuracy)


