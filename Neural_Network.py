# import necessary libraries
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from vecstack import StackingTransformer

# set device to GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# define a custom dataset for PyTorch
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        super().__init__()
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)

# define the architecture of the neural network
class Network(torch.nn.Module):
    def __init__(self, init_size):
        super().__init__()
        # define layers of the network
        self.layers = torch.nn.ModuleList()
        self.layers.append(torch.nn.Linear(init_size, 1024))
        self.layers.append(torch.nn.Linear(1024, 512))
        self.layers.append(torch.nn.Linear(512, 1024))
        self.layers.append(torch.nn.Linear(1024, 256))
        self.layers.append(torch.nn.Linear(256, 2))
        self.layers.apply(self._init_wt)

    def _init_wt(self, layer):
        # initialize weights
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.kaiming_normal_(layer.weight)

    def forward(self, x):
        # define forward pass
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            if i < len(self.layers)-1:
                x = torch.nn.ReLU()(x)
        return x

# load training and test data
train_df = pd.read_csv('upload the train set embeddings obtained from bert.csv', encoding='utf-8')
test_df = pd.read_csv('/upload the test set embeddings obtained from bert', encoding='utf-8')

# separate features and labels
X_train = train_df.drop('label', axis=1).values
Y_train = train_df['label'].values
X_test = test_df.drop('label', axis=1).values
Y_test = test_df['label'].values

print("train")
print(X_train.shape, Y_train.shape)
print("test")
print(X_test.shape, Y_test.shape)

# scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# define training parameters
epochs = 100

# define stratified k-fold cross validation
kfold = StratifiedKFold(n_splits=5, shuffle=True)

# train model using cross validation
for idx, (train, val) in enumerate(kfold.split(X_train, Y_train)):
    # split data into training and validation sets
    x_train_fold, x_val_fold = X_train[train], X_train[val]
    y_train_fold, y_val_fold = Y_train[train], Y_train[val]

    # initialize model and dataloaders
    model_dl = Network(x_train_fold.shape[1]).to(device)
    train_dataloader = DataLoader(MyDataset(x_train_fold, y_train_fold), batch_size=32, shuffle=True)
    val_dataloader = DataLoader(MyDataset(x_val_fold, y_val_fold), batch_size=512, shuffle=False)

    # define optimizer and loss function
    optimizer = torch.optim.Adam(model_dl.parameters(), lr=0.0001)
    criterion = torch.nn.CrossEntropyLoss()

    # train model for each epoch
    for epoch in range(epochs):
        losses = list()
        model_dl.train()
        for batch_idx, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            x, y = batch
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            preds = model_dl(x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        avg_loss = sum(losses)/len(losses)
        print(f'Epoch: {epoch+1}, Average Loss: {avg_loss}')

# evaluate model on the test set
test_dataloader = DataLoader(MyDataset(X_test, Y_test), batch_size=256, shuffle=False)
model_dl.eval()
true_labels, pred_labels = list(), list()
with torch.no_grad():
    for batch_idx, batch in tqdm(enumerate(test_dataloader)):
        x, y = batch
        x = x.to(device)
        preds = model_dl(x)
        preds = list(np.argmax(preds.cpu().detach().numpy(), axis=1))
        true_labels.extend(y.numpy())
        pred_labels.extend(preds)

# print evaluation metrics
print("Test Accuracy: ", accuracy_score(true_labels, pred_labels))
print("Test Precision: ", precision_score(true_labels, pred_labels))
print("Test Recall: ", recall_score(true_labels, pred_labels))
print("Test F1 Score: ", f1_score(true_labels, pred_labels))
