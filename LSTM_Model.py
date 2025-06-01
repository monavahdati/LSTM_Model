import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, recall_score, precision_score, 
                             f1_score, confusion_matrix, roc_auc_score, 
                             precision_recall_curve, roc_curve, auc)
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import shap

# Load data
data = pd.read_csv('bnpl_credit_data_lighte.csv')

# Preprocess the data
data.fillna(0, inplace=True)
data['Age Condition'] = np.where(data['Age'] < 18, 0, 1)
data['Credit_Condition'] = np.where(data['Credit Score'] > 519, 1, 0)

# Define purchase columns
purchase_freq_cols = ['Monthly Purchase Frequency 1', 'Monthly Purchase Frequency 2',
                       'Monthly Purchase Frequency 3', 'Monthly Purchase Frequency 4',
                       'Monthly Purchase Frequency 5', 'Monthly Purchase Frequency 6']
purchase_amount_cols = ['Monthly Purchase Amount 1', 'Monthly Purchase Amount 2',
                         'Monthly Purchase Amount 3', 'Monthly Purchase Amount 4',
                         'Monthly Purchase Amount 5', 'Monthly Purchase Amount 6']

data['Total_Purchase_Frequency'] = data[purchase_freq_cols].sum(axis=1)
data['Total_Purchase_Amount'] = data[purchase_amount_cols].sum(axis=1)
data['Repeat Usage'] = data['Repeat Usage'].map({'Yes': 1, 'No': 0})

# Create credit amount and repayment period based on conditions
def determine_credit(row):
    # Your logic here...
    return 0, 0  # Default no credit

data[['Credit Amount', 'Repayment Period']] = data.apply(determine_credit, axis=1, result_type='expand')

# Define target variable
data['Target'] = np.where(data['Credit_Condition'] & (data['Total_Purchase_Amount'] > 10), 1, 0)

# Prepare features and target
features = data[['Age', 'Credit Score', 'Total_Purchase_Frequency', 
                  'Total_Purchase_Amount', 'Age Condition', 'Rating', 'Repeat Usage']]
target = data['Target']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

# Normalize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert to tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# Create DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

# Define LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x.unsqueeze(1), (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# Model parameters
input_size = 7
hidden_size = 128
num_layers = 4
output_size = 1
dropout = 0.2

# Create the model
model = LSTMModel(input_size, hidden_size, num_layers, output_size, dropout)

# Define optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.BCEWithLogitsLoss()

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)
        predictions = torch.sigmoid(outputs).round()
        total_correct += (predictions == y_batch).sum().item()
        total_samples += y_batch.size(0)

    # Print training progress
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / total_samples:.4f}, Accuracy: {total_correct / total_samples:.4f}')

# Save the model
torch.save(model.state_dict(), 'lstm_model.pth')
