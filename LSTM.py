import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from scipy import stats
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# Load data (paths may need to be adjusted)
stock_prices = pd.read_csv("../input/jpx-tokyo-stock-exchange-prediction/train_files/stock_prices.csv")
secondary_stock_prices = pd.read_csv("../input/jpx-tokyo-stock-exchange-prediction/train_files/secondary_stock_prices.csv")
supplemental_prices = pd.read_csv("../input/jpx-tokyo-stock-exchange-prediction/supplemental_files/stock_prices.csv")
supplemental_secondary_stock_prices = pd.read_csv("../input/jpx-tokyo-stock-exchange-prediction/supplemental_files/secondary_stock_prices.csv")

# Merge datasets
stock_prices = pd.concat([
    stock_prices, 
    secondary_stock_prices,
    supplemental_prices,
    supplemental_secondary_stock_prices
])

# Feature engineering function (includes normalization)
def featuring(data):
    data['ExpectedDividend'] = data['ExpectedDividend'].fillna(0)
    data["SupervisionFlag"] = data["SupervisionFlag"].astype(int)
    data['Target'] = data['Target'].fillna(0)
    
    cols = ['Open', 'High', 'Low', 'Close']
    # Interpolate missing price values linearly
    data.loc[:, cols] = data.loc[:, cols].interpolate(method='linear')
    # Add daily range feature
    data['Daily_Range'] = data['Close'] - data['Open']
    
    # Normalize selected columns with z-score
    for col in ['Open', 'High', 'Low', 'Close', 'Volume', 'Daily_Range']:
        data[col] = stats.zscore(data[col])
    
    # Drop non-feature columns
    return data.drop(['RowId', 'Date'], axis=1)

# Preprocess data
data = featuring(stock_prices)
X_data = data.drop(['Target'], axis=1)
y_data = data['Target']

# Split dataset into train/validation/test
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X_data, y_data, test_size=0.2, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.25, random_state=42
)

# Define LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        # x: [batch, seq_len, input_size]
        out, _ = self.lstm(x)
        # Take the last time step
        out = out[:, -1, :]
        out = self.dropout(out)
        return self.fc(out)

# Sequence creation function
def create_sequences(X, y, seq_length):
    sequences, targets = [], []
    for i in range(len(X) - seq_length):
        sequences.append(X[i:i+seq_length])
        targets.append(y[i+seq_length])
    # Convert to numpy then to torch tensors
    return (
        torch.tensor(np.array(sequences), dtype=torch.float32),
        torch.tensor(np.array(targets), dtype=torch.float32)
    )

# Training settings
seq_length = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prepare sequences for train, validation, and test
X_train_seq, y_train_seq = create_sequences(X_train.values, y_train.values, seq_length)
X_val_seq, y_val_seq     = create_sequences(X_val.values, y_val.values, seq_length)
X_test_seq, y_test_seq   = create_sequences(X_test.values, y_test.values, seq_length)

# Move data to device
X_train_seq, y_train_seq = X_train_seq.to(device), y_train_seq.to(device)
X_val_seq,   y_val_seq   = X_val_seq.to(device),   y_val_seq.to(device)
X_test_seq,  y_test_seq  = X_test_seq.to(device),  y_test_seq.to(device)

# Create DataLoaders
train_loader = DataLoader(TensorDataset(X_train_seq, y_train_seq), batch_size=1024, shuffle=False)
val_loader   = DataLoader(TensorDataset(X_val_seq,   y_val_seq),   batch_size=1024, shuffle=False)
test_loader  = DataLoader(TensorDataset(X_test_seq,  y_test_seq),  batch_size=1024, shuffle=False)

# Initialize model, loss, and optimizer
model = LSTMModel(
    input_size=X_train.shape[1],
    hidden_size=64,
    num_layers=2
).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
epochs = 3

train_losses, val_losses = [], []

# Training loop
for epoch in range(epochs):
    model.train()
    running_train_loss = 0.0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch.view(-1, 1))
        loss.backward()
        optimizer.step()
        running_train_loss += loss.item()
    avg_train_loss = running_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    
    # Validation step
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            y_pred = model(X_batch)
            running_val_loss += criterion(y_pred, y_batch.view(-1, 1)).item()
    avg_val_loss = running_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    
    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

# Test evaluation
model.eval()
running_test_loss = 0.0
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        y_pred = model(X_batch)
        running_test_loss += criterion(y_pred, y_batch.view(-1, 1)).item()
avg_test_loss = running_test_loss / len(test_loader)
print(f"Final Test Loss: {avg_test_loss:.4f}")

# Plot training and validation loss curves
plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs+1), train_losses, label='Train Loss')
plt.plot(range(1, epochs+1), val_losses,   label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss over Epochs')
plt.legend()
plt.grid(True)
plt.show()
