import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from scipy import stats
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# load data
stock_prices = pd.read_csv("../input/jpx-tokyo-stock-exchange-prediction/train_files/stock_prices.csv")
secondary_stock_prices = pd.read_csv("../input/jpx-tokyo-stock-exchange-prediction/train_files/secondary_stock_prices.csv")
supplemental_prices = pd.read_csv("../input/jpx-tokyo-stock-exchange-prediction/supplemental_files/stock_prices.csv")
supplemental_secondary_stock_prices = pd.read_csv("../input/jpx-tokyo-stock-exchange-prediction/supplemental_files/secondary_stock_prices.csv")

stock_prices = pd.concat([
    stock_prices, 
    secondary_stock_prices,
    supplemental_prices,
    supplemental_secondary_stock_prices
])

# featuring (includes normalization)
def featuring(data):
    data['ExpectedDividend'] = data['ExpectedDividend'].fillna(0)
    data["SupervisionFlag"] = data["SupervisionFlag"].astype(int)
    data['Target'] = data['Target'].fillna(0)
    
    cols = ['Open', 'High', 'Low', 'Close']
    data.loc[:,cols] = data.loc[:,cols].interpolate(method='linear')
    data['Daily_Range'] = data['Close'] - data['Open']
    
    # 标准化处理
    for col in ['Open', 'High', 'Low', 'Close', 'Volume', 'Daily_Range']:
        data[col] = stats.zscore(data[col])
    
    return data.drop(['RowId', 'Date'], axis=1)

# pre-process data
data = featuring(stock_prices)
X_data = data.drop(['Target'], axis=1)
y_data = data['Target']

# split dataset
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X_data, y_data, test_size=0.2, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.25, random_state=42
)

# LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # 取最后一个时间步
        out = self.dropout(out)
        return self.fc(out)

def create_sequences(X, y, seq_length):
    sequences = []
    targets = []
    for i in range(len(X) - seq_length):
        sequences.append(X[i:i+seq_length])
        targets.append(y[i+seq_length])
    sequences_np = np.array(sequences)  # 转换为numpy数组
    targets_np = np.array(targets)
    return torch.tensor(sequences_np, dtype=torch.float32), torch.tensor(targets_np, dtype=torch.float32)

seq_length = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# transfer data
X_train_seq, y_train_seq = create_sequences(X_train.values, y_train.values, seq_length)
X_val_seq, y_val_seq = create_sequences(X_val.values, y_val.values, seq_length)
X_test_seq, y_test_seq = create_sequences(X_test.values, y_test.values, seq_length)


X_train_seq, y_train_seq = X_train_seq.to(device), y_train_seq.to(device)
X_val_seq, y_val_seq = X_val_seq.to(device), y_val_seq.to(device)
X_test_seq, y_test_seq = X_test_seq.to(device), y_test_seq.to(device)

# DataLoader
train_dataset = TensorDataset(X_train_seq, y_train_seq)
val_dataset = TensorDataset(X_val_seq, y_val_seq)
test_dataset = TensorDataset(X_test_seq, y_test_seq)

train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)


model = LSTMModel(
    input_size=X_train.shape[1], 
    hidden_size=64, 
    num_layers=2
).to(device)


criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
epochs = 3

train_losses = []  # 初始化训练损失列表
val_losses = []    # 初始化验证损失列表

for epoch in range(epochs):
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch.view(-1, 1))
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    train_losses.append(train_loss)  # 记录训练损失
    
    # 验证阶段
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            y_pred = model(X_batch)
            val_loss += criterion(y_pred, y_batch.view(-1, 1)).item()
    val_loss /= len(val_loader)
    val_losses.append(val_loss)  # 记录验证损失
    
    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

# 测试评估
model.eval()
test_loss = 0
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        y_pred = model(X_batch)
        test_loss += criterion(y_pred, y_batch.view(-1, 1)).item()
test_loss /= len(test_loader)
print(f"Final Test Loss: {test_loss:.4f}")

# 绘制损失曲线（使用正确变量名）
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.show()
