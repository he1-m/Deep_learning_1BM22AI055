import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
torch.manual_seed(0)


X = np.random.randn(1000, 20)
y_class = np.random.randint(0, 2, size=(1000, 1))  
y_reg = np.random.randn(1000, 1)  

X_train, X_val, y_class_train, y_class_val, y_reg_train, y_reg_val = train_test_split(
    X, y_class, y_reg, test_size=0.2, random_state=42
)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_class_train = torch.tensor(y_class_train, dtype=torch.long)  
y_class_val = torch.tensor(y_class_val, dtype=torch.long)
y_reg_train = torch.tensor(y_reg_train, dtype=torch.float32)
y_reg_val = torch.tensor(y_reg_val, dtype=torch.float32)

train_data = TensorDataset(X_train, y_class_train, y_reg_train)
val_data = TensorDataset(X_val, y_class_val, y_reg_val)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

class MultitaskNN(nn.Module):
    def __init__(self):
        super(MultitaskNN, self).__init__()
        self.fc1 = nn.Linear(20, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc_class = nn.Linear(32, 2) 
        self.fc_reg = nn.Linear(32, 1)   

        self.dropout = nn.Dropout(p=0.5) 

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(x) 

        class_out = self.fc_class(x) 
        reg_out = self.fc_reg(x)     

        return class_out, reg_out

model = MultitaskNN()
criterion_class = nn.CrossEntropyLoss() 
criterion_reg = nn.MSELoss()             
optimizer = optim.Adam(model.parameters(), lr=0.001)

def clip_gradients(model, max_norm=1.0):
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

early_stopping_patience = 5 
best_val_loss = float('inf')
patience_counter = 0

num_epochs = 100
train_loss_history = []
val_loss_history = []

for epoch in range(num_epochs):
    model.train()
    running_loss_class = 0.0
    running_loss_reg = 0.0

    for X_batch, y_class_batch, y_reg_batch in train_loader:
        optimizer.zero_grad()

  
        class_out, reg_out = model(X_batch)

  
        loss_class = criterion_class(class_out, y_class_batch.squeeze())
        loss_reg = criterion_reg(reg_out.squeeze(), y_reg_batch.squeeze())

  
        total_loss = loss_class + loss_reg

  
        total_loss.backward()

  
        clip_gradients(model, max_norm=1.0)

  
        optimizer.step()

  
        running_loss_class += loss_class.item()
        running_loss_reg += loss_reg.item()

    avg_train_loss_class = running_loss_class / len(train_loader)
    avg_train_loss_reg = running_loss_reg / len(train_loader)
    train_loss_history.append((avg_train_loss_class, avg_train_loss_reg))

    model.eval()
    val_loss_class = 0.0
    val_loss_reg = 0.0
    with torch.no_grad():
        for X_batch, y_class_batch, y_reg_batch in val_loader:
            class_out, reg_out = model(X_batch)

            loss_class = criterion_class(class_out, y_class_batch.squeeze())
            loss_reg = criterion_reg(reg_out.squeeze(), y_reg_batch.squeeze())

            val_loss_class += loss_class.item()
            val_loss_reg += loss_reg.item()

    avg_val_loss_class = val_loss_class / len(val_loader)
    avg_val_loss_reg = val_loss_reg / len(val_loader)
    val_loss_history.append((avg_val_loss_class, avg_val_loss_reg))

    print(f"Epoch [{epoch+1}/{num_epochs}] - "
          f"Train Loss (Class: {avg_train_loss_class:.4f}, Reg: {avg_train_loss_reg:.4f}) - "
          f"Val Loss (Class: {avg_val_loss_class:.4f}, Reg: {avg_val_loss_reg:.4f})")

    val_loss = avg_val_loss_class + avg_val_loss_reg  
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0  
    else:
        patience_counter += 1
        if patience_counter >= early_stopping_patience:
            print("Early stopping triggered!")
            break

train_loss_class_vals, train_loss_reg_vals = zip(*train_loss_history)
val_loss_class_vals, val_loss_reg_vals = zip(*val_loss_history)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(train_loss_class_vals, label="Train Classification Loss")
plt.plot(train_loss_reg_vals, label="Train Regression Loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Loss')

plt.subplot(1, 2, 2)
plt.plot(val_loss_class_vals, label="Val Classification Loss")
plt.plot(val_loss_reg_vals, label="Val Regression Loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Validation Loss')

plt.tight_layout()
plt.show()
