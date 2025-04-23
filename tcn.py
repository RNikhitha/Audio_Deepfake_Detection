import torch
from torch import nn
from pytorch_tcn import TCN
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from sklearn.metrics import roc_curve

x = np.load('X_eval.npy')
y = np.load('y_eval.npy')

x = np.transpose(x, (0, 3, 1, 2))
x = x.reshape(x.shape[0], x.shape[1], -1)

x_t = torch.tensor(x, dtype=torch.float32)
y_t = torch.tensor(y, dtype=torch.long)

dataset = TensorDataset(x_t, y_t)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# data loaders
batch = 32
train_loader = DataLoader(train_dataset, batch=batch)
test_loader = DataLoader(test_dataset, batch=batch)

# model
class AudioTCN(nn.Module):
    def __init__(self, input_channels, num_classes, num_channels, kernel_size=3, dropout=0.2):
        super(AudioTCN, self).__init__()
        self.tcn = TCN(input_channels,num_channels,kernel_size=kernel_size,dropout=dropout)
        self.classifier = nn.Linear(num_channels[-1], num_classes)

    def forward(self, x):
        y = self.tcn(x)          
        y=torch.mean(y, dim=2)
        return self.classifier(y)


input = x.shape[1]  
output = 2
num_channels = [128, 128, 128]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AudioTCN(input, output, num_channels).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(5):
    model.train()
    running_loss = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader):.4f}")

#torch.save(model.state_dict(), "tcn_model_weights.pth")
model.eval()

all_labels = []
all_probs = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1)[:, 1]  
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.numpy())

fpr, tpr, _ = roc_curve(all_labels, all_probs)
fnr = 1 - tpr
eer = fpr[np.nanargmin(np.absolute(fnr - fpr))]
print(f"Equal Error Rate (EER): {eer:.4f}")
