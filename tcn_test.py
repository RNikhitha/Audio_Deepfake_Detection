import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_curve
from pytorch_tcn import TCN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    

# Load model and weights
tcn_mod = AudioTCN(input_channels=3, num_classes=2, num_channels=[128, 128, 128])
tcn_mod.load_state_dict(torch.load("tcn_model_weights.pth", map_location=device))
tcn_mod = tcn_mod.to(device)
tcn_mod.eval()

# test data
X_test = np.load("X_test.npy")  
Y_test = np.load("y_test.npy")  
# reshaping the X_test 
X_test = torch.from_numpy(X_test).float().to(device)
X_test = np.transpose(X_test, (0, 3, 1, 2))
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], -1) 

# eval
with torch.no_grad():
    logits = tcn_mod(X_test)
    probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

# EER
fpr, tpr, _ = roc_curve(Y_test, probs)
fnr = 1 - tpr
eer = fpr[np.nanargmin(np.abs(fnr - fpr))]
print(f"Equal Error Rate (EER) for TCN: {eer:.4f}")
