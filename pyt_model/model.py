import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define the model class
class TorchModel(nn.Module):
    def __init__(self):
        super(TorchModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=3136, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # Convolutional Layer 1
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)

        # Convolutional Layer 2
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)

        # Flatten the tensor
        x = x.view(-1, 3136)

        # Fully Connected Layer 1
        x = torch.relu(self.fc1(x))

        # Fully Connected Layer 2
        x = self.fc2(x)

        # Apply softmax to get class probabilities
        x = torch.softmax(x, dim=1)

        return x
    
    def predict(self,x):
        x = x.reshape(1,28,28)
        x = torch.from_numpy(x.astype(np.float32))
        x = self.forward(x)
        return x.detach().numpy()



if __name__ == "__main__":
    # Create an instance of the model
    model = TorchModel()

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Display the model architecture
    print(model)
