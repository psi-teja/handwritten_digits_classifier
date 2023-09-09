import torch
import torch.nn as nn
import torch.optim as optim

# Define the model class
class TorchModel(nn.Module):
    """TorchModel is built using CNN architecture.

    Args:
        None

    Attributes:
        conv1 (nn.Conv2d): The first convolutional layer.
        pool (nn.MaxPool2d): The max-pooling layer.
        conv2 (nn.Conv2d): The second convolutional layer.
        fc1 (nn.Linear): The first fully connected layer.
        fc2 (nn.Linear): The second fully connected layer.
    """
    def __init__(self):
        super(TorchModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 28, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(28, 64, kernel_size=3)
        self.fc1 = nn.Linear(64 * 5 * 5, 64)  # Flattened size after pooling
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):       
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)  # Flatten the feature maps
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    # Create an instance of the model
    model = TorchModel()

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Display the model architecture
    print(model)
