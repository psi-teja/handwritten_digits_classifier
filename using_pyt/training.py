import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from model import TorchModel


transform = transforms.Compose([
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Lambda(lambda x: (255 - x) / 255),  # normalizing pixel values to [0,1]
    transforms.Lambda(lambda x: x * (x >= 1))    # Set values < 1 to 0
])


# Define hyperparameters
learning_rate = 0.001
batch_size = 64
epochs = 10

trainset = torchvision.datasets.MNIST(root='using_pyt',train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)


# Initialize the model and optimizer
model = TorchModel()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(epochs):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backpropagation and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {running_loss / (i + 1)}")

print("Training finished")

# Save the trained model if needed
torch.save(model.state_dict(), "using_pyt/mnist_model.pth")