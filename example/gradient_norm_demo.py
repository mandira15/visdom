

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from visdom import Visdom


# initialize Visdom
vis = Visdom()

print("Connected to Visdom")


# CNN model
class SimpleCNN(nn.Module):

    def __init__(self):

        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)

        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(16*26*26, 10)


    def forward(self, x):

        x = self.conv1(x)

        x = self.relu(x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)

        return x


# load MNIST dataset
transform = transforms.ToTensor()

trainset = torchvision.datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=32,
    shuffle=True
)


# initialize model
model = SimpleCNN()

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=0.01)


# training loop
step = 0

for epoch in range(1):

    print("Epoch started")

    for images, labels in trainloader:

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        # THIS CALLS YOUR VISDOM FUNCTION
        grad_norm = vis.log_gradient_norm(model, step)

        optimizer.step()

        print(f"Step {step}, Loss: {loss.item():.4f}, GradNorm: {grad_norm:.4f}")

        step += 1

        if step >= 200:
            break

    break

print("Training complete")