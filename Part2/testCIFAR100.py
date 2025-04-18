# libraries
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import datasets, transforms, models
from torchvision.transforms import ToTensor
import time
import sys

## download dataset ##

transform = transforms.Compose([
    transforms.Resize(224),  # Required for ResNet input
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

training_data = datasets.CIFAR100(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.CIFAR100(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

## subset of dataset ##

gpus = int(sys.argv[1]) if len(sys.argv) > 1 else -1
device_ids = list(range(gpus))

trainloader = torch.utils.data.DataLoader(training_data, batch_size=256)
testloader = torch.utils.data.DataLoader(test_data, batch_size=256)

device = (
    "cuda:0" if torch.cuda.is_available() and gpus >= 0 else
    "cpu"
)

print(f"Using device: {device}")

## set up architecture ##

input_size = 784
hidden_sizes = [128, 64]
output_size = 10

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 100)

if gpus > 1:
    model = nn.DataParallel(model, device_ids=device_ids)

model.to(device)

## train the model ##

# Define the loss
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

epochs = 5
for e in range(epochs):
    running_loss = 0
    model.train()
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Training loss: {running_loss/len(trainloader)}")

## test model ##

model.eval()
correct, total = 0, 0
with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Number Of Images Tested = {total}")
print(f"Model Accuracy = {correct / total:.4f}")
