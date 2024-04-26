from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision.transforms import ToTensor
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DATA_PATH = "E:\\Data\\Datasets\\Parkinson_Speech\\26-29_09_2017_KCL\\SpontaneousDialogue\\data_10.json"


def load_data(data_path):
    """Loads training dataset from json file.

        :param data_path (str): Path to json file containing data
        :return X (ndarray): Inputs
        :return y (ndarray): Targets
    """

    with open(data_path, "r") as fp:
        data = json.load(fp)

    # convert lists to numpy arrays
    X = np.array(data["mfcc"])
    y = np.array(data["labels"])

    print("Data succesfully loaded!")

    return X, y


def prepare_datasets(test_size, validation_size):
    X, y = load_data(DATA_PATH)
    X = torch.from_numpy(X).float().unsqueeze(1)  # Convert X to tensor and add channel dimension
    y = torch.from_numpy(y).long()
    dataset = TensorDataset(X, y)

    test_len = int(len(dataset) * test_size)
    valid_len = int(len(dataset) * validation_size)
    train_len = len(dataset) - test_len - valid_len

    train_data, valid_data, test_data = random_split(dataset, [train_len, valid_len, test_len])
    return X, train_data, valid_data, test_data


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Assuming input_shape is (batch_size, 1, 130, 13) for PyTorch
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3))
        self.pool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3, 3))
        self.pool2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 32, kernel_size=(2, 2))
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0)
        self.bn3 = nn.BatchNorm2d(32)

        # Flatten layer is implicit in the forward method
        self.fc1 = nn.Linear(32 * 17 * 2, 64)  # Adjust the linear layer input size
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = torch.flatten(x, 1)  # Flatten all dimensions except the batch
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def train(model, data_loader, optimizer, criterion):
    model.train()
    for inputs, labels in data_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


def validate(model, data_loader, criterion, device='cpu'):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():  # No gradients needed
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)  # Get the index of the max log-probability
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / len(data_loader)
    accuracy = total_correct / total_samples * 100  # Calculate the accuracy in percentage
    print(f'Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
    return avg_loss, accuracy


def evaluate(model, data_loader, criterion, device='cpu'):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():  # No gradients needed
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)  # Get the index of the max log-probability
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / len(data_loader)
    accuracy = total_correct / total_samples * 100  # Calculate the accuracy in percentage
    print(f'Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
    return avg_loss, accuracy

if __name__ == '__main__':
    X,train_data,valid_data,test_data =  prepare_datasets(0.3,0.2)

    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=32)
    test_loader = DataLoader(test_data, batch_size=32)

    # Set up the model, optimizer, and loss function
    model = Net()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(100):
        train(model, train_loader, optimizer, criterion)
        validate(model, valid_loader, criterion)  # Define a validation function as needed

    # Test the model
    test_accuracy = evaluate(model, test_loader)  # Define an evaluate function