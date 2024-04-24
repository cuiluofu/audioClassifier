import json
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import tensorflow.keras as keras

DATA_PATH = "E:\\Datasets\\Parkinson_Speech\\26-29_09_2017_KCL\\SpontaneousDialogue\\data.json"


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


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.dense_layers = nn.Sequential(
            nn.Linear(X_train.shape[1] * X_train.shape[2], 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.flatten(x)
        logits = self.dense_layers(x)
        probabilities = self.softmax(logits)
        return probabilities




if __name__ == "__main__":
    X, y = load_data(DATA_PATH)
    X = torch.from_numpy(X).float()  # Convert to torch.Tensor and ensure type is float
    y = torch.from_numpy(y).long()  # Convert to torch.Tensor and ensure type is long
    # create train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # Convert numpy arrays to TensorDatasets and then to DataLoaders
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
 # build network topology
    model = keras.Sequential([

        # input layer
        keras.layers.Flatten(input_shape=(X.shape[1], X.shape[2])),

        # 1st dense layer
        keras.layers.Dense(512, activation='relu'),

        # 2nd dense layer
        keras.layers.Dense(256, activation='relu'),

        # 3rd dense layer
        keras.layers.Dense(64, activation='relu'),

        # output layer
        keras.layers.Dense(10, activation='softmax')
    ])

    # compile model
    optimiser = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimiser,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    # train model
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=32, epochs=50)
#     model = NeuralNetwork()
#
#     # Define optimizer and loss function
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
#     loss_function = nn.CrossEntropyLoss()
#
#
#     def train(dataloader, model, loss_fn, optimizer):
#         size = len(dataloader.dataset)
#         model.train()
#         for batch, (X, y) in enumerate(dataloader):
#             X, y = X.to(device), y.to(device)
#             pred = model(X)
#             loss = loss_fn(pred, y)
#
#             # Backpropagation
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             if batch % 100 == 0:
#                 loss, current = loss.item(), batch * len(X)
#                 print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
#
#
#     # Function to perform testing
#     def test(dataloader, model, loss_fn):
#         size = len(dataloader.dataset)
#         num_batches = len(dataloader)
#         model.eval()
#         test_loss, correct = 0, 0
#         with torch.no_grad():
#             for X, y in dataloader:
#                 X, y = X.to(device), y.to(device)
#                 pred = model(X)
#                 test_loss += loss_fn(pred, y).item()
#                 correct += (pred.argmax(1) == y).type(torch.float).sum().item()
#         test_loss /= num_batches
#         correct /= size
#         print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
#
#
#     # Set the device to GPU if available
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model.to(device)
# # Training the model
#     epochs = 50
#     for t in range(epochs):
#         print(f"Epoch {t+1}\n-------------------------------")
#         train(train_loader, model, loss_function, optimizer)
#         test(test_loader, model, loss_function)
#     print("Done!")