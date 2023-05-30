import seaborn as sns
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

# 30 Hz data, sample_len sequence (sample_len/30 seconds)
sample_len = 90
input_channels_count = 9
labels = {0: 'null', 1: 'freestyle', 2: 'breaststroke', 3: 'backstroke', 4: 'butterfly', 5: 'turn'}

# Define model parameters
input_size = sample_len * input_channels_count  # how many features we initially feed the model
learning_rate = 0.0005
num_classes = len(labels)  # The output is prediction results

SEED = 1234

model_path = "NetModel_seq90.pth"


def set_seeds(seed=1234):
    """Set seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


# Set seeds for reproducibility
set_seeds(seed=SEED)

# Set device
cuda = True
device = torch.device("cuda" if (
        torch.cuda.is_available() and cuda) else "cpu")
torch.set_default_tensor_type("torch.FloatTensor")
if device.type == "cuda":
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
print(device)


# if not os.path.exists("/content/SwimStyleRecog"):
#   os.mkdir("/content/SwimStyleRecog")
# os.chdir("/content/SwimStyleRecog")
#
# ! git clone https://github.com/brunnergino/swimming-recognition-lap-counting


def load_data():
    def load_csv_files(directory):
        dfs = []  # List to store individual DataFrames

        # Recursive function to traverse directory and load CSV files
        def recursive_load(directory):
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if file.endswith('.csv'):
                        file_path = os.path.join(root, file)
                        df = pd.read_csv(file_path)
                        dfs.append(df)

        # Call the recursive function
        recursive_load(directory)

        # Merge all DataFrames into a single DataFrame
        merged_df = pd.concat(dfs, ignore_index=True)

        return merged_df

    directory_path = "swimming-recognition-lap-counting/data/processed_30hz_relabeled/"
    # merge all data to one df
    df_raw = load_csv_files(directory_path)
    # remove unnecessary columns
    df = pd.concat([df_raw.iloc[:, 2:11], df_raw.iloc[:, 13]], axis=1)

    sns.countplot(x=df["label"])

    # remove rows with unnecessary labels
    df = df[~df['label'].isin([-1, 6])]

    sns.countplot(x=[labels[i] for i in df["label"]])

    # reshape data by n seconds (sample_len samples)

    new_df_len = len(df) // sample_len * sample_len
    train_len = int((new_df_len * 0.6) // sample_len * sample_len)
    validate_len = int((new_df_len * 0.2) // sample_len * sample_len)
    test_len = int((new_df_len * 0.2) // sample_len * sample_len)
    print(f"{train_len=},{validate_len=},{test_len=}")
    df_train = df.iloc[:train_len, :]
    df_validate = df.iloc[train_len:train_len + validate_len, :]
    df_test = df.iloc[train_len + validate_len:train_len + validate_len + test_len, :]
    df_train.to_pickle("./df_train.pkl")
    df_validate.to_pickle("./df_validate.pkl")
    df_test.to_pickle("./df_test.pkl")


# for first run
if (not os.path.exists("./df_train.pkl") or not os.path.exists("./df_test.pkl")
        or not os.path.exists("./df_validate.pkl")):
    load_data()

df_train = pd.read_pickle("./df_train.pkl")
df_validate = pd.read_pickle("./df_validate.pkl")
df_test = pd.read_pickle("./df_test.pkl")
# df_train["label"].plot()
# plt.show()

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader


class CNNModel(nn.Module):
    def __init__(self, in_ch, num_classes):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_ch, out_channels=59, kernel_size=3, stride=1)
        self.elu1 = nn.ELU()
        self.max_pool1 = nn.MaxPool1d(kernel_size=3, stride=1)
        self.conv2 = nn.Conv1d(59, 19, kernel_size=3, stride=1)
        self.elu2 = nn.ELU()
        self.max_pool2 = nn.MaxPool1d(kernel_size=3, stride=1)
        self.conv3 = nn.Conv1d(19, 5, kernel_size=3, stride=1)
        self.elu3 = nn.ELU()
        self.max_pool3 = nn.MaxPool1d(kernel_size=3, stride=1)
        self.conv4 = nn.Conv1d(5, 1, kernel_size=3, stride=1)
        self.elu4 = nn.ELU()
        self.max_pool4 = nn.MaxPool1d(kernel_size=3, stride=1)
        self.fc = nn.Linear(in_features=sample_len - 16, out_features=128)
        self.elu_fc = nn.ELU()
        self.fc2 = nn.Linear(in_features=128, out_features=num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.elu1(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = self.elu2(x)
        x = self.max_pool2(x)
        x = self.conv3(x)
        x = self.elu3(x)
        x = self.max_pool3(x)
        x = self.conv4(x)
        x = self.elu4(x)
        x = self.max_pool4(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc(x)
        x = self.elu_fc(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


# Define a custom dataset
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        inputs = torch.tensor(self.data[index][:, :-1], dtype=torch.float32)
        # need transpose for [BatchSize, Channels, Sequence] format
        inputs = inputs.transpose(0, 1).contiguous()

        # Most frequent value in the above array
        most_freq_label = np.bincount(np.array(self.data[index][:, -1], dtype=int)).argmax()
        label = torch.tensor(int(most_freq_label), dtype=torch.long)
        return inputs, label

    def __len__(self):
        return len(self.data)


# Load data into a pandas DataFrame (assuming it's named 'df')
# Prepare data by converting it into a format suitable for the custom dataset
train_data = df_train.values.reshape(-1, sample_len, 10)  # Reshape to (num_samples, sequence_length, num_features)
validate_data = df_validate.values.reshape(-1, sample_len, 10)
test_data = df_test.values.reshape(-1, sample_len, 10)

train_dataset = CustomDataset(train_data)
validate_dataset = CustomDataset(validate_data)
test_dataset = CustomDataset(test_data)

# Create data loaders
batch_size = 2*4096 #32
kwargs = {'num_workers': 8, 'pin_memory': True} if device == 'cuda' else {}
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, **kwargs)
validate_loader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize the CNN model
model = CNNModel(in_ch=input_channels_count, num_classes=num_classes)

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))

# Define the loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Training Function
def train(num_epochs, continue_while_accuracy_is_improving=False):
    best_accuracy = 0.0

    print("Begin training...")
    epoch = 1
    prev_accuracy, accuracy = -1.0, 0.0
    while epoch < num_epochs + 1 or (continue_while_accuracy_is_improving and accuracy > prev_accuracy):
        running_train_loss = 0.0
        running_accuracy = 0.0
        running_vall_loss = 0.0
        total = 0

        # Training Loop
        for inputs, outputs in train_loader:
            # forward + backward + optimize
            optimizer.zero_grad()  # zero the parameter gradients
            predicted_outputs = model(inputs)  # forward pass, predict output from the model
            train_loss = loss_fn(predicted_outputs, outputs)  # calculate loss for the predicted output
            train_loss.backward()  # backpropagate the loss
            optimizer.step()  # adjust parameters based on the calculated gradients
            running_train_loss += train_loss.item()  # track the loss value

        # Calculate training loss value
        train_loss_value = running_train_loss / len(train_loader)

        # Validation Loop
        with torch.no_grad():
            model.eval()
            for data in validate_loader:
                inputs, outputs = data
                predicted_outputs = model(inputs)
                val_loss = loss_fn(predicted_outputs, outputs)

                # The label with the highest value will be our prediction
                _, predicted = torch.max(predicted_outputs, 1)
                running_vall_loss += val_loss.item()
                total += outputs.size(0)
                running_accuracy += (predicted == outputs).sum().item()

                # Calculate validation loss value
        val_loss_value = running_vall_loss / len(validate_loader)

        prev_accuracy = accuracy
        # Calculate accuracy as the number of correct predictions in the validation batch divided by the total number of predictions done.
        accuracy = (100 * running_accuracy / total)

        # Save the model if the accuracy is the best
        if accuracy > best_accuracy:
            torch.save(model.state_dict(), model_path)
            best_accuracy = accuracy

            # Print the statistics of the epoch
        print('Completed training batch', epoch, 'Training Loss is: %.4f' % train_loss_value,
              'Validation Loss is: %.4f' % val_loss_value, 'Accuracy is %d %%' % (accuracy))

        epoch += 1


# Function to test the model
def test():
    # Load the model that we saved at the end of the training loop
    model = CNNModel(in_ch=input_channels_count, num_classes=num_classes)
    path = "NetModel_seq90.pth"
    model.load_state_dict(torch.load(path))

    running_accuracy = 0
    total = 0

    with torch.no_grad():
        for data in test_loader:
            inputs, outputs = data
            outputs = outputs.to(torch.float32)
            predicted_outputs = model(inputs)
            _, predicted = torch.max(predicted_outputs, 1)
            total += outputs.size(0)
            running_accuracy += (predicted == outputs).sum().item()

        print('Accuracy of the model based on the test set inputs is: %d %%' % (100 * running_accuracy / total))

    # Optional: Function to test which species were easier to predict


def test_species():
    # Load the model that we saved at the end of the training loop
    model = CNNModel(in_ch=input_channels_count, num_classes=num_classes)
    path = "NetModel_seq90.pth"
    model.load_state_dict(torch.load(path))

    labels_length = len(labels)  # how many labels of Irises we have. = 3 in our database.
    labels_correct = list(0. for i in range(
        labels_length))  # list to calculate correct labels [how many correct setosa, how many correct versicolor, how many correct virginica]
    labels_total = list(0. for i in range(
        labels_length))  # list to keep the total # of labels per type [total setosa, total versicolor, total virginica]

    with torch.no_grad():
        for data in test_loader:
            inputs, outputs = data
            predicted_outputs = model(inputs)
            _, predicted = torch.max(predicted_outputs, 1)

            label_correct_running = (predicted == outputs).squeeze()
            label = outputs[0]
            if label_correct_running.item():
                labels_correct[label] += 1
            labels_total[label] += 1

    label_list = list(labels.keys())
    for i in range(num_classes):
        print('Accuracy to predict %5s : %2d %%' % (label_list[i], 100 * labels_correct[i] / labels_total[i]))

# Training loop
model.to(device)
model.train()

num_epochs = 3
train(num_epochs, continue_while_accuracy_is_improving=True)
print('Finished Training\n')
test()
# test_species()
