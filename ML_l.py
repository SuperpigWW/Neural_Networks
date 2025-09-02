import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Ensure the reproducibility of the training process
torch.manual_seed(62)


def generate_data(quantity=1000):

    # Set the range of x values and add noise to simulate real data y
    # x, noise size (1000, 1)
    x = torch.rand(quantity, 1) * (10 - 2) + 2
    noise = torch.rand(quantity, 1) * 16
    y = x * x + 2 * x + 1 + noise

    return x, y


# Get the x and y tensors
x_tensor, y_tensor = generate_data()


class FittingNet(nn.Module):

    def __init__(self):
        super(FittingNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(1, 32),   # From the input layer to hidden layer 1
            nn.ReLU(),
            nn.Linear(32, 32),   # From the hidden layer 1 to hidden layer 2
            nn.ReLU(),
            nn.Linear(32, 32),   # From the hidden layer 2 to hidden layer 3
            nn.ReLU(),
            nn.Linear(32, 1)   # From the hidden layer 3 to output layer
        )

    # Return the output obtained after the input passes through the neural network
    def forward(self,x):
        return self.network(x)


model = FittingNet()

# Calculate the mean squared error using method nn.MSELoss()
criterion = nn.MSELoss()

# Use the Adam optimizer to optimize the parameters
optimizer = optim.Adam(model.parameters(), lr=0.01)


def train_model(model, x, y, criterion, optimizer, times=1000):

    for t in range(times):

        # Forward pass
        predictions = model(x)
        loss = criterion(predictions, y)

        # Backward pass
        optimizer.zero_grad()   # Zero the gradients
        loss.backward()   # Compute the gradients
        optimizer.step()   # Step the optimizer


train_model(model, x_tensor, y_tensor, criterion, optimizer, times=1000)


def plot_fitting_curve_simple(model, x_data, y_data):

    plt.figure(figsize=(10, 6))

    # Set the model to evaluation mode
    model.eval()
    with torch.no_grad():
        # Generate
        x_test = torch.linspace(2, 10, 300).reshape(-1, 1)
        y_pred = model(x_test)

        # Paint
        plt.scatter(x_data.numpy(), y_data.numpy(), alpha=0.3, s=10, color='blue')
        plt.plot(x_test.numpy(), y_pred.numpy(), 'r-', linewidth=2)

        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid(True)
        plt.show()


plot_fitting_curve_simple(model, x_tensor, y_tensor)




