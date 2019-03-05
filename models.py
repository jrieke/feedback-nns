from torch import nn
import torch.nn.functional as F


class FeedforwardNet(nn.Module):

    def __init__(self, num_hidden=500, dropout=0.5):
        super(FeedforwardNet, self).__init__()
        self.forward1 = nn.Linear(784, num_hidden)
        self.forward2 = nn.Linear(num_hidden, 26)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.forward1(x))
        x = self.dropout(x)
        x = F.log_softmax(self.forward2(x), dim=1)
        return x


class FeedbackNet(nn.Module):

    def __init__(self, num_hidden=500, dropout=0.5, alpha=0.5, num_passes=2):
        super(FeedbackNet, self).__init__()
        self.forward1 = nn.Linear(784, num_hidden)
        self.forward2 = nn.Linear(num_hidden, 26)

        self.feedback1 = nn.Linear(num_hidden, 784)

        self.dropout = nn.Dropout(dropout)
        self.alpha = alpha
        self.num_passes = num_passes

    def forward(self, inp):
        inp = inp.view(-1, 784)

        # 1st forward pass (without feedback).
        x = F.relu(self.forward1(inp))

        for i in range(self.num_passes-1):
            # Feedback pass.
            feedback_activation_1 = self.feedback1(x)
            # Second non-linearity, simulated separate dendritic compartments.
            #feedback_activation_1 = F.relu(feedback_activation_1)

            # 2nd/3rd/... forward pass (with feedback).
            x = F.relu(self.forward1((1 - self.alpha) * inp + self.alpha * feedback_activation_1))

        # Output layer.
        x = self.dropout(x)
        x = F.log_softmax(self.forward2(x), dim=1)

        return x


class ConvNet(nn.Module):
    """ConvNet from the official PyTorch tutorial, achieves around 98 % accuracy on test set (https://github.com/pytorch/examples/blob/master/mnist/main.py)."""

    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
