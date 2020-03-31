import torch
import torch.nn as nn
import torch.nn.functional as F
# print(torch.__version__)


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Hyper-parameters
# sequence_length = 200
# input_size = 3
# hidden_size = 8
# num_layers = 2
# output_size = 3
# batch_size = 128
# num_epochs = 20
# learning_rate = 0.01


# Recurrent neural network (many-to-one)
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        """
        Args:
            input_size:
            hidden_size:
            num_layers:
            output_size:
        """
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers * 2
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=0.3)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        # Set initial hidden and cell states
        # print("x",x.size())
        pixel_to_predict = (torch.sum(x, dim=2) == 0).nonzero().t()
        # print(pixel_to_predict)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM

        # print(x.type(), h0.type(), c0.type())
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        # Decode the hidden state of the last time step
        # print(out.size())

        out = self.fc(out[pixel_to_predict.unbind()])

        return out
