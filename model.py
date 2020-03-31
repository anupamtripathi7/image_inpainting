import torch
import torch.nn as nn


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
            input_size (int): Input size for each LSTM cell
            hidden_size (int): Hidden size for LSTM
            num_layers (int): Number of layers in each cell. Double for Bi-directional
            output_size (int): Output size for LSTM cells
        """
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers * 2
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=0.3)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):

        pixel_to_predict = (torch.sum(x, dim=2) == 0).nonzero().t()
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        out = self.fc(out[pixel_to_predict.unbind()])

        return out
