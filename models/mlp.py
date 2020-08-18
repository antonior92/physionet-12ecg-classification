import torch.nn as nn


# define small classifier
class MlpClassifier(nn.Module):
    """ Simple classifier """

    def __init__(self, seq_len, in_size):
        super(MlpClassifier, self).__init__()
        self.input_size = int(in_size * seq_len)
        self.hidden_dim1 = 512
        self.hidden_dim2 = 256

        self.fc1 = nn.Linear(in_features=self.input_size, out_features=self.hidden_dim1)
        self.fc2 = nn.Linear(in_features=self.hidden_dim1, out_features=self.hidden_dim2)

    def forward(self, src):
        batch_size = src.size(0)

        src2 = src.reshape(batch_size, -1)
        src3 = nn.functional.relu(self.fc1(src2))
        out = nn.functional.relu(self.fc2(src3))

        return out
