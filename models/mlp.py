import torch.nn as nn


# define small classifier
class MlpClassifier(nn.Module):
    """ Simple classifier """

    def __init__(self, args, n_classes, pretrain_stage_config):
        super(MlpClassifier, self).__init__()
        self.input_size = int(args['pretrain_output_size'] * args['seq_length'])
        self.hidden_dim1 = 512
        self.hidden_dim2 = 256

        self.freeze = not args['finetuning']

        self.fc1 = nn.Linear(in_features=self.input_size, out_features=self.hidden_dim1)
        self.fc2 = nn.Linear(in_features=self.hidden_dim1, out_features=self.hidden_dim2)
        self.fc3 = nn.Linear(in_features=self.hidden_dim2, out_features=n_classes)

    def forward(self, src):
        batch_size = src.size(0)

        if self.freeze:
            # detach src
            src1 = src.data
        else:
            src1 = src

        src2 = src1.reshape(batch_size, -1)
        src3 = nn.functional.relu(self.fc1(src2))
        src4 = nn.functional.relu(self.fc2(src3))
        out = self.fc3(src4)

        return out
