import torch.nn as nn

class LoanPredictorDNN(nn.Module):
    """
    A variable-depth Deep Neural Network for loan default prediction.
    """
    def __init__(self, input_size, layer_sizes, dropout_rate=0.3):
        super(LoanPredictorDNN, self).__init__()
        
        layers = []
        in_features = input_size
        
        for out_features in layer_sizes:
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(out_features))
            layers.append(nn.Dropout(dropout_rate))
            in_features = out_features
            
        layers.append(nn.Linear(in_features, 1))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)