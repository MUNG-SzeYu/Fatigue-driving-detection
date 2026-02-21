import torch.nn as nn 

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
    
if __name__ == "__main__":
    input_size = 10
    hidden_size = 20
    output_size = 5
    model = MLP(input_size, hidden_size, output_size)
    print(model)

