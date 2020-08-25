# some very simple net to text ONNX interop
# taken form: https://www.kaggle.com/aakashns/pytorch-basics-linear-regression-from-scratch
import torch.onnx

class SimpleNet(nn.Module):
    # Initialize the layers
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(3, 3)
        self.act1 = nn.ReLU() # Activation function
        self.linear2 = nn.Linear(3, 2)

    def forward(self, x):
        x = self.linear1(x)
        x = self.act1(x)
        x = self.linear2(x)
        return x


def main():
    x = torch.rand(5, 3)
    net = SimpleNet()
    net(x)  # make sure it actually works
    torch.onnx.export(net, x, expanduser("simplenet.onnx"))
