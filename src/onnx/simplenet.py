# some very simple net to text ONNX interop
# taken form: https://www.kaggle.com/aakashns/pytorch-basics-linear-regression-from-scratch
import torch.nn as nn
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
    torch.onnx.export(net, x, "src/onnx/simplenet.onnx")


# serialized model has such values:
# 
# linear1.weight:
# tensor([[ 0.2821, -0.3611,  0.5451],
#         [ 0.2561, -0.5167, -0.4392],
#         [-0.2397, -0.0145,  0.0690]], requires_grad=True)

# linear1.bias:
# tensor([-0.1348, -0.5235,  0.5765], requires_grad=True)

# linear2.weight:
# tensor([[-0.1989,  0.2440, -0.0081],
#         [-0.3160,  0.0791,  0.4595]], requires_grad=True)

# linear2.bias:
# tensor([-0.1370, -0.0544], requires_grad=True) 

# x:
# tensor([[0.5255, 0.1013, 0.1775],
#         [0.0673, 0.4982, 0.5595],
#         [0.3846, 0.7036, 0.4041],
#         [0.1113, 0.9785, 0.9942],
#         [0.0680, 0.8244, 0.2248]])
