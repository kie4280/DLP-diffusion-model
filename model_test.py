import torch 
from torch import nn

class Model(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
      super().__init__(*args, **kwargs) 
      self.covn1 = nn.Identity()


    def forward(self, x):
        return x


def test_save():
    m = Model()
    print(m.state_dict().keys())
    setattr(m, "conv2", nn.Conv2d(12, 12, 2) )
    setattr(m, "conv1", nn.Conv2d(12, 12, 2) )
    print(m.state_dict().keys())
    del m.conv1
    print(m.state_dict().keys())
    
    # print(torch.load("checkpoints/8.pth").keys())




if __name__ == "__main__":
    test_save()