from customConv3d import customConv3d
import numpy as np
import torch

test_data_1 = torch.rand(3,28,28,28)
test_data_2 = torch.rand(4,5,6,7)
test_data_3 = torch.rand(1,1,1,1)


def test1():
    customConv = customConv3d(in_channels=3, out_channels=1, kernel_size=3, stride=1, padding=0\
    , dilation=2, groups=1, bias=True, padding_mode="zeros")
    result, flter, bias_value = customConv(test_data_1)
    torchConv = torch.nn.Conv3d(in_channels=3,out_channels=1, kernel_size=3, stride=1, padding=0\
    , dilation=2, groups=1, bias=True, padding_mode="zeros")
    torchConv.weight.data = torch.tensor(flter)
    torchConv.bias.data = torch.tensor(bias_value)
    customResult = str(np.round(result,2))
    torchResult = str(np.round(np.array(torchConv(test_data_1).data),2))
    assert customResult == torchResult

def test2():
    customConv = customConv3d(in_channels=4, out_channels=2, kernel_size=3, stride=1, padding=0\
    , dilation=1, groups=2, bias=True, padding_mode="zeros")
    result, flter, bias_value = customConv(test_data_2)
    torchConv = torch.nn.Conv3d(in_channels=4,out_channels=2, kernel_size=3, stride=1, padding=0\
    , dilation=1, groups=2, bias=True, padding_mode="zeros")
    torchConv.weight.data = torch.tensor(flter)
    torchConv.bias.data = torch.tensor(bias_value)
    customResult = str(np.round(result,2))
    torchResult = str(np.round(np.array(torchConv(test_data_2).data),2))
    assert customResult == torchResult
    
def test3():
    customConv = customConv3d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0\
    , dilation=1, groups=1, bias=True, padding_mode="zeros")
    result, flter, bias_value = customConv(test_data_3)
    torchConv = torch.nn.Conv3d(in_channels=1,out_channels=1, kernel_size=1, stride=1, padding=0\
    , dilation=1, groups=1, bias=True, padding_mode="zeros")
    torchConv.weight.data = torch.tensor(flter)
    torchConv.bias.data = torch.tensor(bias_value)
    customResult = str(np.round(result,2))
    torchResult = str(np.round(np.array(torchConv(test_data_3).data),2))
    assert customResult == torchResult
    
#test1()
#test2()
test3()
