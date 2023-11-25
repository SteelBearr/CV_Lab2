#Подключение модулей
import numpy as np
import torch

def customConv3d(in_channels, out_channels, kernel_size, stride=1, padding=0\
    , dilation=1, groups=1, bias=True, padding_mode="zeros"):
    #Обёртка
    def wrapper(tens):
        #Проверки параметров
        if (in_channels%groups !=0) or (out_channels%groups !=0):
            raise Exception(f"%s must be divisible by groups" % ("in_channels" if in_channels%groups !=0 else "out_channels"))
        if padding < 0:
            raise Exception(f"Padiing should be 0 or positive")
        if stride < 0:
            raise Exception(f"Stride should be 0 or positive")
        if groups < 0:
            raise Exception(f"Groups should be positive")  
        
        #Смещение
        if bias:
            bias_value = torch.rand(out_channels)
        else:
            bias_value = torch.zeros(out_channels)
        
        #Подложка
        if (padding_mode == 'zeros'):
            pad = torch.nn.ZeroPad3d(padding)
            tens = pad(tens)
        elif (padding_mode == 'reflect'):
            pad = torch.nn.ReflectionPad3d(padding)
            tens = pad(tens)
        elif (padding_mode == 'replicate'):
            pad = torch.nn.ReplicationPad3d(padding)
            tens = pad(tens)
        elif (padding_mode == 'circular'):
            pad = torch.nn.CircularPad3d(padding)
            tens = pad(tens)
        else:
            raise Exception(f"Ivalid padding_mode")
        
        #Размеры ядра
        if type(kernel_size) == tuple:
            flter = torch.rand(out_channels, in_channels//groups, kernel_size[0],kernel_size[1],kernel_size[2])
        elif type(kernel_size) == int:
            flter = torch.rand(out_channels, in_channels//groups, kernel_size,kernel_size,kernel_size)
        else:
            raise Exception(f"Ivalid kernel_size type")
            
        #"Обход" ядром
        res = []
        for chnl in range(out_channels):
            feature_map = np.array([])
            for i in range(0, tens.shape[1] - ((flter.shape[2]- 1) * dilation + 1) + 1, stride):
                for j in range(0, tens.shape[2] - ((flter.shape[3]- 1) * dilation + 1) + 1, stride):
                    for k in range(0, tens.shape[3] - ((flter.shape[4]- 1) * dilation + 1) + 1, stride):
                        total = 0
                        for f in range(in_channels // groups):
                            if groups > 1:
                                cur = tens[chnl * (in_channels // groups) + f]\
                                [i:i + (flter.shape[2] - 1) * dilation + 1 : dilation,\
                                 j:j + + (flter.shape[3] - 1) * dilation + 1 : dilation,\
                                 k:k + + (flter.shape[4] - 1) * dilation + 1 : dilation]
                            else:
                                cur = tens[f]\
                                [i:i + (flter.shape[2] - 1) * dilation + 1 : dilation,\
                                 j:j + + (flter.shape[3] - 1) * dilation + 1 : dilation,
                                 k:k + + (flter.shape[4] - 1) * dilation + 1 : dilation]
                            total += (cur * flter[chnl][f]).sum()
                        feature_map = np.append(feature_map, float(total + bias_value[chnl]))
            res.append(feature_map.reshape((tens.shape[1] - ((flter.shape[2] - 1) * dilation + 1)) // stride + 1,\
                          (tens.shape[2] - ((flter.shape[3] - 1) * dilation + 1)) // stride + 1,\
                          (tens.shape[3] - ((flter.shape[4] - 1) * dilation + 1)) // stride + 1))
        return np.array(res),np.array(flter), np.array(bias_value)
    return wrapper
