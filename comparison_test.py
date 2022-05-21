from ddrnet_23_slim import DualResNet_imagenet as DualResNet_imagenet_tens
from DDRNet_23_slim_official import DualResNet_imagenet_off
from pretrained_model_mapping import set_weight
import torch
import numpy as np
import keras
import tensorflow as tf

def test(torch_output, tf_output, i):
    # this is test func. that can compare edited pytoch and tensorflow model output
    tf_output = tf_output[i]
    torch_layer = list(torch_output.keys())[i]
    torch_output = torch_output[torch_layer]
    
    torch_output = np.transpose(torch_output.detach().numpy(),(0, 2, 3, 1))
    max_diff = np.max(np.abs(tf_output.flatten() - torch_output.flatten()))
    avg_diff = np.average(np.abs(tf_output.flatten() - torch_output.flatten()))
    print(f'Max difference in {torch_layer} is :{max_diff} and avg is: {avg_diff}')
    
    return max_diff

if __name__ == '__main__':

    # define original torch model
    net = DualResNet_imagenet_off(pretrained=True)
    weight_net = net.state_dict()
    net = net.eval()

    # define tensorflow model
    model = DualResNet_imagenet_tens((224, 224, 3), 3, 19)
    set_weight(model, path='DDRNet23s_imagenet.pth', 
           test=True, random_weight=weight_net)

    # torch and tensorflow random input tensor
    net_inputs = torch.Tensor(np.random.rand(3, 3, 224, 224))
    tf_inputs = np.transpose(net_inputs.numpy(), (0, 2, 3, 1))

    net = net.eval()
    with torch.no_grad():
        torch_output = net(net_inputs)

    tf_output = model.predict(tf_inputs)


    res = []
    for i in range(15):
        res.append(test(torch_output, tf_output, i))

