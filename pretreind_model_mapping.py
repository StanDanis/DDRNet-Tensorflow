from torch import load
import numpy as np

def weight_database():

    layers_name_l = ['layer1.0', 'layer1.1',  'layer2.0', 'layer2.1', 'layer3.0', 'layer3.1',
                    'layer4_.0', 'layer4_.1', 'layer4.0', 'layer4.1', 'layer3_.0', 'layer3_.1']
    layers_name = ['layer2.0.downsample', 'layer3.0.downsample', 'layer4.0.downsample', 
                    'layer5.0.downsample', 'layer5_.0.downsample']

    layers_name_l2 = ['layer5.0', 'layer5_.0']

    layers_name2 = ['conv1', 'down4']

    scale = ['spp.scale1', 'spp.scale2', 'spp.scale3', 'spp.scale4']
    scale2 = ['spp.scale0', 'spp.process1','spp.process2', 'spp.process3', 'spp.process4', 
                'spp.compression', 'spp.shortcut']

    return layers_name_l, layers_name, layers_name2, layers_name_l2, scale, scale2


def set_weight(model, path=None, test=False, random_weight=None):
    if path==None:
        path = 'DDRNet23s_imagenet.pth'

    pretrained_state = load(path, map_location='cpu') 

    layers_name_l, layers_name, layers_name2, layers_name_l2, scale, scale2 = weight_database()

    def layer_loop(layer_list, range, suffix, conv_bias, model=model, plusk=[0, 0], 
                    condition=[], down_weight=pretrained_state):

        for name in layer_list:
            for i, k in enumerate(range):
                layer_name = f'{name}.{suffix[0]}{k + plusk[0]}'
                print(layer_name)

                if conv_bias[i] and (name not in condition):
                    weight = np.transpose(down_weight[f'{layer_name}.weight'], (2, 3, 1, 0))
                    bias = down_weight[f'{layer_name}.bias']
                    model.get_layer(layer_name).set_weights([weight, bias])
                else:
                    weight = np.transpose(down_weight[f'{layer_name}.weight'], (2, 3, 1, 0))
                    model.get_layer(layer_name).set_weights([weight])
                
                layer_name = f'{name}.{suffix[1]}{k + plusk[1]}'
                print(layer_name)

                gamma = down_weight[layer_name +'.weight'].numpy()
                beta = down_weight[layer_name +'.bias'].numpy()
                mean = down_weight[layer_name +'.running_mean'].numpy()
                var = down_weight[layer_name +'.running_var'].numpy()
                model.get_layer(layer_name).set_weights([gamma, beta, mean, var])

    layer_loop(layers_name_l2, range(1, 4), ['conv', 'bn'], [False, False, False])
    layer_loop(layers_name_l, range(1, 3), ['conv', 'bn'], [False, False])
    layer_loop(layers_name, range(0, 1), ['', ''], [False, True], plusk=[0, 1])

    layer_loop(layers_name2, range(0, 5, 3), ['', ''], [True, True], plusk=[0, 1], 
                    condition=['down4'])

    if test != None:
        layer_loop(scale, range(0, 1), ['', ''], [False, False], plusk=[3, 1], 
        down_weight=random_weight)
        layer_loop(scale2, range(0, 1), ['', ''], [False, False], plusk=[2, 0], 
        down_weight=random_weight)
        layer_loop(['final_layer'], range(1, 3), ['conv', 'bn'], [False, True], 
            down_weight=random_weight)






