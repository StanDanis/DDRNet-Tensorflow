from tensorflow import keras
from keras import layers
import tensorflow as tf
from pretrained_model_mapping import set_weight


def plusone(num, multi=1):
      return str((int(num) + 1)*multi)

def con_bn_relu(x, filter, kernel_size=3, strides=2, name='', padding='valid', num='0',
                relu=True, bias=True, long_n=True, zeropad=True, pad=1):

    if long_n:
      conv_name = f'{name}.conv{num}'
      bn_name = f'{name}.bn{num}'
      relu_name = f'{name}.relu{num}'
    else:
      conv_name = f'{name}.{num}'
      num = plusone(num)
      bn_name = f'{name}.{num}'
      num = plusone(num)
      relu_name = f'{name}.{num}'

    if zeropad:
      x = layers.ZeroPadding2D(pad)(x)

    x = layers.Conv2D(filters=filter, kernel_size=kernel_size, strides=strides, 
                      padding=padding, name=conv_name, use_bias=bias)(x)
    x = layers.BatchNormalization(
                  momentum=0.1,
                  epsilon=1e-05,
                  name=bn_name
                  )(x)
    if relu:
      x = layers.ReLU(name=relu_name)(x)
    return x, plusone(num)

def residual_block(x, filter, kernel_size=3, stride=[1, 1], name='', padding='valid', num='1',
                relu=[True, False, True], bias=[False, False], downsample=False):

  if downsample:
    x_orig, num_ =  con_bn_relu(x, 
                          filter=filter, 
                          kernel_size=1, 
                          strides=2,
                          padding='valid', 
                          relu=False,
                          name=f'{name}.downsample',
                          long_n=False, 
                          bias=False, 
                          zeropad=False)
    stride = [2, 1]
  else:
    x_orig = x

  x, num = con_bn_relu(x, filter, kernel_size, stride[0], name, padding, num=num, relu=relu[0],
                       bias=bias[0])
  
  x, num = con_bn_relu(x, filter, kernel_size, stride[1], name, padding, num=num, relu=relu[1],
                       bias=bias[1])
  
  x = layers.Add()([x, x_orig]) 

  if relu[2]:
    x = layers.ReLU()(x) 
  
  return x

def bottlenect_residual_block(x, filter, name, pad=[0, 1], kernel_size=[1, 3, 1], stride=[1, 1, 1], 
    padding='valid', num='1', relu=[True, True, False, False], bias=[False, False, False], 
    downsample=True,):
                
  if downsample:
    x_orig, num_ =  con_bn_relu(x, 
                          filter=filter * 2, 
                          kernel_size=1, 
                          strides=stride[1],
                          padding='valid', 
                          relu=False,
                          name=f'{name}.downsample',
                          long_n=False, 
                          bias=False, 
                          pad=pad[0],)
  else:
    x_orig = x

  x, num = con_bn_relu(x, filter, kernel_size[0], stride[0], name, padding, num=num, relu=relu[0],
                       bias=bias[0], zeropad=False)
  
  x, num = con_bn_relu(x, filter, kernel_size[1], stride[1], name, padding, num=num, relu=relu[1],
                       bias=bias[1], pad=pad[1])
  
  x, num = con_bn_relu(x, filter * 2, kernel_size[2], stride[2], name, padding, num=num, relu=relu[2],
                       bias=bias[2], zeropad=False)
  
  x = layers.Add()([x, x_orig]) 
  if relu[3]:
    x = layers.ReLU()(x) 
  return x

def compression(x, filter, name, output_shape):

    x = layers.ReLU()(x)
    x, n = con_bn_relu(x, filter=filter, kernel_size=1, strides=1, name=name, 
                        relu=False, bias=False, long_n=False, zeropad=False)

    x = tf.image.resize(x, output_shape)

    return x

def segmenthead(x, num_class, name, scale_factor=None):

  x = layers.BatchNormalization(
                  momentum=0.1,
                  epsilon=1e-05,
                  name=f'{name}.bn1'
                  )(x)

  x = layers.ReLU()(x)

  x = layers.ZeroPadding2D(1)(x)
  x = layers.Conv2D(filters=64, 
                      kernel_size=3, 
                      strides=1, 
                      padding='valid', 
                      name=f'{name}.conv1', 
                      use_bias=False)(x)

  out = layers.BatchNormalization(
                  momentum=0.1,
                  epsilon=1e-05,
                  name=f'{name}.bn2'
                  )(x)     

  out = layers.ReLU()(out)

  out = layers.Conv2D(filters=num_class, 
                      kernel_size=1, 
                      strides=1, 
                      padding='valid', 
                      name=f'{name}.conv2', 
                      use_bias=True)(out)


  if scale_factor is not None:
    height = x.shape[1] * scale_factor
    width = x.shape[2] * scale_factor
    out = tf.image.resize(out, (height, width))
  return out

def DAPPM(x, branch_filters, outfilters):

    input_shape = x.shape
    height = input_shape[1]
    width = input_shape[2]

    x_list = []

    def avgpool_bn_rel_con(x, pool_size, filter, kernel_size, stride, name, pad=[1, 1], 
                           padding='valid', bias=False, long_n=False, averp=True):

        if ('scale' in name) and (name[-1] in ['1', '2', '3', '4']):
          conv_name = f'{name}.3' 
          bn_name = f'{name}.1' 
          relu_name = f'{name}.2' 
        else:
          conv_name = f'{name}.2' 
          bn_name = f'{name}.0' 
          relu_name = f'{name}.1'
        
        if averp:
            x = layers.ZeroPadding2D(pad[0])(x)
            x = layers.AveragePooling2D(pool_size=pool_size,
                                    strides=stride, 
                                    padding=padding)(x)
        
        x = layers.BatchNormalization(
                    momentum=0.1,
                    epsilon=1e-05,
                    name=bn_name
                    )(x)
        x = layers.ReLU()(x)
            
        x = layers.ZeroPadding2D(pad[1])(x)
        x = layers.Conv2D(filters=filter, 
                        kernel_size=kernel_size, 
                        strides=1, 
                        padding=padding, 
                        name=conv_name, 
                        use_bias=bias)(x)

        return x

    def dappm_step(x, x1, pool, strides, pad, i, filter=branch_filters):
        scale = avgpool_bn_rel_con(x, pool, filter, 1, strides, name=f'spp.scale{i+1}',
                             pad=[pad, 0])
        upsample_scale = tf.image.resize(scale, size=(height,width))
        add = layers.Add()([upsample_scale, x1])  
        process = avgpool_bn_rel_con(add, (0, 0), filter, 3, [1, 1], name=f'spp.process{i+1}', 
                                    pad=[0, 1], averp=False)
        return process

    pool_h = [5, 9, 17, height]
    pool_w = [5, 9, 17, width]
    strides_h = [2, 4, 8, height]
    strides_w = [2, 4, 8, width]
    pads = [2, 4, 8, 0]

    scale0 = avgpool_bn_rel_con(x, (0, 0), 128, 1, [1, 1], 'spp.scale0', averp=False, 
                                pad=[0,0])
    x_list.append(scale0)

    for i in range(4):
        x_list.append(dappm_step(x, x_list[-1], (pool_h[i], pool_w[i]), [strides_h[i], 
                strides_w[i]], pads[i], i))

    shortcut = avgpool_bn_rel_con(x, (0, 0), 128, 1, [1, 1], 'spp.shortcut', averp=False,
                                    pad=[0,0])
    
    combined = layers.concatenate(x_list, axis=-1)
    compression = avgpool_bn_rel_con(combined, (0, 0), 128, 1, [1, 1], 'spp.compression', 
                                    averp=False, pad=[0, 0])
    
    final = layers.Add()([compression, shortcut])
    return final

def DualResNet(shape, batch_size, num_class=19, filters=32, spp_filters=128, head_filters=64, 
                augment=False, comparison_test=False):

    input_layer = keras.Input(shape=shape, batch_size=batch_size, name='input')

    input_shape = input_layer.shape
    height_output = input_shape[1] // 8
    width_output = input_shape[2] // 8

    highres_filters = filters * 2

    conv1, num = con_bn_relu(input_layer, filter=filters,  name='conv1', num='0'
                            ,long_n=False)

    conv1, num = con_bn_relu(conv1, filter=filters, name='conv1', num=num, 
                             long_n=False)

    layer1 = residual_block(conv1, filter=filters, name='layer1.0')
    layer1 = residual_block(layer1, filter=filters, name='layer1.1')

    layer2 = residual_block(layer1, filter=2*filters, stride=[2, 2], name='layer2.0', 
                            downsample=True)
    layer2 = residual_block(layer2, filter=2*filters, name='layer2.1')

    layer3 = residual_block(layer2, filter=4*filters, stride=[2, 2], name='layer3.0', 
                            downsample=True)
    layer3 = residual_block(layer3, filter=4*filters, name='layer3.1', 
                            relu=[True, False, False])

    layer3_ = residual_block(layer2, filter=highres_filters, name= 'layer3_.0')
    layer3_ = residual_block(layer3_, filter=highres_filters,  
                             name= 'layer3_.1', relu=[True, False, False])


    compression3 = compression(layer3, filter=highres_filters, name='compression3', 
                                output_shape=(height_output, width_output))
    compression3 = layers.Add()([layer3_, compression3])
    compression3 = layers.ReLU()(compression3) 

    down3, n = con_bn_relu(layers.ReLU()(layer3_), filter=4*filters, name='down3', 
                           relu=False, bias=False, long_n=False)
    down3 = layers.Add()([layer3, down3])
    down3 = layers.ReLU()(down3)


    layer4 = residual_block(down3, filter=8*filters, stride=[2, 2], name='layer4.0', 
                            downsample=True)
    layer4 = residual_block(layer4, filter=8*filters, name='layer4.1',
                            relu=[True, False, False])

    layer4_ = residual_block(compression3, filter=highres_filters,  
                             name='layer4_.0')
    layer4_ = residual_block(layer4_, filter=highres_filters, name='layer4_.1', 
                             relu=[True, False, False])


    compression4 = compression(layer4, filter=highres_filters, name='compression4', 
                                output_shape=(height_output, width_output))
    compression4 = layers.Add()([layer4_, compression4])
    compression4 = layers.ReLU()(compression4)


    down4, n = con_bn_relu(layers.ReLU()(layer4_), filter=4*filters, name='down4', 
                           bias=False, long_n=False)
    down4, n = con_bn_relu(down4, filter=8*filters, name='down4', num=n, 
                           relu=False, bias=False, long_n=False)
    down4 = layers.Add()([layer4, down4])
    down4 = layers.ReLU()(down4)

    layer5 = bottlenect_residual_block(down4, filter=8*filters, stride=[1, 2, 1], 
            name='layer5.0')

    layer5_ = bottlenect_residual_block(compression4, highres_filters, stride=[1, 1, 1], 
            name='layer5_.0')

    spp0 = DAPPM(layer5, spp_filters, filters*4)
    spp = tf.image.resize(spp0, (height_output, width_output)) 

    x_ = layers.Add()([spp, layer5_])
    x_ = segmenthead(x_, num_class, 'final_layer')

    if comparison_test:
      model = keras.Model(input_layer, [conv1, layer1, layer2, layer3, layer3_, down3,
                        compression3, layer4, layer4_, down4, compression4, layer5_, layer5,
                        spp0, spp, x_])
    else:
      model = keras.Model(input_layer, x_)

    return model

def DualResNet_imagenet(shape, batch, num_class):
  model = DualResNet(shape, batch, num_class)
  set_weight(model)
  return model

if __name__ == '__main__':
  model = DualResNet_imagenet((800, 800, 3), 3, 19)
  print(model)
