import mxnet as mx


def fc(data, num_hidden, name):
    fc1 = mx.sym.FullyConnected(data, num_hidden=num_hidden, no_bias=True, name='fc' + name)
    IN = mx.sym.InstanceNorm(fc1, name='fc_IN' + name)
    out = mx.sym.Activation(IN, act_type='relu', name='act' + name)
    return out


def mapping(data, num_hidden, name):
    fc1 = mx.sym.FullyConnected(data, num_hidden=num_hidden, name=name + '_fc1')
    relu1 = mx.sym.Activation(fc1, act_type='relu', name='_relu1')
    fc2 = mx.sym.FullyConnected(relu1, num_hidden=num_hidden, name=name + 'fc2')
    out = mx.sym.Activation(fc2, act_type='relu', name='_relu2')
    return out


def conv(data, num_filter, kernel, name, stride=(1, 1), slope=0.1, use_px=False):
    pad = (kernel[0] // 2, kernel[1] // 2)
    num_filter = num_filter * 4 if use_px else num_filter
    conv = mx.sym.Convolution(data, kernel=kernel, pad=pad, stride=stride, num_filter=num_filter, name='conv' + name)
    IN = mx.sym.InstanceNorm(conv, name='conv_IN' + name)
    out = mx.sym.LeakyReLU(IN, act_type='leaky', slope=slope, name='act' + name)
    if use_px:
        px1 = mx.sym.reshape(out, shape=(0, -4, -1, 4, -2), name='px1_' + name)  # (B, C, 2, 2, H, W)
        px2 = mx.sym.reshape(px1, shape=(0, 0, -4, 2, 2, -2), name='px2_' + name)  # (B, C, 4, H, W)
        px3 = mx.sym.transpose(px2, axes=(0, 1, 4, 2, 5, 3), name='px3_' + name)  # (B, C, H, 2, W, 2)
        out = mx.sym.reshape(px3, shape=(0, 0, -3, -3), name='out_' + name)  # (B, C, H*2, W*2)
    return out


def lsgan_loss(data, target):
    """

    :param data: (B, C, 16, 16)
    :param target: (B, C, 1, 1)
    :return: a scalar
    """
    return mx.sym.mean(mx.sym.square(mx.sym.broadcast_sub(data, target)))


def binary_cross_entropy(data, target):
    """

    :param data: (B, C, 4, 4)
    :param target: (B, C, 1, 1)
    :return: a scalar
    """
    loss1 = mx.sym.broadcast_mul(target, mx.sym.log(data))
    loss2 = mx.sym.broadcast_mul(mx.sym.ones_like(target) - target, data)
    return -mx.sym.mean(loss1 + loss2)


def mae_loss(data, target):
    """

    :param data: (B, C, 4, 4)
    :param target: (B, C, 1, 1)
    :return: a scalar
    """
    return mx.sym.mean(mx.sym.abs(mx.sym.broadcast_sub(data, target)))
