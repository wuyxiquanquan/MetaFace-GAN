import mxnet as mx


def conv(data, num_filter, kernel, name, stride=(1, 1), slope=0.1, use_px=False):
    pad = (kernel[0] // 2, kernel[1] // 2)
    num_filter = num_filter * 4 if use_px else num_filter
    conv = mx.sym.Convolution(data, kernel=kernel, pad=pad, stride=stride, num_filter=num_filter, name='conv' + name)
    IN = mx.sym.InstanceNorm(conv, name='conv_IN' + name)
    out = mx.sym.LeakyReLU(IN, act_type='leaky', slope=0.1, name='act' + name)
    if use_px:
        px1 = mx.sym.reshape(out, shape=(0, -4, 2, 2, -2), name='px1_' + name)  # (B, C, 2, 2, H, W)
        px2 = mx.sym.transpose(px1, axes=(0, 1, 4, 2, 5, 3), name='px2_' + name)  # (B, C, H, 2, W, 2)
        out = mx.sym.reshape(px2, shape=(0, 0, -3, -3), name='px3_' + name)  # (B, C, H*2, W*2)
    return out


def fc(data, num_hidden, name):
    fc1 = mx.sym.FullyConnected(data, num_hidden=num_hidden, no_bias=True, name='fc' + name)
    IN = mx.sym.InstanceNorm(fc1, name='fc_IN' + name)
    out = mx.sym.Activation(IN, act_type='relu', name='act' + name)
    return out


def QuanDecoder():
    # B, 512
    embedding = mx.sym.var('embedding_vector')
    # label [Hair, Age, Skin, Gender] B, 4
    label = mx.sym.var('label')
    # angle [yaw, roll] B, 2
    angle = mx.sym.var('angle')

    # 1. Deconv (B, 512, 7, 7)
    deconv1 = mx.sym.Deconvolution(embedding, kernel=(7, 7), num_filter=512, no_bias=True, name='decoder_deconv1')
    deconv_IN1 = mx.sym.InstanceNorm(deconv1, name='decoder_deconv_IN1')
    deconv_act1 = mx.sym.LeakyReLU(deconv_IN1, act_type='leaky', slope=0.1, name='decoder_deconv_act1')
    # 2. PX (B, 256, 14, 14)
    px2 = conv(deconv_act1, 256, kernel=(3, 3), name='decoder_2', use_px=True)
    # 3. PX (B, 128, 28, 28)
    px3 = conv(px2, 128, kernel=(3, 3), name='decoder_3', use_px=True)
    # 4. PX (B, 64, 56, 56)
    px4 = conv(px3, 64, kernel=(3, 3), name='decoder_4', use_px=True)
    # ------------------
    # insert
    # ------------------
    # 5. PX (B, 32, 112, 112)
    px5 = conv(px4, 32, kernel=(3, 3), name='decoder_5', slope=0.2, use_px=True)
    # 6. Conv(B, 32, 112, 112) 5x5
    conv6 = conv(px5, 32, kernel=(5, 5), name='decoder_6', slope=0.2, use_px=False)
