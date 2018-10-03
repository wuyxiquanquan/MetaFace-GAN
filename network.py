# -*- coding:utf-8 -*-
import mxnet as mx


def conv(data, num_filter, kernel, name, stride=(1, 1), slope=0.1, use_px=False):
    pad = (kernel[0] // 2, kernel[1] // 2)
    num_filter = num_filter * 4 if use_px else num_filter
    conv = mx.sym.Convolution(data, kernel=kernel, pad=pad, stride=stride, num_filter=num_filter, name='conv' + name)
    IN = mx.sym.InstanceNorm(conv, name='conv_IN' + name)
    out = mx.sym.LeakyReLU(IN, act_type='leaky', slope=slope, name='act' + name)
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


def mapping(data, num_hidden, name):
    fc1 = mx.sym.FullyConnected(data, num_hidden=num_hidden, name=name + '_fc1')
    relu1 = mx.sym.Activation(fc1, act_type='relu', name='_relu1')
    fc2 = mx.sym.FullyConnected(relu1, num_hidden=num_hidden, name=name + 'fc2')
    out = mx.sym.Activation(fc2, act_type='relu', name='_relu2')
    return out


def QuanDecoder():
    # B, 512
    embedding = mx.sym.var('embedding_vector')
    # label [Hair, Age, Skin, Gender] B, 4
    label = mx.sym.var('decoder_label')
    # angle [yaw, roll] B, 2
    angle = mx.sym.var('angle')
    # Mapping for yaw and roll
    yaw_vector = mapping(embedding, 512, 'yaw_mapping')  # B, 512
    roll_vector = mapping(embedding, 512, 'roll_mapping')
    # 系数可考虑
    pose_vector = mx.sym.broadcast_mul(angle[:, :1], yaw_vector) + mx.sym.broadcast_mul(angle[:, 1:], roll_vector)
    vector = mx.sym.elemwise_add(embedding, pose_vector, name='vector_add')
    vector = mx.sym.reshape(vector, shape=(0, 0, 1, 1), name='vector_reshape')  # B, 512, 1, 1
    # 1. Deconv (B, 512, 7, 7)
    deconv1 = mx.sym.Deconvolution(vector, kernel=(7, 7), num_filter=512, no_bias=True, name='decoder_deconv1')
    deconv_IN1 = mx.sym.InstanceNorm(deconv1, name='decoder_deconv_IN1')
    deconv_act1 = mx.sym.LeakyReLU(deconv_IN1, act_type='leaky', slope=0.1, name='decoder_deconv_act1')
    # 2. PX (B, 256, 14, 14)
    px2 = conv(deconv_act1, 256, kernel=(3, 3), name='decoder_2', use_px=True)
    # 3. PX (B, 128, 28, 28)
    px3 = conv(px2, 128, kernel=(3, 3), name='decoder_3', use_px=True)
    # insert labels
    label = mx.sym.reshape(label, shape=(0, 0, 1, 1), name='label_reshape')
    label = mx.sym.broadcast_axis(data=label, axis=(2, 3), size=(28, 28), name='label_broadcast')  # B, 4, 28, 28
    px3 = mx.sym.concat(px3, label, dim=1)  # B, 128+4, 28, 28
    # 4. PX (B, 64, 56, 56)
    px4 = conv(px3, 64, kernel=(3, 3), name='decoder_4', use_px=True)
    # 5. PX (B, 32, 112, 112)
    px5 = conv(px4, 32, kernel=(3, 3), name='decoder_5', slope=0.2, use_px=True)
    # 6. Conv(B, 7, 112, 112) 5x5
    conv6 = conv(px5, 4, kernel=(5, 5), name='decoder_6', slope=0.2, use_px=False)
    # all label mask (B, 1, 112, 112)
    mask = conv6[:, 0:1] + conv6[:, 1:2] + conv6[:, 2:3] + conv6[3:4]
    out = mx.sym.broadcast_mul(conv6[4:], mask, name='attention')  # B, 3, 112, 112

    return out


def QuanDiscr():
    """
        Loss1: fake or real will use lsgan loss
        Loss2: four kind of classification problem will use softmax
        Loss3: angle will use L1 loss
    """
    # B, 3, 112, 112
    fake_image = mx.sym.var('fake_image')
    # B, 7, 1, 1
    # real or fake, facial hair, age, skin, gender, yaw angle, roll angle
    label = mx.sym.var('label')

    # 1. Conv(3, 64, 64)
    conv1 = conv(fake_image, 3, kernel=(3, 3), stride=(2, 2), slope=0.2, name='discr1_1', use_px=False)
    # 2. Conv(64, 32, 32)
    conv2 = conv(conv1, 64, kernel=(3, 3), stride=(2, 2), slope=0.2, name='discr1_2', use_px=False)
    # 3. Conv(128, 16, 16)
    conv3 = conv(conv2, 128, kernel=(3, 3), stride=(2, 2), slope=0.2, name = 'discr1_3', use_px = False)
    # --------
    # Loss 2
    # --------
    # 4. Conv(256, 8, 8)
    conv4 = conv(conv3, 256, kernel=(3, 3), stride=(2, 2), slope=0.2, name='discr1_4', use_px=False)
    # 5. Conv(512, 4, 4)
    out = conv(conv4, 512, kernel=(3, 3), stride=(2, 2), slope=0.2, name='discr1_5', use_px=False)
    # --------
    # Loss 1 and Loss 3
    # --------

def QuanLossModule():
    """
        Model Trainable

        Loss 4 : MSE(L2) of vector (Reconstruction Loss)
        Loss 5 : MAE(L1) of image (Semantic Consistency Loss)
    """
    real_vector = mx.sym.var('real_vector')
    fake_vector = mx.sym.var('fake_vector')
    real_image = mx.sym.var('real_image')
    fake_image = mx.sym.var('fake_image')

    # Loss 4
    loss4 = mx.sym.mean(mx.sym.sqrt(mx.sym.sum(mx.sym.square(mx.sym.elemwise_sub(real_vector, fake_vector)), axis=1)))
    # Loss 5
    loss5 = mx.sym.mean(mx.sym.sum(mx.sym.abs(mx.sym.elemwise_sub(real_image, fake_image)), axis=1))

    loss = loss4 + loss5
    return mx.sym.MakeLoss(loss)





