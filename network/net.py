# -*- coding:utf-8 -*-
import mxnet as mx
from network.unit import *


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
    pose_vector = mx.sym.broadcast_mul(angle.slice_axis(axis=1, begin=0, end=1), yaw_vector) + mx.sym.broadcast_mul(
        angle.slice_axis(axis=1, begin=1, end=2), roll_vector)
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
    conv6 = conv(px5, 7, kernel=(5, 5), name='decoder_6', slope=0.2, use_px=False)
    # all label mask (B, 1, 112, 112)
    mask = mx.sym.sum(conv6.slice_axis(axis=1, begin=0, end=4), axis=1, keepdims=True)
    out = mx.sym.broadcast_mul(conv6.slice_axis(axis=1, begin=4, end=7), mask, name='attention')  # B, 3, 112, 112

    return out


def QuanLoss():
    """
        Model Trainable

        Loss 4 : MSE(L2) of vector (Reconstruction Loss)
        Loss 5 : MAE(L1) of image (Semantic Consistency Loss)
    """
    # B, 512
    real_vector = mx.sym.var('real_vector')
    fake_vector = mx.sym.var('fake_vector')
    # B, 3, 112, 112
    real_image = mx.sym.var('real_image')
    fake_image = mx.sym.var('fake_image')

    # Loss 4
    loss4 = mx.sym.mean(mx.sym.sqrt(mx.sym.sum(mx.sym.square(mx.sym.elemwise_sub(real_vector, fake_vector)), axis=1)))
    # Loss 5
    loss5 = mx.sym.mean(mx.sym.sum(mx.sym.abs(mx.sym.elemwise_sub(real_image, fake_image)), axis=1))

    return mx.sym.MakeLoss(loss4 + loss5)


def QuanDiscr():
    """
    input:
            fake_image : (B, 3, 112, 112)
            gan_label  : (B, 1, )
            cls_lable  : (B, 18, 1, 1)    real or fake, facial hair, age, skin, gender, yaw angle, roll angle
            angle_label: (B, 2, )                             4       6     6     2
    loss:
        Loss1: fake or real will use gan loss
        Loss2: four kind of classification problem will use lsgan
        Loss3: angle will use L1 loss (MAE)

    """
    # B, 3, 112, 112
    fake_image = mx.sym.var('fake_image')
    # real or fake, facial hair, age, skin, gender, yaw angle, roll angle
    #                     4       6     6     2

    # B, 1
    gan_label = mx.sym.var('gan_label').reshape((0, 0, 1, 1))

    # B, 18
    cls_label = mx.sym.var('cls_label').reshape(shape=(0, 0, 1, 1), name='cls_label_shape').reshape((0, 0, 1, 1))

    # B, 2
    angle_lable = mx.sym.var('angle_label').reshape((0, 0, 1, 1))

    # 1. Conv(3, 64, 64)
    conv1 = conv(fake_image, 3, kernel=(3, 3), stride=(2, 2), slope=0.2, name='discr1_1', use_px=False)
    # 2. Conv(64, 32, 32)
    conv2 = conv(conv1, 64, kernel=(3, 3), stride=(2, 2), slope=0.2, name='discr1_2', use_px=False)
    # 3. Conv(128, 16, 16)
    conv3 = conv(conv2, 128, kernel=(3, 3), stride=(2, 2), slope=0.2, name='discr1_3', use_px=False)
    # --------
    # Loss 2
    branch1 = conv(conv3, 18, kernel=(3, 3), stride=(2, 2), slope=0.2, name='discr1_branch_1', use_px=False)
    loss2 = lsgan_loss(branch1, cls_label)
    # --------
    # 4. Conv(256, 8, 8)
    conv4 = conv(conv3, 256, kernel=(3, 3), stride=(2, 2), slope=0.2, name='discr1_4', use_px=False)
    # 5. Conv(512, 4, 4)
    conv5 = conv(conv4, 512, kernel=(3, 3), stride=(2, 2), slope=0.2, name='discr1_5', use_px=False)
    # 6. Conv(3, 4, 4)
    out1 = mx.sym.Convolution(conv5, kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_filter=1, name='discri1_out1')
    out2 = mx.sym.Convolution(conv5, kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_filter=2, name='discri1_out1')
    # --------
    # Loss 1 and Loss 3
    loss1 = binary_cross_entropy(out1, gan_label)
    loss3 = mae_loss(out2, angle_lable)
    # --------
    return mx.sym.MakeLoss(loss1 + loss2 + loss3)


if __name__ == '__main__':
    # print(QuanDecoder().infer_shape(embedding_vector=(10, 512), decoder_label=(10, 4), angle=(10, 2)))
    # print(QuanDiscr().infer_shape(fake_image=(10, 3, 112, 112), gan_label=(10, 1), cls_label=(10, 18), angle_label=(10, 2)))
    print(QuanLoss().infer_shape(real_vector=(10, 512), fake_vector=(10, 512), real_image=(10, 3, 112, 112),
                                 fake_image=(10, 3, 112, 112)))
