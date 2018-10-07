import mxnet as mx
from tensorboardX import SummaryWriter
from network.net import QuanDecoder, QuanDiscr, QuanLoss
import os
from DataIters.FRVTImageIter import FRVTImageIter
from tqdm import tqdm
import time
import numpy as np
from mxnet.io import DataBatch
np.set_printoptions(suppress=True)

batch_size = 100

# --------- multicard ---------
ctx = []


def label_broad(labels):
    # labels [B, 4]
    labels[:, 1] += 4
    labels[:, 2] += 9
    labels[:, 3] += 15
    temp = mx.nd.zeros((batch_size, 17))
    for i in range(4):
        temp[mx.nd.arange(batch_size), labels[:, i]] = 1
    return temp


# cvd = os.environ['CUDA_VISIBLE_DEVICES'].strip()
# cvd = '0,1,2,3,4,5,6,7'
cvd = ''
if len(cvd) > 0:
    for i in range(len(cvd.split(','))):
        ctx.append(mx.gpu(i))
if len(ctx) == 0:
    ctx = [mx.cpu()]
    print('use CPU')
else:
    print('GPU num:', len(ctx))
# -----create Module----
# 1. Module
# 2. bind
# 3. init_params
# 4. init_optimizer
# --------- DECODER ---------
decoder = mx.mod.Module(symbol=QuanDecoder(),
                        data_names=['decoder_real_vector', 'decoder_cls_label', 'decoder_angle_label'],
                        label_names=None,
                        context=ctx, )
decoder.bind(data_shapes=[('decoder_real_vector', (batch_size, 512)), ('decoder_cls_label', (batch_size, 4)),
                          ('decoder_angle_label', (batch_size, 2))],
             label_shapes=None,
             for_training=True, inputs_need_grad=False)
decoder.init_params(initializer=mx.init.Xavier(),
                    allow_missing=True)
decoder.init_optimizer(optimizer='adam',
                       optimizer_params={
                           'learning_rate': 1e-4,
                           'beta1': 0.5,
                       })

# --------- DISCRIMINATOR ---------
discri = mx.mod.Module(symbol=QuanDiscr(),
                       data_names=['discri_image', 'discri_gan_label', 'discri_cls_label', 'discri_angle_label'],
                       label_names=None,
                       context=ctx, )
discri.bind(data_shapes=[('discri_image', (batch_size, 3, 112, 112)), ('discri_gan_label', (batch_size, 1)),
                         ('discri_cls_label', (batch_size, 17)), ('discri_angle_label', (batch_size, 2))],
            label_shapes=None,
            for_training=True, inputs_need_grad=True)
discri.init_params(initializer=mx.init.Xavier(),
                   allow_missing=True)
discri.init_optimizer(optimizer='adam',
                      optimizer_params={
                          'learning_rate': 1e-4,
                          'beta1': 0.5,
                      })

# --------- LossModule ---------
loss_module = mx.mod.Module(symbol=QuanLoss(),
                            data_names=['real_vector', 'fake_vector', 'real_image', 'fake_image'],
                            label_names=None,
                            context=ctx, )
loss_module.bind(data_shapes=[('real_vector', (batch_size, 512)), ('fake_vector', (batch_size, 512)),
                              ('real_image', (batch_size, 3, 112, 112)), ('fake_image', (batch_size, 3, 112, 112))],
                 label_shapes=None,
                 for_training=True, inputs_need_grad=True)

# --------- fixed FR net ---------
fr_module = mx.module.Module.load('/home/wuyuxiang/quan/insightface/model_params/theBest', 0,
                                  data_names=['data', ], label_names=None,
                                  context=ctx, )
fr_module.bind([('data', (batch_size, 3, 112, 112))], None, for_training=True, inputs_need_grad=True)

train_dataiter = FRVTImageIter(batch_size, '/data1/ijb/IJB/IJB-C/protocols/ijbc_metadata.csv')
train_dataiter = mx.io.PrefetchingIter(train_dataiter)
'''
        input:                      size:                      
            data                            [B, 3, 112, 112]                
            
            decoder_real_vector             [B, 512]                        
            decoder_cls_label               [B, 4]
            decoder_angle_label             [B, 2]
            
            discri_image                    [B, 3, 112, 112]
            discri_gan_label                [B, 1]
            discri_cls_label                [B, 17]
            discri_angle_label              [B, 2]
            
            real_vector                     [B, 512]
            fake_vector                     [B, 512]
            real_image                      [B, 3, 112, 112]
            fake_image                      [B, 3, 112, 112]
'''
for epoch in range(1):
    train_dataiter.reset()
    for cur_time, databatch in enumerate((train_dataiter)):
        real_image, labels = databatch.data
        # print(labels.asnumpy())
        # time.sleep(1)
        # =================================================================================== #
        #                             1. Train the discriminator                              #
        # =================================================================================== #
        # discriminator data preparing
        # fake_image
        fr_module.forward(DataBatch([real_image]), is_train=False)
        real_vector = fr_module.get_outputs(merge_multi_context=True)[0]
        decoder.forward(DataBatch([real_vector, labels[:, 0:4], labels[:, 4:6]], None), is_train=True)
        fake_image = decoder.get_outputs()[0]
        discri_cls_label = label_broad(labels[:, 0:4])
        discri_angle_label = labels[:, 4:6]

        # collect fake's grad
        discri.forward(DataBatch([fake_image, mx.nd.zeros((batch_size, 1)), discri_cls_label, discri_angle_label]), is_train=True)
        discri.backward()
        gradDiscri = [[grad.copyto(grad.context) for grad in grads] for grads in discri._exec_group.grad_arrays]
        # collect real's grad
        discri.forward(DataBatch([real_image, mx.nd.ones((batch_size, 1)), discri_cls_label, discri_angle_label]), is_train=True)
        discri.backward()
        for gradsr, gradsf in zip(discri._exec_group.grad_arrays, gradDiscri):
            for gradr, gradf in zip(gradsr, gradsf):
                gradr += gradf
        discri.update()
        break
        # =================================================================================== #
        #                                 2. Train the decoder                                #
        # =================================================================================== #
        # =================================================================================== #
        #                                 3. Miscellaneous                                    #
        # =================================================================================== #