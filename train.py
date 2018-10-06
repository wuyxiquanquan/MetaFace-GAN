import mxnet as mx
from tensorboardX import SummaryWriter
from network.net import QuanDecoder, QuanDiscr, QuanLoss
import os
from DataIters.FRVTImageIter import FRVTImageIter
from tqdm import tqdm
import time
import numpy as np

np.set_printoptions(suppress=True)

batch_size = 100

# --------- multicard ---------
ctx = []
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
                                  data_names=['data'], label_names=None,
                                  context=ctx, )
fr_module.bind([('data', (1, 3, 112, 112))], None, for_training=True, inputs_need_grad=True)

train_dataiter = FRVTImageIter(10, '/data1/ijb/IJB/IJB-C/protocols/ijbc_metadata.csv')
train_dataiter = mx.io.PrefetchingIter(train_dataiter)

for epoch in range(1):
    train_dataiter.reset()
    for cur_time, databatch in enumerate((train_dataiter)):
        real_img, labels = databatch.data
        print(labels.asnumpy())
        time.sleep(1)
