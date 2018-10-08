import mxnet as mx
from tensorboardX import SummaryWriter
from network.net import QuanDecoder, QuanDiscr, QuanLoss
import os
from DataIters.FRVTImageIter import FRVTImageIter
from tqdm import tqdm
import time
import numpy as np
from mxnet.io import DataBatch
from parse_arg import parse_args
from ops import label_broad

args = parse_args()
np.set_printoptions(suppress=True)

# --------- multicard ---------
cvd = os.environ['GPU'].strip()
args.ctx = []
for i in range(len(cvd.split(','))):
    args.ctx.append(mx.gpu(i))
print('GPU num:', len(args.ctx))


args.batch_size = args.per_batch * len(args.ctx)
_rescale = 1.0 / len(args.ctx)
# -----create Module----
# 1. Module
# 2. bind
# 3. init_params
# 4. init_optimizer
# --------- DECODER ---------
decoder = mx.mod.Module(symbol=QuanDecoder(),
                        data_names=['decoder_real_vector', 'decoder_cls_label', 'decoder_angle_label'],
                        label_names=None,
                        context=args.ctx, )
decoder.bind(data_shapes=[('decoder_real_vector', (args.batch_size, 512)), ('decoder_cls_label', (args.batch_size, 4)),
                          ('decoder_angle_label', (args.batch_size, 2))],
             label_shapes=None,
             for_training=True, inputs_need_grad=False)
decoder.init_params(initializer=mx.init.Xavier(),
                    allow_missing=True)
decoder.init_optimizer(optimizer='adam',
                       optimizer_params={
                           'learning_rate': 1e-4,
                           'beta1': 0.5,
                           'rescale_grad': _rescale,
                       })

# --------- DISCRIMINATOR ---------
discri = mx.mod.Module(symbol=QuanDiscr(),
                       data_names=['discri_image', 'discri_gan_label', 'discri_cls_label', 'discri_angle_label'],
                       label_names=None,
                       context=args.ctx, )
discri.bind(data_shapes=[('discri_image', (args.batch_size, 3, 112, 112)), ('discri_gan_label', (args.batch_size, 1)),
                         ('discri_cls_label', (args.batch_size, 17)), ('discri_angle_label', (args.batch_size, 2))],
            label_shapes=None,
            for_training=True, inputs_need_grad=True)
discri.init_params(initializer=mx.init.Xavier(),
                   allow_missing=True)
discri.init_optimizer(optimizer='adam',
                      optimizer_params={
                          'learning_rate': 1e-4,
                          'beta1': 0.5,
                          'rescale_grad': _rescale,
                      })

# --------- LossModule ---------
loss_module = mx.mod.Module(symbol=QuanLoss(),
                            data_names=['real_vector', 'fake_vector', 'real_image', 'fake_image'],
                            label_names=None,
                            context=args.ctx, )
loss_module.bind(data_shapes=[('real_vector', (args.batch_size, 512)), ('fake_vector', (args.batch_size, 512)),
                              ('real_image', (args.batch_size, 3, 112, 112)),
                              ('fake_image', (args.batch_size, 3, 112, 112))],
                 label_shapes=None,
                 for_training=True, inputs_need_grad=True)
loss_module.init_params(initializer=mx.init.Xavier(),
                        allow_missing=True)

# --------- fixed FR net ---------
fr_module = mx.module.Module.load(args.fr_path.split(',')[0],int(args.fr_path.split(',')[1]),
                                  data_names=['data', ], label_names=None,
                                  context=args.ctx, )
fr_module.bind([('data', (args.batch_size, 3, 112, 112))], None, for_training=True, inputs_need_grad=True)
fr_module.init_optimizer(optimizer='adam',
                         optimizer_params={
                             'learning_rate': 1e-4,
                             'beta1': 0.5,
                             'rescale_grad': _rescale,
                         })

train_dataiter = FRVTImageIter(args.batch_size, args.csv_path)
train_dataiter = mx.io.PrefetchingIter(train_dataiter)

# training
for epoch in range(args.epoch):
    train_dataiter.reset()
    for cur_time, databatch in enumerate(tqdm(train_dataiter)):
        real_image, labels = databatch.data
        # print(labels.asnumpy())
        # time.sleep(1)
        # =================================================================================== #
        #                             1. Train the discriminator                              #
        # =================================================================================== #
        # discriminator data preparing
        fr_module.forward(DataBatch([real_image], None), is_train=False)
        real_vector = fr_module.get_outputs()[0]
        decoder.forward(DataBatch([real_vector, labels[:, 0:4], labels[:, 4:6]], None), is_train=True)
        fake_image = decoder.get_outputs()[0]
        discri_cls_label = label_broad(args.batch_size, labels[:, 0:4])
        discri_angle_label = labels[:, 4:6]
        # update
        # collect fake's grad
        discri.forward(
            DataBatch([fake_image, mx.nd.zeros((args.batch_size, 1)), discri_cls_label, discri_angle_label], None),
            is_train=True)
        loss1 = discri.get_outputs()[0]
        discri.backward()
        gradDiscri = [[grad.copyto(grad.context) for grad in grads] for grads in discri._exec_group.grad_arrays]
        # collect real's grad
        discri.forward(
            DataBatch([real_image, mx.nd.ones((args.batch_size, 1)), discri_cls_label, discri_angle_label], None),
            is_train=True)
        discri.backward()
        for gradsr, gradsf in zip(discri._exec_group.grad_arrays, gradDiscri):
            for gradr, gradf in zip(gradsr, gradsf):
                gradr += gradf
        discri.update()
        # =================================================================================== #
        #                                 2. Train the decoder                                #
        # =================================================================================== #
        # decoder data preparing
        decoder_real_vector = real_vector
        decoder_cls_label = labels[:, 0:4]
        decoder_angle_label = labels[:, 4:6]
        fr_module.forward(DataBatch([fake_image], None), is_train=True)
        fake_vector = fr_module.get_outputs()[0]
        # loss_mdoule
        loss_module.forward(DataBatch([real_vector, fake_vector, real_image, fake_image], None), is_train=True)
        loss2 = loss_module.get_outputs()[0]
        loss_module.backward()
        backward_grad = loss_module.get_input_grads()
        #   fake_image
        decoder.backward(backward_grad[3:])
        gradDecoder = [[grad.copyto(grad.context) for grad in grads] for grads in decoder._exec_group.grad_arrays]
        fr_module.backward(backward_grad[1:2])
        decoder.backward(fr_module.get_input_grads())
        for gradsr, gradsf in zip(decoder._exec_group.grad_arrays, gradDecoder):
            for gradr, gradf in zip(gradsr, gradsf):
                gradf += gradr
        # discri
        discri.forward(
            DataBatch([fake_image, mx.nd.ones((args.batch_size, 1)), discri_cls_label, discri_angle_label], None),
            is_train=True)
        discri.backward()
        decoder.backward(discri.get_input_grads()[0:1])
        for gradsr, gradsf in zip(decoder._exec_group.grad_arrays, gradDecoder):
            for gradr, gradf in zip(gradsr, gradsf):
                gradr += gradf
        decoder.update()
        # =================================================================================== #
        #                                 3. Miscellaneous                                    #
        # =================================================================================== #
        loss = loss1 + loss2
        # print(loss)
