import mxnet as mx

def label_broad(batch_size, labels):
    # labels [B, 4]
    labels[:, 1] += 4
    labels[:, 2] += 9
    labels[:, 3] += 15
    temp = mx.nd.zeros((batch_size, 17))
    for i in range(4):
        temp[mx.nd.arange(batch_size), labels[:, i]] = 1
    return temp