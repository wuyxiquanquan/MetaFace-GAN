import mxnet as mx
from pandas import read_csv
import cv2
from math import isnan


class FRVTImageIter(mx.io.DataIter):
    def __init__(self, batch_size, path):
        super().__init__()
        self.batch_size = batch_size
        self.cur = -1
        self.dataset = read_csv(path)
        self.provide_data = [('data', (batch_size, 3, 112, 112))]
        self.provide_label = [
            ('Facial_hair', (batch_size,)),
            ('Age', (batch_size,)),
            ('Skintone', (batch_size,)),
            ('Gender', (batch_size,)),
            ('Yaw', (batch_size,)),
            ('Row', (batch_size,)),
        ]

    def read_image(self, i):

        path = '/data1/ijb/IJB/IJB-C/images/' + self.dataset.FILENAME[i]
        y, x = int(self.dataset.FACE_Y[i]), int(self.dataset.FACE_X[i])
        h, w = int(self.dataset.FACE_HEIGHT[i]), int(self.dataset.FACE_WIDTH[i])
        center_y = h // 2 + y
        center_x = w // 2 + x
        big = max(h, w) // 2
        y_ = y if center_y - big < 0 else center_y - big
        x_ = x if center_x - big < 0 else center_x - big
        img = cv2.resize(cv2.imread(path)[y_: center_y + big, x_: center_x + big, ::-1], (112, 112))
        img = mx.nd.array(img).transpose(axes=(2, 0, 1))
        return img

    def reset(self):
        """Resets the iterator to the beginning of the data."""
        self.cur = -1
        print('IJB-C dataset resetting is complete!')

    def next(self):
        """
            data: [(B, 3, 112, 112), (B, 6)]
        """
        combined_data = [mx.nd.zeros((self.batch_size, 3, 112, 112), dtype='int32'),
                         mx.nd.zeros((self.batch_size, 6))]

        i = 0
        while i < 8:
            self.cur += 1
            if self.dataset.FILENAME[self.cur].split('/')[0] == 'frames' or self.dataset.FACE_HEIGHT[self.cur] < 100 or \
                    self.dataset.FACE_WIDTH[self.cur] < 100 or isnan(self.dataset.FACIAL_HAIR[self.cur]) or isnan(
                self.dataset.AGE[self.cur]) or isnan(self.dataset.SKINTONE[self.cur]) or isnan(
                self.dataset.GENDER[self.cur]):
                continue
            combined_data[0][i] = self.read_image(self.cur)
            combined_data[1][i][0] = self.dataset.FACIAL_HAIR[self.cur]
            combined_data[1][i][1] = self.dataset.AGE[self.cur]
            combined_data[1][i][2] = self.dataset.SKINTONE[self.cur] - 1
            combined_data[1][i][3] = self.dataset.GENDER[self.cur]
            combined_data[1][i][4] = self.dataset.YAW[self.cur]
            combined_data[1][i][5] = self.dataset.ROLL[self.cur]
            i += 1
        return mx.io.DataBatch(combined_data, [], )


if __name__ == '__main__':

    train_dataiter = FRVTImageIter(10, '/data1/ijb/IJB/IJB-C/protocols/ijbc_metadata.csv')
    train_dataiter = mx.io.PrefetchingIter(train_dataiter)

    for i, data in enumerate(train_dataiter):
        break
