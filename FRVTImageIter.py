import mxnet as mx
from pandas import read_csv
import cv2


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

    def __len__(self):
        return len(self.dataset)

    def read_image(self, path):
        path = '/data1/ijb/IJB/IJB-C/images/' + path
        img = cv2.imread(path)[:, :, ::-1]
        img = cv2.resize(img, (112, 112))
        img = mx.nd.array(img).transpose(axes=(2, 0, 1))
        return img

    def reset(self):
        """Resets the iterator to the beginning of the data."""
        self.cur = -1
        print('IJB-C dataset loading is complete!')

    def next(self):
        """
            image, label
        """
        imgs = [mx.nd.zeros((self.batch_size, 3, 112, 112))]
        label = mx.nd.zeros((self.batch_size,))
        labels = [label] * 6
        for i in range(self.batch_size):
            self.cur += 1
            imgs[0][i] = self.read_image(self.dataset.FILENAME[self.cur + 1])
            labels[0][i] = self.dataset.FACIAL_HAIR[self.cur + i]
            labels[1][i] = self.dataset.AGE[self.cur + i]
            labels[2][i] = self.dataset.SKINTONE[self.cur + i]
            labels[2][i] = self.dataset.GENDER[self.cur + i]
            labels[4][i] = self.dataset.YAW[self.cur + i]
            labels[5][i] = self.dataset.ROLL[self.cur + i]

        return mx.io.DataBatch(imgs, labels, )


if __name__ == '__main__':

    train_dataiter = FRVTImageIter(10, '/data1/ijb/IJB/IJB-C/protocols/ijbc_metadata.csv')
    train_dataiter = mx.io.PrefetchingIter(train_dataiter)
    from tqdm import tqdm

    for i, data in enumerate(tqdm(train_dataiter)):
        continue
