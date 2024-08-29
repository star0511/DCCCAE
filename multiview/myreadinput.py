import scipy.io as sio
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import random
import gzip
def load_pickle(f):
    try:
        import cPickle as thepickle
    except ImportError:
        import _pickle as thepickle

    try:
        ret = thepickle.load(f, encoding='latin1')
    except TypeError:
        ret = thepickle.load(f)

    return ret
class DataSet(object):
    
    def __init__(self, images1, images2,y, labels, fake_data=False, one_hot=False,
                 dtype=tf.float32):
        """Construct a DataSet.
        one_hot arg is used only if fake_data is true.  `dtype` can be either
        `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
        `[0, 1]`.
        
        """
        dtype = tf.as_dtype(dtype).base_dtype
        if dtype not in (tf.uint8, tf.float32):
            raise TypeError('Invalid image dtype %r, expected uint8 or float32' % dtype)

        if fake_data:
            self._num_examples = 10000
            self.one_hot = one_hot
        else:
            assert images1.shape[0] == labels.shape[0], (
                'images1.shape: %s labels.shape: %s' % (images1.shape,
                                                        labels.shape))
            assert images2.shape[0] == labels.shape[0], (
                'images2.shape: %s labels.shape: %s' % (images2.shape,
                                                        labels.shape))
            self._num_examples = images1.shape[0]
            # Convert shape from [num examples, rows, columns, depth]
            # to [num examples, rows*columns] (assuming depth == 1)
            #assert images.shape[3] == 1
            #images = images.reshape(images.shape[0],
            #                        images.shape[1] * images.shape[2])
            if dtype == tf.float32 and images1.dtype != np.float32:
                # Convert from [0, 255] -> [0.0, 1.0].
                images1 = images1.astype(np.float32)

            if dtype == tf.float32 and images2.dtype != np.float32:
                images2 = images2.astype(np.float32)

        self._images1 = images1
        self._images2 = images2
        self._labels = labels
        self._y = y
        self._epochs_completed = 0
        self._index_in_epoch = 0
    
    @property
    def images1(self):
        return self._images1
    
    @property
    def images2(self):
        return self._images2
    
    @property
    def labels(self):
        return self._labels
    
    @property
    def y(self):
        return self._y
    
    @property
    def num_examples(self):
        return self._num_examples
    
    @property
    def epochs_completed(self):
        return self._epochs_completed
    
    def next_batch(self, batch_size, fake_data=False):
        """Return the next `batch_size` examples from this data set."""
        if fake_data:
            fake_image = [1] * 2048
            if self.one_hot:
                fake_label = [1] + [0] * 9
            else:
                fake_label = 0
            return [fake_image for _ in xrange(batch_size)], [fake_image for _ in xrange(batch_size)], [fake_label for _ in xrange(batch_size)]
        
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images1 = self._images1[perm]
            self._images2 = self._images2[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        
        end = self._index_in_epoch
        return self._images1[start:end], self._images2[start:end], self._labels[start:end]

def read_dataset_train(filename1,filename2,lable):
    # yfrog 0 真实签名 1 熟练伪造签名
    data = np.load(filename1)
    x1, y, yforg = data.f.x,data.f.y,data.f.yforg
    data = np.load(filename2)
    x2, _, _= data.f.x,data.f.y,data.f.yforg
    if lable in [0,1]:
        yfrogindex = np.where(yforg==lable)
        x2 = x2[yfrogindex]
        x1,y,yforg = x1[yfrogindex], y[yfrogindex], yforg[yfrogindex]

    trainrate=0.8
    lenx = int(x1.shape[0]*trainrate)
    train = DataSet(x1[:lenx], x2[:lenx], y[:lenx].T, yforg[:lenx].T)
    tune = DataSet(x1[lenx:], x2[lenx:], y[lenx:].T, yforg[lenx:].T)
    test = ""
    return train,tune,test
def read_dataset(filename1,filename2):
    data = np.load(filename1)
    x1, y, yforg = data.f.x,data.f.y,data.f.yforg
    data = np.load(filename2)
    x2, _, _= data.f.x,data.f.y,data.f.yforg

    alldata = DataSet(x1, x2, y.T, yforg.T)
    return alldata

