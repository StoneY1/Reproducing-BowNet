import numpy as np
import pandas as pd
import pickle
from scipy import misc
from tqdm import tqdm

def unpickle(file):
    with open(file, 'rb') as fo:
        res = pickle.load(fo, encoding='bytes')
    return res

meta = unpickle('cifar-100/meta')

fine_label_names = [t.decode('utf8') for t in meta[b'fine_label_names']]

train = unpickle('cifar-100/train')

filenames = [t.decode('utf8') for t in train[b'filenames']]
fine_labels = train[b'fine_labels']
data = train[b'data']

train_images = list()
for d in data:
    image = np.zeros((32,32,3), dtype=np.uint8)
    image[...,0] = np.reshape(d[:1024], (32,32)) # Red channel
    image[...,1] = np.reshape(d[1024:2048], (32,32)) # Green channel
    image[...,2] = np.reshape(d[2048:], (32,32)) # Blue channel
    train_images.append(image)


test = unpickle('cifar-100/test')

filenames = [t.decode('utf8') for t in test[b'filenames']]
fine_labels = test[b'fine_labels']
data = test[b'data']

test_images = list()
for d in data:
    image = np.zeros((32,32,3), dtype=np.uint8)
    image[...,0] = np.reshape(d[:1024], (32,32)) # Red channel
    image[...,1] = np.reshape(d[1024:2048], (32,32)) # Green channel
    image[...,2] = np.reshape(d[2048:], (32,32)) # Blue channel
    test_images.append(image)


train_images_90 = []
for img in train_images:
    rot_img = transform.rotate(img,90)
    rot_img = util.img_as_ubyte(rot_img)
    train_images_90.append(rot_img)

train_images_180 = []
for img in train_images:
    rot_img = transform.rotate(img,180)
    rot_img = util.img_as_ubyte(rot_img)
    train_images_180.append(rot_img)

train_images_270 = []
for img in train_images:
    rot_img = transform.rotate(img,270)
    rot_img = util.img_as_ubyte(rot_img)
    train_images_270.append(rot_img)


test_images_90 = []
for img in test_images:
    rot_img = transform.rotate(img,90)
    rot_img = util.img_as_ubyte(rot_img)
    test_images_90.append(rot_img)

test_images_180 = []
for img in test_images:
    rot_img = transform.rotate(img,180)
    rot_img = util.img_as_ubyte(rot_img)
    test_images_180.append(rot_img)

test_images_270 = []
for img in test_images:
    rot_img = transform.rotate(img,270)
    rot_img = util.img_as_ubyte(rot_img)
    test_images_270.append(rot_img)


with open('train_images.pkl', 'wb') as f:
    pickle.dump(train_images, f)

with open('train_images_90.pkl', 'wb') as f:
    pickle.dump(train_images_90, f)

with open('train_images_180.pkl', 'wb') as f:
    pickle.dump(train_images_180, f)

with open('train_images_270.pkl', 'wb') as f:
    pickle.dump(train_images_270, f)

with open('test_images.pkl', 'wb') as f:
    pickle.dump(test_images, f)

with open('test_images_90.pkl', 'wb') as f:
    pickle.dump(test_images_90, f)

with open('test_images_180.pkl', 'wb') as f:
    pickle.dump(test_images_180, f)

with open('test_images_270.pkl', 'wb') as f:
    pickle.dump(test_images_270, f)

print("Done")    
