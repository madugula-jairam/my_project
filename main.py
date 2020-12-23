import numpy as np
import os
import tensorflow as tf2
sr = tf2.data.experimental.shuffle_and_repeat
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import cv2
import glob
from copy import deepcopy
import model
import matplotlib.pyplot as plt

input_dir = 'input_imgs/'
image_type = '.png'
image_scene = 'reflection'
batch_size = 1

path = input_dir + image_scene + '*' + image_type
list_of_images = glob.glob(path)
print(list_of_images)

CROP_PATCH_H = 336
CROP_PATCH_W = 448

def _read_image_random_size(filename):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_png(image_string, channels=3)
    image_decoded.set_shape([CROP_PATCH_H, CROP_PATCH_W, 3])
    x = tf.cast(image_decoded, dtype=tf.float32) / 255.0
    # x -> Tensor("truediv:0", dtype=float32)
    return x
def resize_and_save(img_path):
    original_img = cv2.imread(img_path)
    print('Original image size :',original_img.shape)
    NEW_H = int(np.ceil(float(original_img.shape[0]) / 16.0)) * 16
    NEW_W = int(np.ceil(float(original_img.shape[1]) / 16.0)) * 16
    new_img = cv2.resize(original_img, dsize=(NEW_W, NEW_H), interpolation=cv2.INTER_CUBIC)
    print('new image size :',new_img.shape)
    new_path = os.path.join('resized_images', os.path.split(img_path)[-1])
    cv2.imwrite(new_path, new_img)
    return new_path
def to_tensor_data(path):
    #['resized_images\\reflection_I1.png']
    dataset_F0 = tf.data.Dataset.from_tensor_slices(tf.constant(['path']))
    #<DatasetV1Adapter shapes: (), types: tf.string>
    dataset_F0 = dataset_F0.apply(sr(buffer_size=21, count=None, seed=6)).map(
        _read_image_random_size)
    dataset_F0 = dataset_F0.prefetch(16)
    #<DatasetV1Adapter shapes: (336, 448, 3), types: tf.float32>
    return dataset_F0
if not os.path.exists('resized_images'):
    os.makedirs('resized_images')
path = []
fused_frame = []
for img_path in list_of_images:
    new_path = resize_and_save(img_path)
    path.append(new_path)
print(path)
for img_path in path:
    dataset = to_tensor_data(img_path)
    batch = dataset.batch(batch_size).make_initializable_iterator()
    fused_frame.append(batch.get_next())
print(fused_frame)

"""initialflow_decomposition"""
model1 = model.Flow_Decomposition(CROP_PATCH_H // 16, CROP_PATCH_W // 16)
Flow = model1.build_model(fused_frame)
print(Flow)


arr_ = np.squeeze(Flow[0])
plt.imshow(arr_)
plt.show()