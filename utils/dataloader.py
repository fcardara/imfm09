import numpy as np
import tensorflow as tf
import cv2
from sklearn.model_selection import train_test_split

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, images, categories, batch_size=32, shuffle=True, normalize=False, resize_to=None, **kwargs):
        super().__init__(**kwargs)
        self.images = images
        self.categories = categories
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.normalize = normalize
        self.resize_to = resize_to
        self.indexes = np.arange(len(self.images))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.ceil(len(self.images) / self.batch_size))

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_images = self.images[batch_indexes]

        if self.resize_to:
            resized_batch = []
            for img in batch_images:
                resized = cv2.resize(img, self.resize_to[::-1])  # OpenCV expects (width, height)
                resized_batch.append(resized)
            batch_images = np.array(resized_batch)
        
        if self.normalize:
            batch_images = batch_images.astype('float32') / 255.0
            
        batch_labels = self.categories[batch_indexes]
        return batch_images, batch_labels

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

class PreprocessedDataGenerator(DataGenerator):
    def __init__(self, *args, preprocess_fn=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.preprocess_fn = preprocess_fn

    def __getitem__(self, index):
        batch_images, batch_labels = super().__getitem__(index)
        if self.preprocess_fn is not None:
            batch_images = self.preprocess_fn(batch_images)
        return batch_images, batch_labels


def load_data_npy(image_path='images.npy', label_path='categories.npy',
                  test_size=0.25, val_size=0.20, random_state=42, resize_to=None):
    images = np.load(image_path, mmap_mode='r')
    categories = np.load(label_path, mmap_mode='r')

    if resize_to is not None:
        resized_images = []
        for img in images:
            resized = cv2.resize(img, resize_to[::-1])  # OpenCV usa (width, height)
            resized_images.append(resized)
        images = np.array(resized_images)

    images_train_full, images_test, categories_train_full, categories_test = train_test_split(
        images, categories, test_size=test_size, random_state=random_state, stratify=categories)

    images_train, images_val, categories_train, categories_val = train_test_split(
        images_train_full, categories_train_full, test_size=val_size, random_state=random_state, stratify=categories_train_full)

    return images_train, categories_train, images_val, categories_val, images_test, categories_test
