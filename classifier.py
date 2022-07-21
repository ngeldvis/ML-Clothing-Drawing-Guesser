import tensorflow as tf
from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D
from keras.preprocessing import image
import numpy as np
import cv2 as cv
import os.path

from canvas import Canvas

print(tf.__version__)

LOOKUP_TABLE = {
    0: 'Shirt',
    1: 'Pants',
    2: 'Shirt',
    3: 'Dress',
    4: 'Coat',
    5: 'Shoe',
    6: 'Shirt',
    7: 'Shoe',
    8: 'Bag',
    9: 'Shoe'
}

class Classifier:

    def __init__(self) -> None:
        self.model = self.create_model()

    def create_model(self) -> Sequential:

        mnist = tf.keras.datasets.fashion_mnist
        (training_images, training_labels), (test_images, test_labels) = mnist.load_data()

        training_images  = training_images / 255.0
        test_images = test_images / 255.0

        model = Sequential([
            Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Flatten(),
            Dense(256, activation=tf.nn.relu),
            Dense(10, activation=tf.nn.softmax)
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(), 
            loss='sparse_categorical_crossentropy', 
            metrics=['accuracy']
        )

        model.fit(training_images, training_labels, epochs=7)

        return model

    def show_image(self, img):
        from matplotlib import pyplot as plt
        plt.imshow(img)
        plt.show()

    def predict(self, img):
        img_arr = image.image_utils.img_to_array(img)
        img_arr = np.expand_dims(img_arr, axis=0)
        # img_arr = training_images[500]

        results = self.model.predict(np.vstack([img_arr]), verbose=0)[0]
        index = np.where(results == np.amax(results))[0][0]

        return LOOKUP_TABLE[index]

    def predict_drawing(self, img):
        img = image.image_utils.load_img('images/image.png', color_mode='grayscale', target_size=(28, 28))
        return self.predict(img)

    def predict_file(self, image_file):
        img = image.image_utils.load_img(image_file, color_mode='grayscale', target_size=(28, 28))
        return self.predict(img)


def main() -> None:
    classifier = Classifier()
    # classifier.model.summary()

    while True:
        if input("'draw' or 'file': ") == 'draw':
            canvas = Canvas()
            img = canvas.draw_image()
            result = classifier.predict_drawing(img)
            print(result)

        else: 
            while True:
                file = input('please enter the path to the image file: ')
                if os.path.exists(file):
                    break
                else:
                    print("sorry, that file doesn't exist")

            result = classifier.predict_file(file)
            print(result)

        while True:
            another = input('would you like to try another image? ')
            if os.path.exists(another):
                result = classifier.predict_file(another)
                print(result)
            else:
                break

        if another not in ['yes', 'Yes', 'y', 'Y']:
            break

if __name__ == '__main__':
    main()
