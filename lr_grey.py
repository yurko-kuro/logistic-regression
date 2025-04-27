import numpy as np
import h5py
import time
import logging
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2


class CatImageClassifier:
    def __init__(self, image_path=None):
        self.image_path = image_path
        self.num_px = None
        self.train_set_x = None
        self.train_set_y = None
        self.test_set_x = None
        self.test_set_y = None
        self.classes = None
        self.w = None
        self.b = None
        self.start_time = time.time()

        # Configure logging
        logging.basicConfig(
            filename='model_training_grey.log',  # Log file name
            level=logging.INFO,  # Logging level
            format='%(asctime)s - %(levelname)s - %(message)s'  # Log format
        )

    def rgb2gray(self, rgb):
        """Convert RGB image to grayscale."""
        logging.info("Converting RGB image to grayscale.")
        return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

    def load_dataset_gray(self):
        """Load dataset and convert images to grayscale."""
        logging.info("Loading the dataset and converting images to grayscale.")

        train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
        train_set_x_orig = np.array(train_dataset["train_set_x"][:])
        train_set_y_orig = np.array(train_dataset["train_set_y"][:])

        test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
        test_set_x_orig = np.array(test_dataset["test_set_x"][:])
        test_set_y_orig = np.array(test_dataset["test_set_y"][:])

        classes = np.array(test_dataset["list_classes"][:])

        # Convert images to grayscale
        train_set_x_gray = np.array([self.rgb2gray(img) for img in train_set_x_orig])
        test_set_x_gray = np.array([self.rgb2gray(img) for img in test_set_x_orig])

        # Reshape labels
        train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
        test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

        logging.info("Dataset loaded and converted to grayscale.")

        return train_set_x_gray, train_set_y_orig, test_set_x_gray, test_set_y_orig, classes

    def preprocess_data(self):
        """Preprocess the data."""
        logging.info("Preprocessing the data.")

        train_set_x_gray, train_set_y, test_set_x_gray, test_set_y, classes = self.load_dataset_gray()

        self.num_px = train_set_x_gray[0].shape[0]

        # Flatten the datasets and normalize the pixel values
        m_train = train_set_x_gray.shape[0]
        m_test = test_set_x_gray.shape[0]

        train_set_x_flatten = train_set_x_gray.reshape(m_train, -1).T
        test_set_x_flatten = test_set_x_gray.reshape(m_test, -1).T

        self.train_set_x = train_set_x_flatten / 255.
        self.test_set_x = test_set_x_flatten / 255.

        self.train_set_y = train_set_y
        self.test_set_y = test_set_y

        logging.info("Data preprocessing complete.")

    def sigmoid(self, x):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-x))

    def initialize_with_zeros(self, dim):
        """Initialize weights and bias with zeros."""
        w = np.zeros([dim, 1])
        b = 0
        return w, b

    def propagate(self, w, b, X, Y):
        """Forward and backward propagation."""
        m = X.shape[1]
        A = self.sigmoid(np.dot(w.T, X) + b)
        cost = -np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A)) / m

        dw = np.dot(X, (A - Y).T) / m
        db = np.sum(A - Y) / m

        cost = np.squeeze(cost)
        grads = {"dw": dw, "db": db}

        logging.debug(f"Cost after propagation: {cost:.5f}")

        return grads, cost

    def optimize(self, w, b, X, Y, num_iterations, learning_rate, print_cost=False):
        """Gradient descent optimization."""
        costs = []

        for i in range(num_iterations):
            grads, cost = self.propagate(w, b, X, Y)
            dw = grads["dw"]
            db = grads["db"]

            w = w - learning_rate * dw
            b = b - learning_rate * db

            if i % 100 == 0:
                costs.append(cost)
                if print_cost:
                    logging.info(f"Cost after iteration {i}: {cost:.5f}")
                    print(f"Cost after iteration {i}: {cost:.5f}")

        parameters = {"w": w, "b": b}
        grads = {"dw": dw, "db": db}
        return parameters, grads, costs

    def predict(self, w, b, X):
        """Make predictions using learned weights and bias."""
        m = X.shape[1]
        Y_prediction = np.zeros((1, m))
        w = w.reshape(X.shape[0], 1)

        A = self.sigmoid(np.dot(w.T, X) + b)

        for i in range(A.shape[1]):
            Y_prediction[0][i] = 1 if A[0][i] > 0.5 else 0

        return Y_prediction

    def model(self, num_iterations=2000, learning_rate=0.5, print_cost=False):
        """Train the logistic regression model."""
        w, b = self.initialize_with_zeros(self.train_set_x.shape[0])

        parameters, grads, costs = self.optimize(w, b, self.train_set_x, self.train_set_y, num_iterations,
                                                 learning_rate, print_cost)

        self.w = parameters["w"]
        self.b = parameters["b"]

        Y_prediction_test = self.predict(self.w, self.b, self.test_set_x)
        Y_prediction_train = self.predict(self.w, self.b, self.train_set_x)

        train_accuracy = 100 - np.mean(np.abs(Y_prediction_train - self.train_set_y)) * 100
        test_accuracy = 100 - np.mean(np.abs(Y_prediction_test - self.test_set_y)) * 100

        logging.info(f"Train accuracy: {train_accuracy}%")
        logging.info(f"Test accuracy: {test_accuracy}%")

        d = {
            "costs": costs,
            "Y_prediction_test": Y_prediction_test,
            "Y_prediction_train": Y_prediction_train,
            "w": self.w,
            "b": self.b,
            "learning_rate": learning_rate,
            "num_iterations": num_iterations
        }
        return d

    def save_model(self):
        """Save model weights to file."""
        np.savez('model_weights_grey.npz', w=self.w, b=self.b)
        logging.info("Weights saved!")

    def predict_image(self, image_path):
        """Classify an image."""
        img = mpimg.imread(image_path)
        gray = self.rgb2gray(img)

        gray_resized = cv2.resize(gray, (self.num_px, self.num_px)).reshape((1, self.num_px * self.num_px)).T
        gray_resized = gray_resized / 255.0

        image_predict = self.predict(self.w, self.b, gray_resized)
        logging.info(f"Prediction for the image: {image_predict}")

        if (image_predict == 1).item():
            logging.info("The image contains a cat.")
            print('The image contains a cat.')
        else:
            logging.info("The image does not contain a cat.")
            print('The image does not contain a cat.')

        # Display the image
        plt.imshow(gray, cmap=plt.get_cmap('gray'))
        plt.show()

    def run(self, image_path):
        """Execute the full pipeline: Load, preprocess, train, predict."""
        self.image_path = image_path
        self.preprocess_data()
        self.model(num_iterations=2000, learning_rate=0.005, print_cost=True)
        self.save_model()
        self.predict_image(image_path)
        logging.info(f"Script execution completed in {time.time() - self.start_time} seconds.")


# Example usage:
image_path = 'images/photo_6.jpg'
classifier = CatImageClassifier(image_path)
classifier.run(image_path)
