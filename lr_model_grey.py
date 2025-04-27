import numpy as np
import cv2
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import logging


class CatImagePredictor:
    def __init__(self, image_path=None):
        self.image_path = image_path
        self.num_px = 64  # Image size
        self.w = None
        self.b = None

        # Configure logging
        logging.basicConfig(
            filename='model_prediction_grey.log',  # Log file name
            level=logging.INFO,  # Logging level
            format='%(asctime)s - %(levelname)s - %(message)s'  # Log format
        )

    def rgb2gray(self, rgb):
        """Convert RGB image to grayscale."""
        logging.info("Converting RGB image to grayscale.")
        return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

    def sigmoid(self, x):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-x))

    def predict(self, w, b, X):
        """Make predictions using learned weights and bias."""
        m = X.shape[1]
        A = self.sigmoid(np.dot(w.T, X) + b)
        Y_prediction = np.zeros((1, m))

        for i in range(A.shape[1]):
            Y_prediction[0][i] = 1 if A[0][i] > 0.5 else 0
        return Y_prediction

    def load_model(self, model_path='model_weights_grey.npz'):
        """Load model weights from a file."""
        logging.info("Loading model weights from file.")
        weights = np.load(model_path)
        self.w = weights['w']
        self.b = weights['b']
        logging.info("Model weights loaded successfully.")

    def preprocess_image(self, image_path):
        """Preprocess the input image."""
        logging.info(f"Loading and preprocessing image: {image_path}")
        img = imageio.imread(image_path)

        # Convert to grayscale
        gray = self.rgb2gray(img)

        # Resize image and normalize
        gray_resized = cv2.resize(gray, (self.num_px, self.num_px)).reshape((1, self.num_px * self.num_px)).T
        gray_resized = gray_resized / 255.0

        logging.info("Image preprocessing complete.")

        return gray_resized, gray

    def predict_image(self, image_path):
        """Classify the image and print the result."""
        # Load and preprocess image
        gray_resized, gray = self.preprocess_image(image_path)

        # Make prediction
        image_predict = self.predict(self.w, self.b, gray_resized)
        logging.info(f"Prediction for the image: {image_predict}")

        # Display the result
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
        """Run the full pipeline: Load model, preprocess image, predict."""
        logging.info(f"Starting prediction for image: {image_path}")
        self.load_model()  # Load model weights
        self.predict_image(image_path)  # Make prediction
        logging.info("Prediction process completed.")


# Example usage:
image_path = 'images/photo_2.jpg'
predictor = CatImagePredictor(image_path)
predictor.run(image_path)
