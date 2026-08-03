import numpy as np
from tensorflow.keras.preprocessing import image


def preprocess_image(img_path):
    """
    Load image, resize it, normalize it,
    and prepare it for prediction.
    """

    # Load image
    img = image.load_img(img_path, target_size=(128, 128))

    # Convert image to array
    img_array = image.img_to_array(img)

    # Normalize pixel values
    img_array = img_array / 255.0

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    return img_array