from typing import Iterable, Tuple

import numpy as np
from gymnasium import spaces


def concatenate_categorical_data_to_images(images, categoricals):
    """
    Concatenate categorical data to images.
    Args:
        images (np.ndarray): A batch of images.
        categoricals (np.ndarray): A batch of categorical data.
    Returns:
        A batch of images with the categorical data concatenated as extra channels. The extra channels are just the
        number of the category at each pixel, so they are not one-hot encoded.
    """
    _, height, width = images.shape[1:]
    extra_channels = np.broadcast_to(
        categoricals[:, :, np.newaxis, np.newaxis], (categoricals.shape[0], categoricals.shape[1], height, width)
    )
    return np.concatenate([images, extra_channels], axis=1)


def separate_image_and_categorical_state(
    state: np.ndarray,
    categorical_spaces: Iterable[spaces.Discrete],
) -> Tuple[np.ndarray, Iterable[np.ndarray]]:
    """
    Separate the image and categorical components of the state.
    Args:
        state: A preprocessed batch of states.
        categorical_spaces: The spaces of the categorical components of the state.
    Returns:
        A tuple of (image, categorical), where `image` is a batch of images and `categorical` is a list of batches
        of categorical data, one-hot encoded.
    """
    _, total_channels, _, _ = state.shape
    image_channels = total_channels - len(categorical_spaces)
    image = state[:, :image_channels]
    categorical = []
    for i, space in enumerate(categorical_spaces):
        category_values = state[:, image_channels + i, 0, 0]  # Smeared across all pixels; just take one.
        categorical.append(np.eye(space.n)[category_values])
    return image, categorical
