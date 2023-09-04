import os
import tensorflow as tf


def get_image_paths_and_labels(data_dir, samples):
    """
    Get image paths and labels for each image in the dataset.
    :param samples: Samples of the dataset.
    :return: List of image paths and list of image labels.
    """
    image_paths = []
    corrected_labels = []

    for sample in samples:
        line_split = sample.strip().split(' ')

        # Each line split has the following format:
        # part1/part1-part2/part1-part2-part3.png 
        image_name = line_split[0]
        part1 = image_name.split('-')[0]
        part2 = image_name.split('-')[1]
        img_path = os.path.join(data_dir, part1, part1 + '-' + part2, image_name + '.png')

        if os.path.exists(img_path):
            image_paths.append(img_path)
            corrected_labels.append(sample.strip('\n'))

    return image_paths, corrected_labels


def clean_labels(labels):
    """
    Clean labels by removing additional information.
    :param labels: List of labels.
    :return: Cleaned list of labels.
    """
    cleaned_labels = []
    for label in labels:
        cleaned_labels.append(label.split()[-1].strip())

    return cleaned_labels


def distortion_free_resize(image, img_size):
    """
    Resize image without distortion.
    
    Args:
        image: Image to resize.
        img_size: Size of the image.

    Returns:
        Resized image.
    """
    w, h = img_size
    image = tf.image.resize(image, (h, w), preserve_aspect_ratio=True)

    # Check padding
    pad_height = h - tf.shape(image)[0]
    pad_width = w - tf.shape(image)[1]

    # Pad image
    if pad_height % 2 != 0:
        height = pad_height // 2
        pad_height_top = height + 1
        pad_height_bottom = height
    else:
        pad_height_top = pad_height_bottom = pad_height // 2

    if pad_width % 2 != 0:
        width = pad_width // 2
        pad_width_left = width + 1
        pad_width_right = width
    else:
        pad_width_left = pad_width_right = pad_width // 2
    
    image = tf.pad(image, [[pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right], [0, 0]])

    image = tf.transpose(image, perm=[1, 0, 2])
    image = tf.image.flip_left_right(image)
    return image


def preprocess_image(image_path, img_dimensions):
    """
    Resize image to required dimensions with distortion and normalize pixel values.

    Args:
        image_path: Path to image.
        img_dimensions: Dimensions to resize image to.

    Returns:
        Preprocessed image.
    """
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=1)

    image = distortion_free_resize(image, img_dimensions)
    image = tf.cast(image, tf.float32) / 255.0 # type: ignore
    return image


def vectorize_label(label, vectorizer, max_length, padding_token=0):
    """
    Vectorize label using vectorizer.

    Args:
        label: Label to vectorize.
        vectorizer: Vectorizer to use.

    Returns:
        Vectorized label.
    """
    label = vectorizer(tf.strings.unicode_split(label, 'UTF-8'))
    length = tf.shape(label)[0]

    pad_amount = max_length - length

    label = tf.pad(label, paddings=[[0, pad_amount]], constant_values=padding_token)
    return label


def process_image_label(image_path, label, img_dimensions, vectorizer, max_length, padding_token=0):
    """
    Process image and label.

    Args:
        image_path: Path to image.
        img_dimensions: Dimensions to resize image to.
        label: Label to vectorize.
        vectorizer: Vectorizer to use.
        max_length: Maximum length of label.
        padding_token: Padding token to use.
    
    Returns:
        Dictionary containing processed image and label.
    """
    image = preprocess_image(image_path, img_dimensions)
    label = vectorize_label(label, vectorizer, max_length, padding_token)
    return {'image': image, 'label': label}
