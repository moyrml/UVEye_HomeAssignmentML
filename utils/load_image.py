import cv2


def load_image(image_path, scale_to=None):
    img = cv2.imread(str(image_path))
    assert img is not None, f'None image for {image_path}'

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if scale_to is not None:
        # EDA showed we have square images of various sizes
        img = cv2.resize(img, (scale_to, scale_to))

    return img
