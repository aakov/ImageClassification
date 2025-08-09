import scipy.io
from PIL import Image

class CarImage:
    def __init__(self, filename, label, bbox):
        self.filename = filename
        self.label = label  # Integer label (index of class)
        self.bbox = bbox  # (x1, y1, x2, y2)

    def __repr__(self):
        return f"CarImage(filename={self.filename}, label={self.label}, bbox={self.bbox})"

    def load_image(self):
        return Image.open(self.filepath).convert('RGB')

    def crop_to_bbox(self):
        img = self.load_image()
        x1, y1, x2, y2 = self.bbox
        return img.crop((x1, y1, x2, y2))

# File paths
META_PATH = "stanford-cars-dataset/versions/1/car_devkit/devkit/cars_meta.mat"
TRAIN_ANNOS_PATH = "stanford-cars-dataset/versions/1/car_devkit/devkit/cars_train_annos.mat"
TEST_ANNOS_PATH = "stanford-cars-dataset/versions/1/car_devkit/devkit/cars_test_annos.mat"


def load_class_names():
    meta = scipy.io.loadmat(META_PATH)
    return [c[0] for c in meta['class_names'][0]]


def load_train_images(mat_path):
    mat = scipy.io.loadmat(mat_path)
    annotations = mat['annotations'][0]
    images = []

    for ann in annotations:
        filename = ann['fname'][0]
        label = int(ann['class'][0][0]) - 1  # 0-based indexing
        bbox = (
            int(ann['bbox_x1'][0][0]),
            int(ann['bbox_y1'][0][0]),
            int(ann['bbox_x2'][0][0]),
            int(ann['bbox_y2'][0][0])
        )
        images.append(CarImage(filename, label, bbox))

    return images


def load_test_images(mat_path):
    mat = scipy.io.loadmat(mat_path)
    annotations = mat['annotations'][0]
    images = []

    for ann in annotations:
        filename = ann['fname'][0]
        label = None  # Test data doesn't have labels
        bbox = (
            int(ann['bbox_x1'][0][0]),
            int(ann['bbox_y1'][0][0]),
            int(ann['bbox_x2'][0][0]),
            int(ann['bbox_y2'][0][0])
        )
        images.append(CarImage(filename, label, bbox))

    return images

class_names = load_class_names()
print(f"Loaded {len(class_names)} classes")
print("First class name:", class_names[0])

train_images = load_train_images(TRAIN_ANNOS_PATH)
print(f"Loaded {len(train_images)} training images")
print("First image:", train_images[0])

test_images = load_test_images(TEST_ANNOS_PATH)
print(f"Loaded {len(test_images)} test images")
print("First test image:", test_images[0])
