import random
import cv2
from matplotlib import pyplot as plt
import albumentations as A


BOX_COLOR = (255, 0, 0)
TEXT_COLOR = (255, 255, 255) 

imgs_dir = "/cluster/home/ammaa/Downloads/Ammars/Models/Fracture_Detection/YOLOV9/data/Split-Authors/images/train/"
label_dir = "/cluster/home/ammaa/Downloads/Ammars/Models/Fracture_Detection/YOLOV9/data/Split-Authors/labels/train/"
output_imgs_dir = "/cluster/home/ammaa/Downloads/Ammars/Models/Fracture_Detection/YOLOV9/Augmented_Set/images"
output_labels_dir = "/cluster/home/ammaa/Downloads/Ammars/Models/Fracture_Detection/YOLOV9/Augmented_Set/labels"


# Helper Methods

def visualize_bbox(img, bbox, class_name, color=BOX_COLOR, thickness=2):
    """Visualizes a single bounding box on the image"""
    x_min, y_min, w, h = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)

    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35,
        color=TEXT_COLOR,
        lineType=cv2.LINE_AA,
    )
    return img


def yolo_to_coco(yolo_coords, image_width, image_height):
    x_center, y_center, width, height = yolo_coords
    x_min = max(0, (x_center - width / 2) * image_width)
    y_min = max(0, (y_center - height / 2) * image_height)
    width = width * image_width
    height = height * image_height
    
    return [x_min, y_min, width, height]


def coco_to_yolo(coco_coords, image_width, image_height):
    x_min, y_min, width, height = coco_coords
    x_center = (x_min + width / 2) / image_width
    y_center = (y_min + height / 2) / image_height
    width = width / image_width
    height = height / image_height
    
    return [x_center, y_center, width, height]
    
def get_image_size(image_path):
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    return width, height


def plot_image(img_dir, label_dir):
    image = cv2.imread(img_dir)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = image.copy()
    

    yolo_bbox = []

    with open(label_dir, 'rb') as label_f:
        for line in label_f:
            line = str(line)
            line = line.replace("\\n'", '')
            line = line.replace("b'", '')
            numbers = line.split(' ')
            x_center, y_center, w, h = float(numbers[1]), float(numbers[2]), float(numbers[3]), float(numbers[4])
            yolo_bbox = [x_center, y_center, w, h]


    height, width, _ = img.shape
    bboxes = [yolo_to_coco(yolo_bbox, width, height)]
    category_ids = [0]
    category_id_to_name = {0: 'fractured'}

    for bbox, category_id in zip(bboxes, category_ids):
        class_name = category_id_to_name[category_id]
        img = visualize_bbox(img, bbox, class_name)

    plt.figure(figsize=(12, 12))
    plt.axis('off')
    plt.imshow(img)


# Perform Augmentation

for i, img in enumerate(os.listdir(imgs_dir)):
    filename = os.path.splitext(os.path.basename(img))[0]
    label_path = os.path.join(label_dir, filename + ".txt")
    yolo_bbox = []

    with open(label_path, 'rb') as label_f:
        for line in label_f:
            line = str(line)
            line = line.replace("\\n'", '')
            line = line.replace("b'", '')
            numbers = line.split(' ')
            x_center, y_center, w, h = float(numbers[1]), float(numbers[2]), float(numbers[3]), float(numbers[4])
            yolo_bbox = [x_center, y_center, w, h]


    image = cv2.imread(os.path.join(imgs_dir, img))
    height, width, _ = image.shape
    coco_bbox = [yolo_to_coco(yolo_bbox, width, height)]
    category_ids = [0]
    category_id_to_name = {0: 'fractured'}
    

    transform = A.Compose(
    [
    A.MultiplicativeNoise(multiplier=[0.5, 1.5], elementwise=True, per_channel=True, p=1),
    A.HorizontalFlip(p=0.5),
      A.RandomRotate90(),
        A.Flip(),
        A.Transpose(),
        A.GaussNoise(),
        A.OneOf([
            A.MotionBlur(p=.2),
            A.MedianBlur(blur_limit=3, p=0.1),
            A.Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        A.OneOf([
            A.OpticalDistortion(p=0.3),
            A.GridDistortion(p=.1),
        ], p=0.2),
        A.OneOf([
            A.CLAHE(clip_limit=2),
            A.RandomBrightnessContrast(),
        ], p=0.3),
        A.HueSaturationValue(p=0.3),
    ],
    bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']),
)


    seed = random.seed(random.randint(0, 100))
    transformed = transform(image=image, bboxes=coco_bbox, category_ids=category_ids)
    transformed_img = transformed['image']
    transformed_bbox = transformed['bboxes']
    transformed_category = transformed['category_ids']


    img = transformed_img.copy()
    yolo_bbox = coco_to_yolo(transformed_bbox[0], img.shape[1], img.shape[0])
    yolo_bbox = [yolo_bbox]

    txt_filename = os.path.join(output_labels_dir, filename + '_' + str(i) + ".txt")
    img_filename = os.path.join(output_imgs_dir, filename + '_' + str(i) + ".jpg")

    with open(txt_filename, "w") as txt_file:
        txt_file.write('0 ')
        for bbox in yolo_bbox:
            txt_file.write(" ".join(str(coord) for coord in bbox) + "\n")

    cv2.imwrite(img_filename, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


