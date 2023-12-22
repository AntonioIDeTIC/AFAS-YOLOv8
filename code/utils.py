#!/usr/bin/python3
import os
import cv2
import numpy as np


def extract_image_name(image_file):
    """
    Extract the image name from the given image file.

    Parameters:
    - image_file (str): Image file name.

    Returns:
    - image_name (str): Extracted image name.
    """
    
    image_name = image_file.split('.')[0]
    first_underscore_index = image_name.find("_")
    second_underscore_index = image_name.find("_", first_underscore_index + 1)
    return image_name[:second_underscore_index]

def create_mask(image, polygon_coordinates):
    """
    Create a binary mask based on polygon coordinates.

    Parameters:
    - image: Input image.
    - polygon_coordinates (array): Array of polygon coordinates.

    Returns:
    - mask: Binary mask.
    """
    
    mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    pts = np.array([polygon_coordinates], dtype=np.int32)
    cv2.fillPoly(mask, pts, 255)
    return mask


def normalize(image, from_min, from_max, to_min, to_max):
    """
    Normalize the pixel values of an image.

    Parameters:
    - image: Input image (numpy array).
    - min_value: Minimum value of the original image.
    - max_value: Maximum value of the original image.
    - new_min: Minimum value of the normalized image.
    - new_max: Maximum value of the normalized image.

    Returns:
    - normalized_image: Image with pixel values normalized between new_min and new_max.
    """
    from_range = from_max - from_min
    to_range = to_max - to_min
    scaled = np.array((image - from_min) / float(from_range), dtype=float)
    
    normalized_image = to_min + (scaled * to_range)
    
    return normalized_image

def preprocess(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Preprocess the input image by applying a linear normalization and CLAHE (Contrast Limited Adaptive Histogram Equalization).

    Parameters:
    - image: Input image (numpy array).
    - clip_limit: CLAHE clip limit (default is 2.0).
    - tile_grid_size: CLAHE tile grid size (default is (8, 8)).

    Returns:
    - img_processed: Image after preprocessing with CLAHE.
    """
    # Create CLAHE object with specified parameters
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    # Normalize uint16 image if needed
    if image.dtype == 'uint16':
        image = np.asarray(normalize(image, np.min(image), np.max(image), 0, 255), dtype=np.uint8)

    # Apply CLAHE based on image shape
    if image.shape == (640, 640, 3):
        # Process each channel separately
        result_channels = [clahe.apply(channel) for channel in cv2.split(image)]
        img_processed = np.dstack(result_channels)
    else:
        # Apply CLAHE to the entire image
        img_processed = clahe.apply(image)
        img_processed = np.dstack([img_processed] * 3)  # Convert single channel to 3 channels

    return img_processed


def get_list_images(dir, extension):
    """
    Get a list of all files in the directory with the desired extension, sorted by numbers.

    Parameters:
    - dir: Directory path.
    - extension: File extension (e.g., '.jpg').

    Returns:
    - files: List of file names with the specified extension, sorted by numbers.
    """
    # Get a list of all files in the directory with the desired extension
    files = [f for f in os.listdir(dir) if f.endswith(extension)]

    # Custom sorting function to sort files by numbers after the first underscore
    def sort_by_numbers(file_name):
        # Extract numbers from the part of the file name after the first underscore
        numbers = [int(s) for s in file_name.split('_')[1].split() if s.isdigit()]
        return numbers

    files = sorted(files, key=sort_by_numbers)

    return files

def load_label(label_path, image_width, image_height):
    """
    Load labels from a file containing class indices and normalized coordinates.

    Parameters:
    - label_path: Path to the label file.
    - image_width: Width of the corresponding image.
    - image_height: Height of the corresponding image.

    Returns:
    - labels: List of tuples, each containing class index and pixel coordinates.
    """
    with open(label_path, 'r') as file:
        lines = file.readlines()
    
    labels = []
    for line in lines:
        parts = line.strip().split(' ')
        class_index = int(parts[0])
        coordinates = [float(x) for x in parts[1:]]
        
        # Convert normalized coordinates to pixel values
        coordinates = [int(image_width * x) if i % 2 == 0 else int(image_height * x) for i, x in enumerate(coordinates)]
        
        labels.append((class_index, coordinates))
    
    return labels

def yolo_to_polygon_coordinates(yolo_coordinates):
    """
    Convert YOLO coordinates to a list of (x, y) points.

    Parameters:
    - yolo_coordinates: List of YOLO coordinates.

    Returns:
    - polygon_coordinates: List of (x, y) points.
    """
    # Convert YOLO coordinates to a list of (x, y) points
    polygon_coordinates = [[yolo_coordinates[i], yolo_coordinates[i + 1]] for i in range(0, len(yolo_coordinates), 2)]
    return polygon_coordinates



# Function to detect classes
def segment_image(model, image, conf): 
    """
    Segment an image using a given model and confidence threshold.

    Parameters:
    - model: Object detection model.
    - image: Input image.
    - conf: Confidence threshold for detection.

    Returns:
    - boxes: Bounding boxes in xyxy format.
    - points: Segmentation points.
    - masks: Binary masks.
    - clss: Class indices.
    - probs: Confidence scores.

    Note:
    - The function checks if there are detections before processing.
    """
    result = model.predict(image, conf=conf, verbose=False)[0]

    if result:
        # Extract detection information
        clss = result.boxes.cls.cpu().numpy()
        probs = result.boxes.conf.cpu().numpy()
        boxes = result.boxes.xyxy.cpu().numpy()
        masks = result.masks.cpu().data.numpy()

        # Convert segmentation points to a NumPy array
        points = np.array(result.masks.xy, dtype=object)

        return boxes, points, masks, clss, probs
    else:
        print("No detection")
        return None, None, None, None, None

def calculate_iou(mask1, mask2):
    """
    Calculate the Intersection over Union (IoU) between two binary masks.

    Parameters:
    - mask1: Binary mask 1.
    - mask2: Binary mask 2.

    Returns:
    - iou: Intersection over Union value.
    """
    intersection = np.logical_and(mask1, mask2)
    intersection_area = np.count_nonzero(intersection)
    mask1_area = np.count_nonzero(mask1)
    mask2_area = np.count_nonzero(mask2)
    max_area = max(mask1_area, mask2_area)
    iou = intersection_area / max_area

    return iou

def convert_mask_to_binary(mask):
    """
    Convert a mask to a binary representation.

    Parameters:
    - mask: Input mask.

    Returns:
    - binary_mask: Binary representation of the mask.
    """
    return np.where(mask == 255, 1, 0)
    
def find_false_negatives(ground_truth_masks, polygon_coordinates, predicted_masks, iou_threshold):
    """
    Find false negatives (FN) by comparing ground truth masks with predicted masks.

    Parameters:
    - ground_truth_masks: List of ground truth binary masks.
    - polygon_coordinates: List of ground truth polygon coordinates.
    - predicted_masks: List of predicted binary masks.
    - iou_threshold: Intersection over Union (IoU) threshold for considering a match.

    Returns:
    - polygon_arr: List of ground truth polygon coordinates corresponding to false negatives.
    """
    polygon_arr = []  # False negatives polygon

    for gt_mask, gt_pol in zip(ground_truth_masks, polygon_coordinates):
        found_match = False

        # Check each predicted mask for a match
        for pr_mask in predicted_masks:
            iou = calculate_iou(gt_mask, pr_mask)  # You can choose which IoU function to use

            if iou >= iou_threshold:
                found_match = True
                break

        if not found_match:
            polygon_arr.append(gt_pol)  # False negatives polygon

    return polygon_arr

