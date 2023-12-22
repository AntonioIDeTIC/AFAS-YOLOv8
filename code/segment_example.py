import os
import cv2
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from ultralytics import YOLO
import utils

def load_global_variables():
    """
    Load global variables for the image processing application.

    Returns:
    - path (str): Current working directory.
    - segment_model: YOLO model loaded from the specified path.
    - image_dir (str): Directory containing input images.
    - label_dir (str): Directory containing label files.
    - image_files (list): List of image filenames.
    - current_index (int): Current index for image navigation.
    """
    segment_model = YOLO(os.path.join('..', "model/model_186.pt"))
    image_dir = os.path.join('..', "original_data/valid/images/")
    label_dir = os.path.join('..', "original_data/valid/labels/")
    image_files = utils.get_list_images(image_dir, extension='.tiff')
    current_index = 0

    return segment_model, image_dir, label_dir, image_files, current_index

def process_key_event(event, image_files, current_index):
    """
    Process a key event to update the image index.

    Parameters:
    - event: Key event object.
    - image_files (list): List of image filenames.
    - current_index (int): Current index for image navigation.

    Returns:
    - new_index (int): Updated image index.
    """
    
    if event.key == 'right':
        new_index = (current_index + 1) % len(image_files) if image_files else 0
    elif event.key == 'left':
        new_index = (current_index - 1) % len(image_files) if image_files else 0
    elif event.key == 'q':
        plt.close()
        new_index = current_index if image_files else 0
    else:
        new_index = current_index

    # Handle the case where the new index is None
    if new_index is None:
        return current_index

    return new_index


def process_ground_truth_labels(labels, image):
    """
    Process ground truth labels to generate masks, class indices, and polygon coordinates.

    Parameters:
    - labels (list): List of ground truth labels, each containing class index and YOLO coordinates.
    - image: Input image.

    Returns:
    - ground_truth_masks (list): List of binary masks corresponding to ground truth labels.
    - ground_truth_class_indices (list): List of class indices corresponding to ground truth labels.
    - ground_truth_points (list): List of polygon coordinates corresponding to ground truth labels.
    """
    
    ground_truth_masks, ground_truth_points = [], []

    for label in labels:
        _, yolo_coordinates = label
        polygon_coordinates = np.asarray(utils.yolo_to_polygon_coordinates(yolo_coordinates))

        mask = utils.create_mask(image, polygon_coordinates)
        binary_mask = utils.convert_mask_to_binary(mask)
        
        ground_truth_masks.append(binary_mask)
        ground_truth_points.append(polygon_coordinates)

    return ground_truth_masks, ground_truth_points


def highlight_false_negatives(image_processed, overlay, gt_points, alpha):
    """
    Highlight false negatives on the processed image.

    Parameters:
    - image_processed: Processed image.
    - overlay: Overlay image.
    - gt_points (list): List of ground truth polygon coordinates.
    - alpha: Alpha parameter for image blending.

    Returns:
    - img_adaptive_new: Image with false negatives highlighted.
    """
    img_adaptive_new = image_processed.copy()  # Initialize img_adaptive_new before the loop
    for gt_point in gt_points:
        cv2.polylines(image_processed, np.int32([gt_point]), isClosed=True, color=(0, 255, 0), thickness=1)
        cv2.fillPoly(overlay, np.int32([gt_point]), (194, 247, 191))
        img_adaptive_new = cv2.addWeighted(overlay, alpha, image_processed, 1 - alpha, 0)
    return img_adaptive_new

def draw_and_fill(image_processed, overlay, points, r, color, filling, text, alpha):
    """
    Draw bounding box, label, and filled polygon on the processed image.

    Parameters:
    - image_processed: Processed image.
    - overlay: Overlay image.
    - points: Polygon coordinates.
    - r: Bounding box coordinates.
    - color: Color for drawing.
    - text: Text label.
    - alpha: Alpha parameter for image blending.
    - clss: Class index.

    Returns:
    - img_adaptive_new: Image with added annotations.
    """
    cv2.rectangle(image_processed, r[:2], r[2:], color=color, thickness=1)
    cv2.putText(
        image_processed,
        text,
        (r[0], (r[1] - 10)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.6,
        color=color,
        thickness=1)
    cv2.polylines(image_processed, np.int32([points]), isClosed=True, color=color, thickness=1)
    
   
    cv2.fillPoly(overlay, np.int32([points]), filling)

    img_adaptive_new = cv2.addWeighted(overlay, alpha, image_processed, 1 - alpha, 0)
    return img_adaptive_new

def plot_images(image, img_adaptive_new, title1, title2):
    """
    Plot original and processed images side by side.

    Parameters:
    - image: Original image.
    - img_adaptive_new: Processed image.
    - title1: Title for the original image.
    - title2: Title for the processed image.
    """
    plt.clf()
    plt.subplot(121)
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.title(title1, fontsize = 18)
    
    plt.subplot(122)
    plt.imshow(img_adaptive_new, cmap='gray')
    plt.axis('off')
    plt.title(title2, fontsize = 18)
    
    # Manually add legend
    legend_elements = [
        Line2D([0], [0], color=(0, 0, 1), lw=2, label='Powerline'),
        Line2D([0], [0], color=(1, 0, 0), lw=2, label='Tower'),
        Line2D([0], [0], color=(0, 1, 0), lw=2, label='False Negative'),
    ]

    legend = plt.legend(handles=legend_elements, loc = 'upper left', bbox_to_anchor=(1, 1))

    for line in legend.get_lines():
        line.set_linewidth(1.0)

def display_current_image(image_files, current_index, image_dir, label_dir, segment_model):
    """
    Display the current image with segmentation and ground truth annotations.

    Parameters:
    - image_files: List of image filenames.
    - current_index: Current index for image navigation.
    - image_dir: Directory containing input images.
    - label_dir: Directory containing label files.
    - segment_model: YOLO model for segmentation.
    """
    current_file = os.path.join(image_dir, image_files[current_index])
    
    image_name = utils.extract_image_name(image_files[current_index])
    alpha = 0.25
    
    print(f"Processing image: {image_name}")

    label_path = os.path.join(label_dir, image_files[current_index].replace(image_files[current_index].split('.')[-1], "txt"))

    image = cv2.resize(plt.imread(current_file), (640, 640))
    image_processed = utils.preprocess(image)
    overlay = image_processed.copy()

    boxes, points, masks, clss, probs = utils.segment_image(segment_model, image_processed, conf=0.5)
    labels = utils.load_label(label_path, image.shape[0], image.shape[1])

    ground_truth_masks, ground_truth_points = process_ground_truth_labels(labels, image)
    gt_points = utils.find_false_negatives(ground_truth_masks, ground_truth_points, masks, 0.5)
    
    img_adaptive_new = highlight_false_negatives(image_processed, overlay, gt_points, alpha)

    if boxes is not None:
        for i, box in enumerate(boxes):
            r = box.astype(int)

            if clss[i] == 0 and len(points[i]) > 0:
                img_adaptive_new = draw_and_fill(image_processed, overlay, points[i], r, (0, 0, 255), (135, 206, 250), f"powerline: {probs[i]:.2f}", alpha)

            elif clss[i] == 1 and len(points[i]) > 0:
                img_adaptive_new = draw_and_fill(image_processed, overlay, points[i], r, (255, 0, 0), (240, 128, 128), f"tower: {probs[i]:.2f}", alpha)

        plot_images(image, img_adaptive_new, f"Original Image: {image_name}", f"Segmented Image: {image_name}")

    else:
        plot_images(image, img_adaptive_new, f"Original Image: {image_name}", f"Not segmented image: {image_name}")

def main():
    segment_model, image_dir, label_dir, image_files, current_index = load_global_variables()
    fig, _ = plt.subplots(figsize=(18, 8))

    def on_key(event):
        nonlocal current_index
        
        # Set an initial value if current_index is None
        current_index = current_index if current_index is not None else 0
        current_index = process_key_event(event, image_files, current_index)

        # Check if current_index is None after processing the key event
        if current_index is None:
            return

        display_current_image(image_files, current_index, image_dir, label_dir, segment_model)
        plt.draw()

    fig.canvas.mpl_connect('key_press_event', on_key)
    display_current_image(image_files, current_index, image_dir, label_dir, segment_model)
    plt.show()

if __name__ == "__main__":
    main()
