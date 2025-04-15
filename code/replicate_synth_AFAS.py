#!/usr/bin/python3
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import shutil
import shapely
from shapely.geometry import Polygon, LineString
from shapely import intersection
import random
import time
import utils
from SubpixelFireGeneration import FireEffectGenerator as FEG

last_iter = [np.array([1, 1])]

def polygon_to_bbox(polygon_vertices):
    """
    Convert a polygon represented by x and y coordinate lists to a bounding box.

    Parameters:
        polygon_vertices (tuple): A tuple of two lists (x_list, y_list) representing the polygon's vertices.

    Returns:
        tuple: A bounding box in the format (x_min, y_min, x_max, y_max).
    """
    xlist, ylist = polygon_vertices

    x_min = min(xlist)
    y_min = min(ylist)
    x_max = max(xlist)
    y_max = max(ylist)

    return x_min, y_min, x_max, y_max 


def is_overlapped(target, polygons):
    """
    Check if a target polygon overlaps with any polygon in a list.

    Parameters:
        target (shapely.geometry.Polygon): The target polygon to test.
        polygons (list): A list of shapely Polygon objects.

    Returns:
        bool: True if any intersection is found; False otherwise.
    """
    result = map(intersection, [target for _ in polygons], polygons)
    return len(list(filter(lambda x: not x.is_empty, result))) > 0


def find_crossing_points(bounding_box, polygon, x_coordinate, y_coordinate, cls):
    """
    Find a crossing point between a vertical or horizontal line and a polygon within a bounding box.

    Parameters:
        bounding_box (tuple): The bounding box (x_min, y_min, x_max, y_max).
        polygon (shapely.geometry.Polygon): The polygon to check for intersections.
        x_coordinate (float): X coordinate for vertical line.
        y_coordinate (float): Y coordinate for horizontal line.
        cls (int): Direction flag (0 = vertical, 1 = horizontal).

    Returns:
        tuple: A crossing point's coordinates and direction (1 or -1) if found, else (None, None).
    """
    if cls == 0:
        line = LineString([(x_coordinate, bounding_box[1]), (x_coordinate, bounding_box[3])])
    else:
        line = LineString([(bounding_box[1], y_coordinate), (bounding_box[3], y_coordinate)])

    if np.asarray(line.xy[0][0]) == last_iter[0][0] and np.asarray(line.xy[0][1]) == last_iter[0][1]:
        line = LineString([(x_coordinate, bounding_box[1]), (x_coordinate, bounding_box[3])])

    intersections = line.intersection(polygon)

    if intersections.geom_type == 'MultiLineString' or intersections.geom_type == 'GeometryCollection':
        intersections = [x for x in list(intersections.geoms) if x.geom_type == 'LineString']   
        intersections = random.choice(intersections)

    elif intersections.geom_type == 'MultiPoint' or intersections.geom_type == 'Point':
        return None, None

    element = random.choice([0, 1])
    del last_iter[0]
    last_iter.append(np.asarray(line.xy[0]))

    return intersections.coords[element], 1 if intersections.coords[element][1] > intersections.coords[1 - element][1] else -1


def square(point, distance, direction, fire_source_size):
    """
    Generate a square-shaped polygon centered at a point offset by distance in a given direction.

    Parameters:
        point (tuple): The (x, y) center point.
        distance (float): The distance to shift in y-direction.
        direction (int): Direction modifier (1 or -1).
        fire_source_size (int): Half the side length of the square (actual side will be fire_source_size * 2).

    Returns:
        tuple: A tuple containing the square polygon, and its center x, y coordinates.
    """
    x, y = point
    distance *= direction
    y += distance
    half_side_length = fire_source_size * 2 

    return Polygon(((x - half_side_length, y - half_side_length),
                    (x + half_side_length, y - half_side_length),
                    (x + half_side_length, y + half_side_length),
                    (x - half_side_length, y + half_side_length))), x, y

    
path = os.getcwd()

# Define the directory where the  images are located
image_dir = os.path.join(path, "../AFAS/original_data/valid/images/")
label_dir = os.path.join(path, "../AFAS/original_data/valid/labels/")

tiff_files = utils.get_list_images(image_dir, extension = '.tiff')
jpeg_files = utils.get_list_images(image_dir, extension = '.jpeg')


image_files = tiff_files + jpeg_files
image_files.sort()

txt_files = utils.get_list_images(label_dir, extension = '.txt')

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

iterations = np.arange(0, 2, 1)
synth_AFAS = ['AFAS-SubtleFire', 'AFAS-IntenseFire']


#################### GENERATE AFAS-SubtleFire and AFAS-IntenseFire for the energy retention metric evaluation ####################

for iter in iterations:

    for s in synth_AFAS:
        if s == 'AFAS-SubtleFire':
            factor = 0.2
        else:
            factor = 0.5
            
        for d in range(2, 3):
            
            imgdir = os.path.join(path, f"../synth_AFAS/energy_retention/{s}/iteration_{iter}/bbox_dataset_d{d}/valid/images/")
            labeldir = os.path.join(path, f"../synth_AFAS/energy_retention/{s}/iteration_{iter}/bbox_dataset_d{d}/valid/labels/")

            if not os.path.exists(imgdir):
                os.makedirs(imgdir)
            if not os.path.exists(labeldir):
                os.makedirs(labeldir)

            for k in range(0, 20):
                for idx, img_name in enumerate(image_files):
                    polygons = []
                    classes = []

                    current_file = os.path.join(image_dir, image_files[idx])
                    image_ = plt.imread(current_file)
                    image = image_.copy()
                    
                    if image.shape == (640, 640, 3):
                        image = image[:,:,0]
            
                    # Randomly apply fire to the image
                    image_width, image_height = image.shape[1], image.shape[0]
                    
                    tresh = np.max(image)

                    if tresh > 9000:
                        tresh == np.mean(image) + 1000
                        
                    fire_source_intensity = int(tresh * (0.3 * np.random.rand() + factor))  # Adjust the intensity as needed
                    fire_source_size = 0.55  # Adjust the size as needed
                    
                    # predict by YOLOv8
                    label_name = label_dir +  os.path.splitext(img_name)[0] + '.txt'
                    labels = utils.load_label(label_name, image.shape[0], image.shape[1]) # txt_files
                    for label in labels:
                        class_index, yolo_coordinates = label
                        
                        # if class_index == 0:
                        polygon_coordinates = np.asarray(utils.yolo_to_polygon_coordinates(yolo_coordinates), dtype=np.int32)
                        polygons.append(Polygon(polygon_coordinates))
                        classes.append(class_index)
                    
                    count = 0
                    while True:
                        try:
                            random_index = random.choice(range(len(polygons)))
                            random_polygon = polygons[random_index]
                            random_class = classes[random_index]
                            bbox = polygon_to_bbox(random_polygon.exterior.coords.xy)
                            xmin, ymin, xmax, ymax = bbox
                            x_coordinate = random.randint(xmin, xmax)
                            y_coordinate = random.randint(ymin, ymax)
                            crossing_points, direction = find_crossing_points(bbox, random_polygon, x_coordinate, y_coordinate, random_class)
                            if crossing_points:
                                crossing_points = np.array(crossing_points, dtype=int)
                                gaussian_box, x, y = square(crossing_points, d, direction, fire_source_size)
                                if not is_overlapped(gaussian_box, polygons):
                                    break
                            if count == 20:
                                break
                            count += 1

                        except shapely.errors.GEOSException:
                            pass
                        except IndexError:
                            pass
                        
                    firegen = FEG()
                    
                    if image.dtype == 'uint16':
                        # Initialize the fire effect generator
                        synth_image = firegen.apply_fire(image, x, y, image.shape[0], fire_source_intensity, fire_source_size)
                        synth_image = np.asarray(utils.normalize(synth_image, np.min(image), np.max(image), 0, 255), dtype=np.uint8)
                        img_adaptive_eq = clahe.apply(synth_image)
                        img_adaptive_eq = np.dstack([img_adaptive_eq, img_adaptive_eq, img_adaptive_eq])
                    else:
                        synth_image = firegen.apply_fire(image, x, y, image.shape[0], fire_source_intensity, fire_source_size)
                        img_adaptive_eq = clahe.apply(synth_image)
                        img_adaptive_eq = np.dstack([img_adaptive_eq, img_adaptive_eq, img_adaptive_eq])

                    # plt.imshow(synth_image)
                    # plt.show()
                    identifier = f"_x{x}_y{y}_intensity_{fire_source_intensity}"
                    # # Get the file extension from the original file
                    # # Construct the new filename with the custom identifier
                    new_img_filename = os.path.splitext(current_file)[0].split('/')[-1] + identifier + '.jpg'
                    new_label_filename = os.path.splitext(current_file)[0].split('/')[-1] + identifier + '.txt'

                    # # Save the image with the custom identifier in the 'test' directory
                    output_img_path = os.path.join(imgdir, new_img_filename)
                    output_label_path = os.path.join(labeldir, new_label_filename)

                    if count != 20:
                        cv2.imwrite(output_img_path, img_adaptive_eq)
                        shutil.copy(label_dir + os.path.splitext(current_file)[0].split('/')[-1] + '.txt', output_label_path)


#################### GENERATE AFAS-SubtleFire and AFAS-IntenseFire for the two-sample KS test evaluation ####################
for iter in iterations:
    for s in synth_AFAS:
        imgdir =  os.path.join(path, f"../synth_AFAS/ks2test/{s}/iteration_{iter}/valid/images/")
        labeldir = os.path.join(path, f"../synth_AFAS/ks2test/{s}/iteration_{iter}/valid/labels/")

        if not os.path.exists(imgdir):
            os.makedirs(imgdir)
        if not os.path.exists(labeldir):
            os.makedirs(labeldir)

        if s == 'AFAS-SubtleFire':
            factor = 0.2
        else:
            factor = 0.5
        for k in range(0, 20):
            for idx, img_name in enumerate(image_files):

                current_file = os.path.join(image_dir, image_files[idx])
                image_ = plt.imread(current_file)
                image = image_.copy()
                
                if image.shape == (640, 640, 3):
                    image = image[:,:,0]


                # Randomly apply fire to the image
                image_width, image_height = image.shape[1], image.shape[0]
                
                tresh = np.max(image)

                
                if tresh > 9000:
                    tresh == np.mean(image) + 1000
                    
                fire_source_intensity = int(tresh * (0.3 * np.random.rand() + factor))  # Adjust the intensity as needed
                fire_source_size = 0.55  # Adjust the size as needed

                x = random.randint(0, image_width)
                y = random.randint(0, image_height)
                
                firegen = FEG()
                
                if image.dtype == 'uint16':
                    synth_image = firegen.apply_fire(image, x, y, image.shape[0], fire_source_intensity, fire_source_size)
                    synth_image = np.asarray(utils.normalize(synth_image, np.min(image), np.max(image), 0, 255), dtype=np.uint8)
                    img_adaptive_eq = clahe.apply(synth_image)
                    img_adaptive_eq = np.dstack([img_adaptive_eq, img_adaptive_eq, img_adaptive_eq])
                else:
                    synth_image = firegen.apply_fire(image, x, y, image.shape[0], fire_source_intensity, fire_source_size)
                    img_adaptive_eq = clahe.apply(synth_image)
                    img_adaptive_eq = np.dstack([img_adaptive_eq, img_adaptive_eq, img_adaptive_eq])

                identifier = f"_x{x}_y{y}_intensity_{fire_source_intensity}"
                # # Get the file extension from the original file
                # # Construct the new filename with the custom identifier
                new_img_filename = os.path.splitext(current_file)[0].split('/')[-1] + identifier + '.jpg'
                new_label_filename = os.path.splitext(current_file)[0].split('/')[-1] + identifier + '.txt'
                
                output_img_path = os.path.join(imgdir, new_img_filename)
                output_label_path = os.path.join(labeldir, new_label_filename)
                

                cv2.imwrite(output_img_path, img_adaptive_eq)
                shutil.copy(label_dir + os.path.splitext(current_file)[0].split('/')[-1] + '.txt', output_label_path)