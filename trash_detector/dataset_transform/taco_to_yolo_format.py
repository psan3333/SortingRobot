import json
import numpy as np
import cv2

"""
info
images
annotations
scene_annotations
categories
scene_categories
"""

"""
image_data:
            "id": 1,
            "width": 1537,
            "height": 2049,
            "file_name": "batch_1/000008.jpg",
            "license": null,
            "flickr_url": "https://farm66.staticflickr.com/65535/47803331152_ee00755a2e_o.png",
            "coco_url": null,
            "date_captured": null,
            "flickr_640_url": "https://farm66.staticflickr.com/65535/47803331152_19beae025a_z.jpg"

annotation:
            "id": 54,
            "image_id": 19,
            "category_id": 12,  
            "area": 108233.5,
            "bbox": [
                503.0,
                808.0,
                445.0,
                357.0
            ],
            "iscrowd": 0
"""

# labels_mapping = {
#     0: "aluminium foil",
#     1: "battery",
#     2: "blister pack",
#     3: "bottle",
#     4: "bottle cap",
#     5: "broken glass",
#     6: "can",
#     7: "cartoon",
#     8: "cup",
#     9: "food waste",
#     10: "glass jar",
#     11: "lid",
#     12: "other plastic",
#     13: "paper",
#     14: "paper bag",
#     15: "plastic bag & wrapper",
#     16: "plastic container",
#     17: "plastic gloves",
#     18: "plastic utensils",
#     19: "pop tab",
#     20: "rope & strings",
#     21: "scrap metal",
#     22: "shoe",
#     23: "squeezable tube",
#     24: "straw",
#     25: "styrofoam piece",
#     26: "unlabeled litter",
#     27: "cigarette",
# }

labels_mapping = [
    "aluminium foil",
    "battery",
    "blister pack",
    "bottle",
    "bottle cap",
    "broken glass",
    "can",
    "carton",
    "cup",
    "food waste",
    "glass jar",
    "lid",
    "other plastic",
    "paper",
    "paper bag",
    "plastic bag & wrapper",
    "plastic container",
    "plastic glooves",
    "plastic utensils",
    "pop tab",
    "rope & strings",
    "scrap metal",
    "shoe",
    "squeezable tube",
    "straw",
    "styrofoam piece",
    "unlabeled litter",
    "cigarette",
]


def bbox_to_yolo(bbox_annotation, image_data, categories):
    global labels_mapping
    image_height = image_data["height"]
    image_width = image_data["width"]
    bbox_x, bbox_y, bbox_width, bbox_height = bbox_annotation["bbox"]
    yolo_form_width = float(bbox_width / image_width)
    yolo_form_height = float(bbox_height / image_height)

    yolo_form_x = float((bbox_x + bbox_width / 2) / image_width)
    yolo_form_y = float((bbox_y + bbox_height / 2) / image_height)

    # map label to config formats
    label_to_yolo = labels_mapping.index(
        categories[bbox_annotation["category_id"]]["supercategory"].lower()
    )
    return (
        label_to_yolo,
        round(yolo_form_x, 6),
        round(yolo_form_y, 6),
        round(yolo_form_width, 6),
        round(yolo_form_height, 6),
    )


yolo_images_dir = "./data/images/"
yolo_labels_dir = "./data/labels/"
taco_data_dir = "./taco_data/"
anns_file = "./taco_data/bbox_anns.json"
cnt = 0
with open(anns_file, "r") as anns_file:
    bbox_anns = json.load(anns_file)
    annotations = bbox_anns["annotations"]
    categories = bbox_anns["categories"]
    images_data = bbox_anns["images"]
    visited_ids = np.zeros(len(images_data))
    for annotation in annotations:
        cnt += 1
        print(cnt, end=" ")
        if cnt % 50 == 0:
            print()
        image_data = images_data[annotation["image_id"]]
        yolo_bbox = bbox_to_yolo(annotation, image_data, categories)
        image_filename = taco_data_dir + image_data["file_name"]
        image_id = image_data["id"]
        if visited_ids[image_id] == 0:
            new_filename = yolo_images_dir + f"{str(image_id).rjust(6, '0')}.jpg"
            image = cv2.imread(image_filename)
            cv2.imwrite(new_filename, image)
            visited_ids[image_id] = 1
        with open(
            f"{yolo_labels_dir}/{str(image_id).rjust(6, '0')}.txt", "a"
        ) as labels_file:
            labels_file.write(
                f"{yolo_bbox[0]} {yolo_bbox[1]} {yolo_bbox[2]} {yolo_bbox[3]} {yolo_bbox[4]}\n"
            )
