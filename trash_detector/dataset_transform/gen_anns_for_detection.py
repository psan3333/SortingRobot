import json

anns_url = "./data/annotations.json"
bbox_anns_url = "./data/bbox_anns.json"

"""
info
images
annotations
scene_annotations
categories
scene_categories
"""

"""
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

with open(anns_url, "r") as file:
    data = json.load(file)
    bbox_anns = []

    for ann in data["annotations"]:
        bbox_anns.append(
            {
                "id": ann["id"],
                "image_id": ann["image_id"],
                "category_id": ann["category_id"],
                "area": ann["area"],
                "bbox": ann["bbox"],
                "iscrowd": ann["iscrowd"],
            }
        )

    overall_annotations = {
        "info": data["info"],
        "images": data["images"],
        "annotations": bbox_anns,
        "scene_annotations": data["scene_annotations"],
        "categories": data["categories"],
        "scene_categories": data["scene_categories"],
    }

with open(bbox_anns_url, "w") as bbox_anns_file:
    bbox_anns = json.dumps(overall_annotations, indent=4)
    bbox_anns_file.write(bbox_anns)
