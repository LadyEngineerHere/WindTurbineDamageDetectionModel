import os
import json

# Define the path to the main project folder and annotations folder
project_folder = '/Users/amandanassar/Desktop/WindTurbineProject'
annotations_folder = os.path.join(project_folder, 'annotations')

# Subfolders inside the annotations folder
subfolders = ['crack', 'erosion', 'lightening', 'vg panel']

# Initialize a dictionary to store the combined annotations
combined_annotations = {
    "images": [],
    "annotations": [],
    "categories": [
        {"id": 0, "name": "crack", "supercategory": "none"},
        {"id": 1, "name": "erosion", "supercategory": "none"},
        {"id": 2, "name": "lightening", "supercategory": "none"},
        {"id": 3, "name": "vg panel", "supercategory": "none"}
    ]
}

# Function to load and combine JSON files
def load_and_combine_json(file_path, label, image_id_start):
    with open(file_path, 'r') as file:
        data = json.load(file)
        for idx, image_annotation in enumerate(data):
            image_annotation["annotations"] = [
                {
                    "label": label,
                    "coordinates": ann["coordinates"]
                } for ann in image_annotation.get("annotations", [])
            ]
            image_annotation["id"] = image_id_start + idx + 1
            combined_annotations["images"].append(image_annotation)

# Iterate over each subfolder
image_id = 0
for subfolder in subfolders:
    subfolder_path = os.path.join(annotations_folder, subfolder)
    
    # Iterate over each JSON file in the subfolder
    for filename in os.listdir(subfolder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(subfolder_path, filename)
            label = subfolder
            load_and_combine_json(file_path, label, image_id)
            image_id += len(os.listdir(subfolder_path))

# Save the combined annotations to a new JSON file in the main project folder
output_file = os.path.join(project_folder, 'annotations.json')
with open(output_file, 'w') as outfile:
    json.dump(combined_annotations, outfile, indent=4)

print(f'Combined annotations saved to {output_file}')
