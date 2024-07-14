import json
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch

class CustomDataset(Dataset):
    def __init__(self, dataset_root, annotation_file, transform=None):
        self.dataset_root = dataset_root
        self.transform = transform
        with open(annotation_file) as f:
            self.annotations = json.load(f)["images"]

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        img_name = annotation["image"]
        
        # Attempt to find the correct subfolder for the image
        subfolders = ["cracks", "LE Erosion", "Lightening", "VG Panel"]
        img_path = None
        for subfolder in subfolders:
            potential_path = os.path.join(self.dataset_root, "Data images", subfolder, img_name)
            if os.path.exists(potential_path):
                img_path = potential_path
                break
        
        if img_path is None:
            raise FileNotFoundError(f"Image {img_name} not found in any subfolder.")
        
        image = Image.open(img_path).convert("RGB")

        # Apply transformations if specified
        if self.transform:
            image = self.transform(image)

        boxes = []
        labels = []
        for anno in annotation["annotations"]:
            label = anno["label"]
            if label == "crack":
                label_id = 0
            elif label == "erosion":
                label_id = 1
            elif label == "lightening":
                label_id = 2
            elif label == "vg panel":
                label_id = 3
            else:
                continue  # Skip this annotation if label is not recognized

            labels.append(label_id)
            x = anno["coordinates"]["x"]
            y = anno["coordinates"]["y"]
            width = anno["coordinates"]["width"]
            height = anno["coordinates"]["height"]
            boxes.append([x, y, x + width, y + height])

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64)
        }

        return image, target

def collate_fn(batch):
    images = []
    targets = []
    
    for item in batch:
        image, target = item
        images.append(image)
        targets.append(target)
    
    images = torch.stack(images, 0)
    
    return images, targets

if __name__ == "__main__":
    dataset_root = "/Users/amandanassar/Desktop/WindTurbineProject/"
    annotation_file = "/Users/amandanassar/Desktop/WindTurbineProject/annotations.json"
    
    # Define your transformation pipeline, including resizing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to a fixed size
        transforms.ToTensor(),
    ])
    
    train_dataset = CustomDataset(dataset_root, annotation_file, transform=transform)
    
    # Define DataLoader with custom collate_fn
    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    
    # Print the length of the dataset
    print(f"Dataset length: {len(train_dataset)}")
    
    # Iterate through the dataset
    for i in range(len(train_dataset)):
        img, target = train_dataset[i]
        print(f"Index: {i}")
        print(f"Image size: {img.size()}")
        print(f"Target annotations: {target}")
        print("------------------------------------")
