import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader
from dataset import CustomDataset, collate_fn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
in_channel = 3
learning_rate = 1e-3
batch_size = 10
num_epochs = 10

# Load Data
dataset_root = "/Users/amandanassar/Desktop/WindTurbineProject/"
annotation_file = "/Users/amandanassar/Desktop/WindTurbineProject/annotations.json"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = CustomDataset(dataset_root, annotation_file, transform=transform)

# Split dataset into training and testing sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# Model
model = torchvision.models.googlenet(pretrained=True)
num_classes = 4  # Update this to match your dataset (crack, erosion, lightening, vg panel)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train Network
for epoch in range(num_epochs):
    losses = []
    model.train()
    for batch_idx, (data, targets) in enumerate(train_loader):
        try:
            if data is None or targets is None:
                raise ValueError("Received None data or targets")
                
            print(f"Batch {batch_idx}: data shape: {data.shape}, targets: {targets}")

            data = data.to(device=device)
            batch_loss = 0
            for i in range(data.size(0)):
                img = data[i].unsqueeze(0)  # Add batch dimension
                target = targets[i]
                target_labels = target['labels'].to(device=device)
                
                scores = model(img)
                loss = criterion(scores, target_labels.float())  # Convert labels to float for BCEWithLogitsLoss
                batch_loss += loss
            
            batch_loss /= data.size(0)  # Average loss for the batch
            losses.append(batch_loss.item())
            
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            
            if batch_idx % 10 == 0:  # Print every 10 batches
                print(f'Epoch [{epoch}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {batch_loss.item():.4f}')
            
        except Exception as e:
            print(f"Error in batch {batch_idx}: {e}")
            print(f"Data shape: {data.shape if data is not None else 'None'}")
            print(f"Targets: {targets}")
            continue
    
    if losses:
        print(f'Cost at epoch {epoch} is {sum(losses) / len(losses)}')
    else:
        print(f'No valid batches in epoch {epoch}')

# Update the check_accuracy function
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()
    
    with torch.no_grad():
        for x, y in loader:
            try:
                if x is None or y is None:
                    raise ValueError("Received None x or y")
                
                x = x.to(device=device)
                for i in range(x.size(0)):
                    img = x[i].unsqueeze(0)  # Add batch dimension
                    target = y[i]
                    target_labels = target['labels'].to(device=device)
                    
                    scores = model(img)
                    _, predictions = scores.max(1)
                    num_correct += (predictions == target_labels).sum().item()
                    num_samples += len(target_labels)
            except Exception as e:
                print(f"Error in accuracy check: {e}")
                continue
    
    if num_samples > 0:
        accuracy = float(num_correct) / float(num_samples) * 100
        print(f'Got {num_correct} / {num_samples} with accuracy {accuracy:.2f}%')
    else:
        print("No valid samples for accuracy check")
    model.train()

# After training, check accuracy
print("Checking accuracy on Training Set")
check_accuracy(train_loader, model)

print("Checking accuracy on Test Set")
check_accuracy(test_loader, model)
