import torch
from torch.nn.functional import cross_entropy
from data.dataloader import get_dataloaders
from src.model import SpatialVisionFusion
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
import os

def train(model, dataloader, optimizer, scheduler, device, epoch, writer):
    model.train()
    total_loss = 0.0
    for batch_idx, data_dict in enumerate(dataloader):
        # later on change images to also include metadata features and update model forward function to take in both image and metadata features
        images, labels = data_dict['image'].to(device), data_dict['label'].to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 10 == 0:
            print(f'Epoch [{epoch}], Batch [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}')
    
    avg_loss = total_loss / len(dataloader)
    writer.add_scalar('Loss/train', avg_loss, epoch)
    scheduler.step()
    
def validate(model, dataloader, device, epoch, writer):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data_dict in dataloader:
            images, labels = data_dict['image'].to(device), data_dict['label'].to(device)
            outputs = model(images)
            loss = cross_entropy(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100. * correct / total
    avg_loss = total_loss / len(dataloader)
    writer.add_scalar('Loss/val', avg_loss, epoch)
    writer.add_scalar('Accuracy/val', accuracy, epoch)
    print(f'Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
    return avg_loss, accuracy

def test(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data_dict in dataloader:
            # Same with validation and test, update to include metadata features later on
            images, labels = data_dict['image'].to(device), data_dict['label'].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100. * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # To train different models change the model here 
    model = SpatialVisionFusion().to(device)
    
    
    # Uncomment below to train ResNet-50 baseline instead of SpatialVisionFusion
    # Comment out the SpatialVisionFusion import and model initialization at the top of this file if training ResNet-50 baseline instead
    # model = models.resnet50(pretrained=True)
    # model.fc = torch.nn.Linear(model.fc.in_features, 6)  # Change final layer for 6 classes
    # model = model.to(device)
    
    model_type = "SpatialVisionFusion"  # Change to "ResNet50" if training ResNet-50 baseline
    # model_type = "ResNet50"
    
    #Replace with path to data on your machine after running data_lod.py
    path_to_data = '/Users/noahtakashima/.cache/kagglehub/datasets/mahdavi1202/skin-cancer/versions/1'
    train_loader, val_loader, test_loader = get_dataloaders(path_to_data, batch_size=32)
    
    optimizer = Adam(model.parameters(), lr=1e-4)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    
    def get_next_experiment_id(base_dir: str) -> int:
        if not os.path.isdir(base_dir):
            return 1
        existing = []
        for name in os.listdir(base_dir):
            if name.startswith(f'{model_type}_experiment_'):
                suffix = name.replace(f'{model_type}_experiment_', '')
                if suffix.isdigit():
                    existing.append(int(suffix))
        return (max(existing) + 1) if existing else 1

    experiment_id = get_next_experiment_id('tensor_board')
    log_dir = os.path.join('tensor_board', f'{model_type}_experiment_{experiment_id}')
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    models_dir = os.path.join('models', f'{model_type}_experiment_{experiment_id}')
    os.makedirs(models_dir, exist_ok=True)
    best_val_loss = float('inf')
    
    num_epochs = 20
    for epoch in range(1, num_epochs + 1):
        train(model, train_loader, optimizer, scheduler, device, epoch, writer)
        val_loss, val_acc = validate(model, val_loader, device, epoch, writer)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(models_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
            }, best_path)
    test(model, test_loader, device)
    writer.close()
    
    
if __name__ == "__main__":
    main()