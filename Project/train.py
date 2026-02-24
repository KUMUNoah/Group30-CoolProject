import torch
import torch.nn as nn
import numpy as np
from data.dataloader import get_dataloaders
from src.model import MultiModalCNNFusion
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
import os

NUM_CLASSES = 6
CLASS_NAMES = ["BCC", "SCC", "ACK", "SEK", "MEL", "NEV"]

def train(model, dataloader, optimizer, scheduler, criterion, device, epoch, writer):
    model.train()
    total_loss = 0.0
    for batch_idx, data_dict in enumerate(dataloader):
        images, labels, metadata = data_dict['image'].to(device), data_dict['label'].to(device), data_dict['metadata'].to(device)

        optimizer.zero_grad()
        outputs = model(images, metadata)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 10 == 0:
            print(f'Epoch [{epoch}], Batch [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}')

    avg_loss = total_loss / len(dataloader)
    writer.add_scalar('Loss/train', avg_loss, epoch)
    # Log current learning rate
    writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
    scheduler.step()

def validate(model, dataloader, criterion, device, epoch, writer):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    class_correct = np.zeros(NUM_CLASSES)
    class_total = np.zeros(NUM_CLASSES)

    with torch.no_grad():
        for data_dict in dataloader:
            images, labels, metadata = data_dict['image'].to(device), data_dict['label'].to(device), data_dict['metadata'].to(device)
            outputs = model(images, metadata)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            for cls in range(NUM_CLASSES):
                mask = (labels == cls)
                class_correct[cls] += (predicted[mask] == labels[mask]).sum().item()
                class_total[cls] += mask.sum().item()

    accuracy = 100. * correct / total
    avg_loss = total_loss / len(dataloader)
    writer.add_scalar('Loss/val', avg_loss, epoch)
    writer.add_scalar('Accuracy/val', accuracy, epoch)
    print(f'Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
    print('  Per-class accuracy:')
    for i, name in enumerate(CLASS_NAMES):
        if class_total[i] > 0:
            cls_acc = 100. * class_correct[i] / class_total[i]
            writer.add_scalar(f'Accuracy/val_{name}', cls_acc, epoch)
            print(f'    {name}: {cls_acc:.1f}% ({int(class_correct[i])}/{int(class_total[i])})')
    return avg_loss, accuracy

def test(model, dataloader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    class_correct = np.zeros(NUM_CLASSES)
    class_total = np.zeros(NUM_CLASSES)

    with torch.no_grad():
        for data_dict in dataloader:
            images, labels, metadata = data_dict['image'].to(device), data_dict['label'].to(device), data_dict['metadata'].to(device)
            outputs = model(images, metadata)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            for cls in range(NUM_CLASSES):
                mask = (labels == cls)
                class_correct[cls] += (predicted[mask] == labels[mask]).sum().item()
                class_total[cls] += mask.sum().item()

    accuracy = 100. * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    print('  Per-class accuracy:')
    for i, name in enumerate(CLASS_NAMES):
        if class_total[i] > 0:
            cls_acc = 100. * class_correct[i] / class_total[i]
            print(f'    {name}: {cls_acc:.1f}% ({int(class_correct[i])}/{int(class_total[i])})')

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # To train different models change the model here
    model = MultiModalCNNFusion().to(device)

    model_type = "MultiModalCNNFusion"

    #Replace with path to data on your machine after running data_load.py
    path_to_data = '/Users/noahtakashima/.cache/kagglehub/datasets/mahdavi1202/skin-cancer/versions/1'
    train_loader, val_loader, test_loader = get_dataloaders(path_to_data, batch_size=32)

    # Class-weighted loss with label smoothing: handles class imbalance and reduces overconfidence
    class_weights = train_loader.dataset.class_weights().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    num_epochs = 20
    optimizer = Adam([
        {'params': model.cnn.parameters(), 'lr': 1e-5},        # backbone: slow
        {'params': model.metadata_mlp.parameters(), 'lr': 1e-3},
        {'params': model.classifier.parameters(), 'lr': 1e-3},
    ])
    # CosineAnnealingLR decays LR smoothly to eta_min instead of dropping sharply every 5 epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

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
    best_path = os.path.join(models_dir, 'best_model.pt')

    for epoch in range(1, num_epochs + 1):
        train(model, train_loader, optimizer, scheduler, criterion, device, epoch, writer)
        val_loss, val_acc = validate(model, val_loader, criterion, device, epoch, writer)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
            }, best_path)

    # Load best checkpoint (lowest val loss) for final test evaluation
    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"\nLoaded best model from epoch {checkpoint['epoch']} (val_loss={checkpoint['val_loss']:.4f}, val_acc={checkpoint['val_acc']:.2f}%)")
    test(model, test_loader, criterion, device)
    writer.close()


if __name__ == "__main__":
    main()
