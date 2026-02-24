import torch
import torch.nn as nn
from torchvision import models
import timm

class MultiModalCNNFusion(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Load pre-trained efficientnet_b0 model
        self.cnn = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        
        # Create a simple MLP for metadata processing
        self.metadata_mlp = nn.Sequential(
            nn.Linear(26, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Final classification layer
        self.classifier = nn.Sequential(
            nn.Linear(1344, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 6)
        )
    
    def forward(self, x, metadata=None):
        cnn_features = self.cnn.avgpool(self.cnn.features(x)).flatten(1)
        metadata_features = self.metadata_mlp(metadata)
        combined_features = torch.cat([cnn_features, metadata_features], dim=1)
        output = self.classifier(combined_features)
        return output

class SpatialVisionFusion(nn.Module):
    def __init__(self, shared_dim=512):
        super().__init__()
        
        # Load pre-trained ResNet-50 and ViT models
        self.resnet = models.resnet50(pretrained=True)
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)
        
        # Extract features from final layer of CNN
        self.resnet_feature_extractor = nn.Sequential(*list(self.resnet.children())[:-2])  # Remove the final classification layer
        
        # Reshape ResNet features
        self.resnet_projection = nn.Conv2d(2048, shared_dim, kernel_size=1)  # Project ResNet features to shared dimension

        # Linear layers to project features to a common dimension
        self.vit_projection = nn.Linear(768, shared_dim)
        
        # Metadata projection layer (if using metadata features)
        self.metadata_projection = nn.Linear(26, shared_dim)
        
        # Cross Attention Layer
        self.cross_attention = nn.MultiheadAttention(shared_dim, num_heads=8, batch_first=True)
        
        # Final classification layer
        self.classifier = nn.Sequential(
            nn.Linear(shared_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 6)
        )
    
    def forward(self, x, metadata=None):
        # Extract features from ResNet
        resnet_features = self.resnet_feature_extractor(x)
        # (B, 2048, 7, 7)
        resnet_features = self.resnet_projection(resnet_features)
        # (B, shared_dim, 7, 7)
        
        # Extract features from ViT
        vit_features = self.vit.forward_features(x)
        # (B, seq_len, 768) - forward_features returns sequence of tokens
        vit_features = vit_features[:, 0, :]  # Extract class token (B, 768)
        vit_proj = self.vit_projection(vit_features)
        # Linear Function Transforms vector to (B, shared_dim)
        vit_query = vit_proj.unsqueeze(1)
        # (B, 1, shared_dim) - Attention query expects (B, seq_len, embed_dim) format
        
        # Reshape features for cross attention
        resnet_flatten = resnet_features.flatten(2).transpose(1, 2)
        # (B, 49, shared_dim) - Flatten spatial dimensions fro 2D to 1D and transpose for attention format
        
        # Apply cross attention
        attended_features, attention_weights = self.cross_attention(vit_query, resnet_flatten, resnet_flatten, need_weights=True)
        attended_features = attended_features.squeeze(1)
        # (B, shared_dim) - Remove sequence dimension after attention
        
        # Later On: ADDING METADATA FEATURES TO CLASSIFIER
        
        # Metadata is a tensor of (B, 26) - 26 metadata features
        # We can project metadata features to the same shared dimension as vision features using a linear layer
        metadata_features = self.metadata_projection(metadata)
        metadata_features = metadata_features.unsqueeze(1)
        
        # Apply cross attention between metadata features and ResNet features
        meta_attended_features, meta_attention_weights = self.cross_attention(metadata_features, resnet_flatten, resnet_flatten, need_weights=True)
        meta_attended_features = meta_attended_features.squeeze(1)
        
        # Combine attended features with metadata features
        combined_features = torch.cat([attended_features, meta_attended_features], dim=1)
        
        # Pass attended features through the classifier
        output = self.classifier(combined_features)
        
        return output