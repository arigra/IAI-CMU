###############################################################################################
                                        ## Imports ##
###############################################################################################


import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, TensorDataset
from sklearn.covariance import EmpiricalCovariance
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import random
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


# Set page config
st.set_page_config(
    page_title="Enhanced OOD Detector - Mission Control",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CIFAR-100 class names
CIFAR100_CLASSES = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 
    'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 
    'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 
    'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 
    'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
]

###############################################################################################
                                        ## CNN ##
###############################################################################################
class ImprovedCNN(nn.Module):
    """Improved CNN architecture with expandable multi-class head"""
    def __init__(self, num_classes=10):
        super(ImprovedCNN, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Residual blocks
        self.block1 = self._make_residual_block(64, 128, stride=2)
        self.block2 = self._make_residual_block(128, 256, stride=2)
        self.block3 = self._make_residual_block(256, 512, stride=2)
        
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Feature extractor
        self.fc1 = nn.Linear(512, 256)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)  # Feature layer
        self.dropout2 = nn.Dropout(0.5)
        
        # Multi-class classification head (expandable)
        self.num_classes = num_classes
        self.max_classes = 15  # 10 CIFAR-10 + 5 CIFAR-100
        self.fc3 = nn.Linear(128, self.max_classes)
        
        # Track which classes are active
        self.active_classes = list(range(num_classes))  # Start with CIFAR-10 classes
        self.class_names = {}  # Map class_idx to name
        
        # Initialize CIFAR-10 class names
        cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                          'dog', 'frog', 'horse', 'ship', 'truck']
        for i, name in enumerate(cifar10_classes):
            self.class_names[i] = name
        
        # Initialize unused weights to prevent issues
        with torch.no_grad():
            self.fc3.weight[num_classes:].fill_(0)
            self.fc3.bias[num_classes:].fill_(0)
    
    def _make_residual_block(self, in_channels, out_channels, stride=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def add_new_class(self, class_name):
        """Add a new class to the model"""
        if len(self.active_classes) >= self.max_classes:
            raise ValueError(f"Maximum number of classes ({self.max_classes}) reached")
        
        new_class_idx = len(self.active_classes)
        self.active_classes.append(new_class_idx)
        self.class_names[new_class_idx] = class_name
        self.num_classes = len(self.active_classes)
        
        return new_class_idx
    
    def get_active_logits(self, x):
        """Get logits only for active classes"""
        full_logits = self.fc3(x)
        return full_logits[:, :self.num_classes]
    
    def forward(self, x):
        # Initial conv
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Residual blocks
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        # Global average pooling
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        
        # Feature extraction
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        features = F.relu(self.fc2(x))
        x = self.dropout2(features)
        
        # Classification - return only active classes
        logits = self.get_active_logits(x)
        
        return logits, features
    
    def get_class_name(self, class_idx):
        """Get class name from index"""
        return self.class_names.get(class_idx, f"Class_{class_idx}")
    
    def get_class_idx(self, class_name):
        """Get class index from name"""
        for idx, name in self.class_names.items():
            if name == class_name:
                return idx
        return None
    



###############################################################################################
                                    ## Contrastive Loss ##
###############################################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F

class SupConLoss(nn.Module):
    """Fixed Supervised Contrastive Loss implementation"""
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """
        Args:
            features: hidden vector of shape [bsz, feature_dim] or [bsz, n_views, feature_dim].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = features.device
        batch_size = features.shape[0]

        # Handle single view case - don't add unnecessary dimensions
        if len(features.shape) == 2:
            # Features are [batch_size, feature_dim]
            features = features.unsqueeze(1)  # [batch_size, 1, feature_dim]
        elif len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        # Normalize features to unit vectors for better contrastive learning
        features = F.normalize(features, dim=2, p=2)

        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        # compute mean of log-likelihood over positive
        # Only compute loss for samples that have positive pairs
        mask_sum = mask.sum(1)
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask_sum + 1e-12)
        
        # Filter out samples with no positive pairs
        valid_samples = mask_sum > 0
        if valid_samples.sum() == 0:
            return torch.tensor(0.0, requires_grad=True, device=device)
        
        mean_log_prob_pos = mean_log_prob_pos[valid_samples]

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss


class ImprovedContrastiveLoss(nn.Module):
    """Simpler, more stable contrastive loss implementation"""
    def __init__(self, temperature=0.5, margin=2.0):
        super(ImprovedContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.margin = margin
    
    def forward(self, features, labels):
        """
        Args:
            features: feature vectors [batch_size, feature_dim]
            labels: ground truth labels [batch_size]
        """
        batch_size = features.size(0)
        
        # Normalize features
        features = F.normalize(features, dim=1, p=2)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # Create masks
        labels = labels.contiguous().view(-1, 1)
        mask_positive = torch.eq(labels, labels.T).float().to(features.device)
        mask_negative = 1.0 - mask_positive
        
        # Remove diagonal (self-similarity)
        mask_eye = torch.eye(batch_size, device=features.device)
        mask_positive = mask_positive * (1.0 - mask_eye)
        mask_negative = mask_negative * (1.0 - mask_eye)
        
        # Check if we have positive pairs
        if mask_positive.sum() == 0:
            return torch.tensor(0.0, requires_grad=True, device=features.device)
        
        # For numerical stability
        similarity_matrix_stable = similarity_matrix - similarity_matrix.max(dim=1, keepdim=True)[0].detach()
        
        # Compute positive and negative similarities
        pos_similarities = similarity_matrix_stable * mask_positive
        neg_similarities = similarity_matrix_stable * mask_negative
        
        # InfoNCE-style loss
        exp_logits = torch.exp(similarity_matrix_stable)
        
        # Positive pairs
        pos_exp = exp_logits * mask_positive
        
        # All pairs (for denominator)
        all_exp = exp_logits * (1.0 - mask_eye)  # exclude self
        
        # Compute loss for each sample
        losses = []
        for i in range(batch_size):
            pos_sum = pos_exp[i].sum()
            if pos_sum > 0:
                all_sum = all_exp[i].sum()
                loss_i = -torch.log(pos_sum / (all_sum + 1e-12) + 1e-12)
                losses.append(loss_i)
        
        if len(losses) == 0:
            return torch.tensor(0.0, requires_grad=True, device=features.device)
        
        return torch.stack(losses).mean()


# Updated training function to fix the contrastive loss integration
def train_model_with_fixed_contrastive(detector, samples_per_class=100, epochs=50, batch_size=64, lr=0.001, 
                                      num_classes=10, loss_type="CrossEntropy", progress_bar=None, status_text=None):
    """Enhanced training with FIXED contrastive loss"""
    
    # Load data
    train_dataset, test_dataset, detector.num_classes = detector.load_cifar10_subset(samples_per_class, num_classes)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    # Create test loader
    test_transform = detector.get_augmentation_transform(training=False)
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=test_transform
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Create model
    detector.model = detector.create_model(num_classes)
    
    # Store original training data for later use
    detector._store_training_data(train_loader)
    
    # Training setup with improved optimization
    if loss_type == "Contrastive":
        # Use the FIXED supervised contrastive loss
        contrastive_criterion = ImprovedContrastiveLoss(temperature=0.1)  # Lower temperature
        classification_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        use_contrastive = True
        print("🔗 Using FIXED Supervised Contrastive Loss + Classification Loss")
    else:
        classification_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        use_contrastive = False
        
    optimizer = torch.optim.AdamW(detector.model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training loop
    train_accuracies = []
    train_losses = []
    val_accuracies = []
    best_acc = 0
    
    for epoch in range(epochs):
        # Training phase
        detector.model.train()
        running_loss = 0.0
        running_ce_loss = 0.0
        running_contrastive_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            mask = targets < num_classes
            inputs = inputs[mask]
            targets = targets[mask]
            
            if len(inputs) == 0:
                continue
            
            inputs, targets = inputs.to(detector.device), targets.to(detector.device)
            
            optimizer.zero_grad()
            outputs, features = detector.model(inputs)

            # Classification loss
            classification_loss = classification_criterion(outputs, targets)
            
            if use_contrastive:
                # Contrastive loss with FIXED implementation
                contrastive_loss = contrastive_criterion(features, targets)
                
                # Balanced combination - both losses are important
                total_loss = 0.7 * classification_loss + 0.3 * contrastive_loss
                
                # Track individual losses
                running_ce_loss += classification_loss.item()
                running_contrastive_loss += contrastive_loss.item()
            else:
                total_loss = classification_loss

            total_loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(detector.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += total_loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        epoch_accuracy = 100. * correct / total if total > 0 else 0
        epoch_loss = running_loss / len(train_loader)
        train_accuracies.append(epoch_accuracy)
        train_losses.append(epoch_loss)
        
        # Print loss breakdown for debugging
        if use_contrastive and epoch % 10 == 0:
            ce_avg = running_ce_loss / len(train_loader)
            cont_avg = running_contrastive_loss / len(train_loader)
            print(f"Epoch {epoch}: CE Loss: {ce_avg:.4f}, Contrastive Loss: {cont_avg:.4f}, Total: {epoch_loss:.4f}")
        
        # Validation phase
        val_acc = detector._evaluate_model(test_loader)
        val_accuracies.append(val_acc)
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_state = detector.model.state_dict()
        
        scheduler.step()
        
        # Update progress
        if progress_bar:
            progress_bar.progress((epoch + 1) / epochs)
        if status_text:
            status_text.text(f'Epoch {epoch+1}/{epochs}: Loss={epoch_loss:.3f}, Train Acc={epoch_accuracy:.2f}%, Val Acc={val_acc:.2f}%')
    
    # Load best model
    detector.model.load_state_dict(best_model_state)
    
    # Rest of the function remains the same...
    detector._calibrate_temperature(test_loader)
    detector.feature_extractor = lambda x: detector.model(x)[1]
    detector._compute_cluster_statistics(train_loader)
    
    detector.new_class_learner = NewClassLearner(detector.model, detector.device)
    detector.new_class_learner.class_inv_covariances = detector.class_inv_covariances
    
    if detector.training_features is not None:
        detector.new_class_learner.store_original_training_data(
            detector.training_features, detector.training_labels
        )
    
    test_acc = detector._evaluate_model(test_loader)
    
    return {
        'train_accuracies': train_accuracies,
        'train_losses': train_losses,
        'val_accuracies': val_accuracies,
        'test_accuracy': test_acc,
        'best_accuracy': best_acc
    }


###############################################################################################
                                    ## Sample Dicovery ##
###############################################################################################


class SampleDiscoveryEngine:
    """Engine for discovering samples from target classes in random batches"""
    def __init__(self, detector, target_classes, weights):
        self.detector = detector
        self.target_classes = target_classes  # Dict: {class_name: class_idx}
        self.weights = weights
        self.discovery_history = []
        self.found_samples = {class_name: [] for class_name in target_classes.keys()}
        self.all_evaluated_samples = []
        
        # Create the full pool of samples once during initialization
        self._create_sample_pool()
        
    def _create_sample_pool(self):
        """Create the full pool of samples to draw from"""
        transform = self.detector.get_augmentation_transform(training=False)
    
        # Load datasets - USE TEST SET for CIFAR-10 discovery
        cifar10_test = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform  # train=False!
        )
        cifar100_test = torchvision.datasets.CIFAR100(
            root='./data', train=False, download=True, transform=transform
        )
        
        # Create CIFAR-10 test samples pool (100 per class from test set)
        self.cifar10_pool = {}
        for class_idx in range(10):
            class_samples = []
            
            for i, (img, label) in enumerate(cifar10_test):  # Use test set
                if label == class_idx:
                    class_samples.append({
                        'image': img,
                        'label': label,
                        'original_index': i,
                        'class_name': self.detector.cifar10_classes[label],
                        'source_dataset': 'CIFAR-10-test',  # Update label
                        'is_target_class': False,
                        'target_class_name': None
                    })
                    if len(class_samples) >= 100:  # Still limit to 100 per class
                        break
            
            self.cifar10_pool[class_idx] = class_samples
            
        # Create CIFAR-100 target classes pool (50 per selected class)
        self.cifar100_pool = {}
        for class_name, class_idx in self.target_classes.items():
            class_samples = []
            for i, (img, label) in enumerate(cifar100_test):
                if label == class_idx:
                    class_samples.append({
                        'image': img,
                        'label': label,
                        'original_index': i,
                        'class_name': class_name,
                        'source_dataset': 'CIFAR-100',
                        'is_target_class': True,
                        'target_class_name': class_name
                    })
            
            # Limit to 50 samples per class or all if less than 50
            if len(class_samples) > 50:
                class_samples = random.sample(class_samples, 50)
            
            self.cifar100_pool[class_name] = class_samples
        
        # Combine all samples into one master pool
        self.master_pool = []
        
        # Add all CIFAR-10 unseen samples
        for class_samples in self.cifar10_pool.values():
            self.master_pool.extend(class_samples)
        
        # Add all CIFAR-100 target class samples
        for class_samples in self.cifar100_pool.values():
            self.master_pool.extend(class_samples)
        
        print(f"Created sample pool with {len(self.master_pool)} total samples:")
        print(f"  - CIFAR-10 unseen: {sum(len(samples) for samples in self.cifar10_pool.values())}")
        print(f"  - CIFAR-100 target: {sum(len(samples) for samples in self.cifar100_pool.values())}")
    
    def sample_random_batch(self, batch_size=100):
        """Sample random batch of specified size from the master pool"""
        if batch_size > len(self.master_pool):
            print(f"Warning: Requested batch size {batch_size} is larger than pool size {len(self.master_pool)}")
            batch_size = len(self.master_pool)
        
        # Randomly sample from master pool
        selected_samples = random.sample(self.master_pool, batch_size)
        
        # Convert to the expected format
        batch_data = {
            'images': [sample['image'] for sample in selected_samples],
            'labels': [sample['label'] for sample in selected_samples],
            'class_names': [sample['class_name'] for sample in selected_samples],
            'indices': [sample['original_index'] for sample in selected_samples],
            'is_target_class': [sample['is_target_class'] for sample in selected_samples],
            'target_class_name': [sample['target_class_name'] for sample in selected_samples],
            'source_dataset': [sample['source_dataset'] for sample in selected_samples]
        }
        
        return batch_data
    
    def evaluate_batch(self, batch_data):
        """Evaluate batch for pure OOD detection (no prior knowledge of target classes)"""
        # Extract features
        features = self.detector.new_class_learner.extract_features(batch_data['images'])
        
        # Compute selection scores based ONLY on CIFAR-10 knowledge
        scores = self.compute_pure_ood_scores(
            batch_data['images'], 
            features, 
            self.detector.class_means,  # Only CIFAR-10 class means
            self.weights
        )
        
        # Get top 10 samples
        top_indices = np.argsort(scores)[-10:][::-1]
        
        # Evaluate detection for each sample (but model doesn't know target classes)
        detection_results = []
        for i, img in enumerate(batch_data['images']):
            detection_result = self._evaluate_sample_detection(
                img, batch_data['class_names'][i], batch_data['is_target_class'][i]
            )
            detection_results.append(detection_result)
        
        # Store results
        batch_results = {
            'batch_id': len(self.discovery_history),
            'total_samples': len(batch_data['images']),
            'target_samples_in_batch': sum(batch_data['is_target_class']),
            'top_10_indices': top_indices,
            'all_scores': scores,
            'all_detection_results': detection_results,
            'batch_data': batch_data,
            'timestamp': time.time()
        }
        
        # Store found target samples (for evaluation purposes)
        for idx in range(len(batch_data['images'])):
            if batch_data['is_target_class'][idx]:
                target_class = batch_data['target_class_name'][idx]
                sample_info = {
                    'image': batch_data['images'][idx],
                    'score': scores[idx],
                    'detection_result': detection_results[idx],
                    'batch_id': batch_results['batch_id'],
                    'original_index': batch_data['indices'][idx],
                    'rank_in_top_10': idx in top_indices,
                    'rank_position': list(top_indices).index(idx) + 1 if idx in top_indices else None,
                    'user_assigned_class': None  # To be filled during manual selection
                }
                self.found_samples[target_class].append(sample_info)
        
        self.discovery_history.append(batch_results)
        return batch_results

    def compute_pure_ood_scores(self, images, features, cifar10_class_means, weights):
        """Compute scores based only on CIFAR-10 knowledge (no target class knowledge)"""
        scores = []
        criteria = {
            'diversity': [],
            'distance_from_cifar10': [],
            'mahalanobis_distance': [],
            'consistency': [],
            'confidence': []
        }
        
        # 1. Diversity (among all discovered samples so far)
        all_discovered_features = []
        for samples in self.found_samples.values():
            for sample in samples:
                if 'features' in sample:
                    all_discovered_features.append(sample['features'])
        
        if len(all_discovered_features) > 0:
            existing_features = np.array(all_discovered_features)
            for feat in features:
                distances = np.linalg.norm(existing_features - feat, axis=1)
                diversity = np.mean(distances) if len(distances) > 0 else 1.0
                criteria['diversity'].append(diversity)
        else:
            criteria['diversity'] = [1.0] * len(features)
        
        # 2. Distance from CIFAR-10 classes ONLY
        for feat in features:
            min_dist = float('inf')
            for class_mean in cifar10_class_means.values():  # Only CIFAR-10
                dist = np.linalg.norm(feat - class_mean)
                min_dist = min(min_dist, dist)
            criteria['distance_from_cifar10'].append(min_dist if min_dist != float('inf') else 1.0)
        
        # 3. Mahalanobis distance from CIFAR-10 classes
        if hasattr(self.detector, 'class_inv_covariances'):
            mahal_distances = self.detector.new_class_learner.compute_mahalanobis_distance(
                features, cifar10_class_means, self.detector.class_inv_covariances
            )
            criteria['mahalanobis_distance'] = mahal_distances.tolist()
        else:
            criteria['mahalanobis_distance'] = criteria['distance_from_cifar10'].copy()
        
        # 4. Consistency
        for feat in features:
            consistency = 1.0 / (1.0 + np.var(feat))
            criteria['consistency'].append(consistency)
        
        # 5. Confidence (uncertainty about CIFAR-10 classes)
        confidences = []
        for img in images:
            with torch.no_grad():
                if isinstance(img, np.ndarray):
                    img = torch.from_numpy(img)
                img = img.unsqueeze(0).to(self.detector.device)
                logits, _ = self.detector.model(img)
                probs = F.softmax(logits, dim=1)
                max_cifar10_prob = probs.max().item()
                confidences.append(1.0 - max_cifar10_prob)  # High score for low confidence
        criteria['confidence'] = confidences
        
        # Normalize criteria
        for key in criteria:
            values = np.array(criteria[key])
            if values.max() > values.min():
                criteria[key] = ((values - values.min()) / (values.max() - values.min())).tolist()
            else:
                criteria[key] = [0.5] * len(values)
        
        # Compute weighted scores
        for i in range(len(features)):
            score = sum(weights.get(key, 0) * criteria[key][i] for key in criteria)
            scores.append(score)
        
        return np.array(scores)
    
    def _evaluate_sample_detection(self, img, true_class_name, is_target_class):
        """Evaluate if the model can detect this sample as novel/interesting"""
        self.detector.model.eval()
        
        with torch.no_grad():
            if isinstance(img, np.ndarray):
                img_tensor = torch.from_numpy(img)
            else:
                img_tensor = img
            
            img_tensor = img_tensor.unsqueeze(0).to(self.detector.device)
            logits, features = self.detector.model(img_tensor)
            
            # Apply temperature scaling
            scaled_logits = logits / self.detector.temperature
            probs = F.softmax(scaled_logits, dim=1)
            
            max_prob = probs.max().item()
            predicted_class_idx = probs.argmax().item()
            predicted_class_name = self.detector.cifar10_classes[predicted_class_idx] if predicted_class_idx < len(self.detector.cifar10_classes) else "Unknown"
            
            # Check if model detected this as novel (low confidence)
            novelty_threshold = 0.7  # Can be made configurable
            detected_as_novel = max_prob < novelty_threshold
            
            return {
                'true_class': true_class_name,
                'is_target_class': is_target_class,
                'predicted_class': predicted_class_name,
                'max_confidence': max_prob,
                'detected_as_novel': detected_as_novel,
                'correct_detection': detected_as_novel if is_target_class else not detected_as_novel
            }
    
    def get_pool_info(self):
        """Get information about the sample pool"""
        return {
            'total_samples': len(self.master_pool),
            'cifar10_samples': sum(len(samples) for samples in self.cifar10_pool.values()),
            'cifar100_samples': sum(len(samples) for samples in self.cifar100_pool.values()),
            'target_classes': list(self.target_classes.keys())
        }
    
    def get_found_samples_summary(self):
        """Get summary of found target samples"""
        summary = {}
        for class_name, samples in self.found_samples.items():
            if samples:
                total_found = len(samples)
                correctly_detected = sum(1 for s in samples if s['detection_result']['correct_detection'])
                in_top_10 = sum(1 for s in samples if s['rank_in_top_10'])
                avg_score = np.mean([s['score'] for s in samples])
                
                summary[class_name] = {
                    'total_found': total_found,
                    'correctly_detected': correctly_detected,
                    'detection_rate': correctly_detected / total_found if total_found > 0 else 0,
                    'in_top_10': in_top_10,
                    'top_10_rate': in_top_10 / total_found if total_found > 0 else 0,
                    'avg_score': avg_score
                }
        
        return summary


###############################################################################################
                                ## New Class Learner ##
###############################################################################################

class NewClassLearner:
    """Enhanced version for multi-class continual learning"""
    def __init__(self, base_model, device):
        self.base_model = base_model
        self.device = device
        self.new_class_samples = {}  # Per class
        self.new_class_features = {}  # Per class
        self.selection_scores = {}  # Per class
        self.selection_criteria = {}
        self.trained_classes = set()
        
        # Store original CIFAR-10 training data for negative sampling
        self.original_training_data = None
        self.original_training_labels = None
        
    def store_original_training_data(self, features, labels):
        """Store original CIFAR-10 training data for use as negatives"""
        self.original_training_data = features
        self.original_training_labels = labels
    
    def train_new_classes(self, selected_samples_dict, epochs=30, lr=0.001, freeze_backbone=True):
        """Train multiple new classes simultaneously with option to freeze backbone"""
        if not selected_samples_dict:
            return {}
        
        # Add new classes to model
        class_mapping = {}
        for class_name in selected_samples_dict.keys():
            if class_name not in self.trained_classes:
                class_idx = self.base_model.add_new_class(class_name)
                class_mapping[class_name] = class_idx
                self.trained_classes.add(class_name)
        
        # Prepare training data
        all_images = []
        all_labels = []
        
        # Add new class samples
        for class_name, samples in selected_samples_dict.items():
            if samples:
                class_idx = class_mapping.get(class_name, self.base_model.get_class_idx(class_name))
                for sample in samples:
                    all_images.append(sample['image'])
                    all_labels.append(class_idx)
        
        if not all_images:
            return {}
        
        # Create training dataset
        train_dataset = TensorDataset(
            torch.stack(all_images),
            torch.tensor(all_labels)
        )
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        # FREEZE/UNFREEZE BACKBONE
        if freeze_backbone:
            # Freeze backbone (feature extractor)
            for name, param in self.base_model.named_parameters():
                if 'fc3' not in name:  # Don't freeze the classification head
                    param.requires_grad = False
            
            # Only optimize classification head
            optimizer = torch.optim.Adam(
                [p for p in self.base_model.parameters() if p.requires_grad], 
                lr=lr
            )
            print("Training with FROZEN backbone (only classification head updated)")
        else:
            # Train entire model
            for param in self.base_model.parameters():
                param.requires_grad = True
            
            optimizer = torch.optim.Adam(self.base_model.parameters(), lr=lr)
            print("Training with UNFROZEN backbone (entire model updated)")
        
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        results = {}
        epoch_losses = []
        epoch_accuracies = []
        
        self.base_model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for batch_images, batch_labels in train_loader:
                batch_images = batch_images.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                logits, features = self.base_model(batch_images)
                loss = criterion(logits, batch_labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Statistics
                total_loss += loss.item()
                _, predicted = logits.max(1)
                total += batch_labels.size(0)
                correct += predicted.eq(batch_labels).sum().item()
            
            epoch_loss = total_loss / len(train_loader)
            epoch_accuracy = 100. * correct / total
            
            epoch_losses.append(epoch_loss)
            epoch_accuracies.append(epoch_accuracy)
        
        # Unfreeze all parameters after training (for future use)
        for param in self.base_model.parameters():
            param.requires_grad = True
        
        # Store results for all trained classes
        for class_name in selected_samples_dict.keys():
            if selected_samples_dict[class_name]:
                results[class_name] = {
                    'losses': epoch_losses,
                    'accuracies': epoch_accuracies,
                    'final_loss': epoch_losses[-1] if epoch_losses else 0,
                    'final_accuracy': epoch_accuracies[-1] if epoch_accuracies else 0,
                    'samples_used': len(selected_samples_dict[class_name]),
                    'class_name': class_name,
                    'class_idx': class_mapping.get(class_name, self.base_model.get_class_idx(class_name)),
                    'backbone_frozen': freeze_backbone
                }
        
        return results
    def incremental_train_single_class(self, class_name, samples, epochs=30, lr=0.001):
        """Train a single new class incrementally"""
        if not samples:
            return None
        
        # Add new class to model
        if class_name not in self.trained_classes:
            class_idx = self.base_model.add_new_class(class_name)
            self.trained_classes.add(class_name)
        else:
            class_idx = self.base_model.get_class_idx(class_name)
        
        # Prepare training data - mix of new class samples and some old samples
        train_images = []
        train_labels = []
        
        # Add new class samples
        for sample in samples:
            train_images.append(sample['image'])
            train_labels.append(class_idx)
        
        # Add some samples from existing classes to prevent forgetting
        if self.original_training_data is not None:
            # Sample some original training data
            n_old_samples = min(len(samples) * 2, len(self.original_training_data))
            old_indices = np.random.choice(len(self.original_training_data), n_old_samples, replace=False)
            
            # Note: This would require storing original images, not just features
            # For now, we'll focus on the new class training
        
        # Create dataset
        train_dataset = TensorDataset(
            torch.stack(train_images),
            torch.tensor(train_labels)
        )
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        # Training setup
        optimizer = torch.optim.Adam(self.base_model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        epoch_losses = []
        epoch_accuracies = []
        
        self.base_model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for batch_images, batch_labels in train_loader:
                batch_images = batch_images.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                logits, features = self.base_model(batch_images)
                loss = criterion(logits, batch_labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Statistics
                total_loss += loss.item()
                _, predicted = logits.max(1)
                total += batch_labels.size(0)
                correct += predicted.eq(batch_labels).sum().item()
            
            epoch_loss = total_loss / len(train_loader)
            epoch_accuracy = 100. * correct / total
            
            epoch_losses.append(epoch_loss)
            epoch_accuracies.append(epoch_accuracy)
        
        return {
            'losses': epoch_losses,
            'accuracies': epoch_accuracies,
            'final_loss': epoch_losses[-1] if epoch_losses else 0,
            'final_accuracy': epoch_accuracies[-1] if epoch_accuracies else 0,
            'samples_used': len(samples),
            'class_name': class_name,
            'class_idx': class_idx
        }
    
    def evaluate_all_classes(self, test_loader):
        """Evaluate model on all classes"""
        self.base_model.eval()
        correct = 0
        total = 0
        class_correct = {}
        class_total = {}
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs, _ = self.base_model(inputs)
                
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # Per-class statistics
                for i in range(targets.size(0)):
                    label = targets[i].item()
                    class_name = self.base_model.get_class_name(label)
                    
                    if class_name not in class_correct:
                        class_correct[class_name] = 0
                        class_total[class_name] = 0
                    
                    class_total[class_name] += 1
                    if predicted[i] == label:
                        class_correct[class_name] += 1
        
        # Calculate per-class accuracies
        class_accuracies = {}
        for class_name in class_total:
            class_accuracies[class_name] = 100. * class_correct[class_name] / class_total[class_name]
        
        return {
            'overall_accuracy': 100. * correct / total,
            'class_accuracies': class_accuracies,
            'total_samples': total,
            'correct_samples': correct
        }
    
    def init_class(self, class_name):
        """Initialize tracking for a new class"""
        if class_name not in self.new_class_samples:
            self.new_class_samples[class_name] = []
            self.new_class_features[class_name] = []
            self.selection_scores[class_name] = []
    
    def extract_features(self, images):
        """Extract features using the base model"""
        self.base_model.eval()
        features_list = []
        
        with torch.no_grad():
            for img in images:
                if isinstance(img, np.ndarray):
                    img = torch.from_numpy(img)
                img = img.unsqueeze(0).to(self.device)
                _, features = self.base_model(img)
                features_list.append(features.cpu().numpy().squeeze())
        
        return np.array(features_list)
    
    def compute_mahalanobis_distance(self, features, class_means, class_inv_covariances):
        """Compute Mahalanobis distance from existing classes"""
        min_distances = []
        
        for feat in features:
            min_dist = float('inf')
            for class_idx in class_means:
                if class_idx in class_inv_covariances:
                    diff = feat - class_means[class_idx]
                    # Mahalanobis distance
                    dist = np.sqrt(np.dot(np.dot(diff, class_inv_covariances[class_idx]), diff))
                    min_dist = min(min_dist, dist)
            min_distances.append(min_dist if min_dist != float('inf') else 1.0)
        
        return np.array(min_distances)


###############################################################################################
                                    ## OOD Detector ##
###############################################################################################
class EnhancedOODDetector:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = None
        self.feature_extractor = None
        self.class_means = None
        self.class_covariances = None
        self.class_inv_covariances = None
        self.training_features = None
        self.training_labels = None
        self.training_images = None  # Store original training images
        self.pca = None
        self.tsne = None
        self.cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                               'dog', 'frog', 'horse', 'ship', 'truck']
        self.temperature = 1.0
        self.new_class_learner = None
        
    def create_model(self, num_classes=10):
        """Create improved CNN model with multi-class support"""
        return ImprovedCNN(num_classes).to(self.device)
    
    def get_augmentation_transform(self, training=True):
        """Enhanced data augmentation for better generalization"""
        if training:
            return transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        else:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
    
    def load_cifar10_subset(self, samples_per_class=5000, num_classes=10):
        """Load CIFAR-10 with enhanced augmentation"""
        train_transform = self.get_augmentation_transform(training=True)
        test_transform = self.get_augmentation_transform(training=False)
        
        # Load full datasets
        train_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=train_transform
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=test_transform
        )
        
        # Create subset with specified samples per class
        train_indices = []
        class_counts = {i: 0 for i in range(num_classes)}
        
        for idx, (_, label) in enumerate(train_dataset):
            if label < num_classes and class_counts[label] < samples_per_class:
                train_indices.append(idx)
                class_counts[label] += 1
                
            if all(count >= samples_per_class for count in class_counts.values()):
                break
        
        train_subset = Subset(train_dataset, train_indices)
        
        if samples_per_class >= 5000:  # Use full training set
            train_subset = train_dataset
        
        return train_subset, test_dataset, num_classes
    
    def train_model(self, samples_per_class=100, epochs=50, batch_size=64, lr=0.001, 
               num_classes=10, loss_type="CrossEntropy", progress_bar=None, status_text=None):
        """Enhanced training with better optimization"""
        
        # Load data
        train_dataset, test_dataset, self.num_classes = self.load_cifar10_subset(samples_per_class, num_classes)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        
        # Create test loader
        test_transform = self.get_augmentation_transform(training=False)
        test_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=test_transform
        )
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        # Create model
        self.model = self.create_model(num_classes)
        
        # Store original training data for later use
        self._store_training_data(train_loader)
        
        # Training setup with improved optimization
        if loss_type == "Contrastive":
            # Use the better supervised contrastive loss
            contrastive_criterion = SupConLoss(temperature=0.5, base_temperature=0.5)
            classification_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
            use_contrastive = True
            st.info("🔗 Using Supervised Contrastive Loss + Classification Loss")
        else:
            classification_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
            use_contrastive = False
            
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        # Training loop
        train_accuracies = []
        train_losses = []
        val_accuracies = []
        best_acc = 0
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                mask = targets < num_classes
                inputs = inputs[mask]
                targets = targets[mask]
                
                if len(inputs) == 0:
                    continue
                
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                outputs, features = self.model(inputs)

                if use_contrastive:
                    # Combine classification loss with contrastive loss
                    classification_loss = classification_criterion(outputs, targets)
                    
                    # Contrastive loss expects normalized features
                    normalized_features = F.normalize(features, dim=1)
                    contrastive_loss = contrastive_criterion(normalized_features.unsqueeze(1), targets)
                    
                    # Weighted combination - classification is primary, contrastive is auxiliary
                    loss = classification_loss + 0.05 * contrastive_loss
                    
                    # For debugging (you can remove this)
                    if batch_idx == 0 and epoch % 10 == 0:
                        print(f"Epoch {epoch}: CE Loss: {classification_loss:.4f}, Contrastive Loss: {contrastive_loss:.4f}")
                else:
                    loss = classification_criterion(outputs, targets)

                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            
            epoch_accuracy = 100. * correct / total if total > 0 else 0
            epoch_loss = running_loss / len(train_loader)
            train_accuracies.append(epoch_accuracy)
            train_losses.append(epoch_loss)
            
            # Validation phase
            val_acc = self._evaluate_model(test_loader)
            val_accuracies.append(val_acc)
            
            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                best_model_state = self.model.state_dict()
            
            scheduler.step()
            
            # Update progress
            if progress_bar:
                progress_bar.progress((epoch + 1) / epochs)
            if status_text:
                status_text.text(f'Epoch {epoch+1}/{epochs}: Loss={epoch_loss:.3f}, Train Acc={epoch_accuracy:.2f}%, Val Acc={val_acc:.2f}%')
        
        # Load best model
        self.model.load_state_dict(best_model_state)
        
        # Calibrate temperature
        self._calibrate_temperature(test_loader)
        
        # Create feature extractor
        self.feature_extractor = lambda x: self.model(x)[1]
        
        # Compute cluster statistics
        self._compute_cluster_statistics(train_loader)
        
        # Initialize new class learner
        self.new_class_learner = NewClassLearner(self.model, self.device)
        self.new_class_learner.class_inv_covariances = self.class_inv_covariances
        
        # Store original training data in the new class learner
        if self.training_features is not None:
            self.new_class_learner.store_original_training_data(
                self.training_features, self.training_labels
            )
        
        # Final evaluation
        test_acc = self._evaluate_model(test_loader)
        
        return {
            'train_accuracies': train_accuracies,
            'train_losses': train_losses,
            'val_accuracies': val_accuracies,
            'test_accuracy': test_acc,
            'best_accuracy': best_acc
        }
    
    def _store_training_data(self, train_loader):
        """Store original training data for continual learning"""
        self.model.eval()
        all_images = []
        all_features = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, targets in train_loader:
                mask = targets < self.num_classes
                inputs = inputs[mask]
                targets = targets[mask]
                
                if len(inputs) == 0:
                    continue
                
                inputs = inputs.to(self.device)
                _, features = self.model(inputs)
                
                all_images.extend(inputs.cpu())
                all_features.append(features.cpu().numpy())
                all_labels.extend(targets.cpu().numpy())
        
        if all_features:
            self.training_images = all_images
            self.training_features = np.concatenate(all_features, axis=0)
            self.training_labels = np.array(all_labels)
    
    def _calibrate_temperature(self, loader):
        """Temperature scaling for confidence calibration"""
        self.model.eval()
        logits_list = []
        labels_list = []
        
        with torch.no_grad():
            for inputs, targets in loader:
                mask = targets < self.num_classes
                inputs = inputs[mask]
                targets = targets[mask]
                
                if len(inputs) == 0:
                    continue
                
                inputs = inputs.to(self.device)
                outputs, _ = self.model(inputs)
                logits_list.append(outputs)
                labels_list.append(targets)
        
        if not logits_list:
            return
            
        logits = torch.cat(logits_list).cpu()
        labels = torch.cat(labels_list).cpu()
        
        # Find optimal temperature
        temps = torch.linspace(0.1, 5.0, 50)
        best_temp = 1.0
        best_nll = float('inf')
        
        for temp in temps:
            scaled_logits = logits / temp
            nll = F.cross_entropy(scaled_logits, labels).item()
            if nll < best_nll:
                best_nll = nll
                best_temp = temp.item()
        
        self.temperature = best_temp
    
    def _compute_cluster_statistics(self, train_loader):
        """Compute class means and covariances with improved numerical stability"""
        self.model.eval()
        
        all_features = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, targets in train_loader:
                mask = targets < self.num_classes
                inputs = inputs[mask]
                targets = targets[mask]
                
                if len(inputs) == 0:
                    continue
                
                inputs = inputs.to(self.device)
                _, features = self.model(inputs)
                all_features.append(features.cpu().numpy())
                all_labels.append(targets.numpy())
        
        if not all_features:
            return
        
        all_features = np.concatenate(all_features, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        # Compute class-wise statistics
        self.class_means = {}
        self.class_covariances = {}
        self.class_inv_covariances = {}
        
        for class_idx in range(self.num_classes):
            class_features = all_features[all_labels == class_idx]
            
            if len(class_features) == 0:
                continue
            
            class_mean = np.mean(class_features, axis=0)
            self.class_means[class_idx] = class_mean
            
            if len(class_features) > 1:
                # Center the features
                centered = class_features - class_mean
                
                # Compute empirical covariance
                emp_cov = np.dot(centered.T, centered) / (len(class_features) - 1)
                
                # Add regularization to improve numerical stability
                feature_dim = emp_cov.shape[0]
                
                # Method 1: Diagonal regularization
                regularization = 1e-4
                regularized_cov = emp_cov + regularization * np.eye(feature_dim)
                
                # Method 2: Shrinkage estimation for additional stability
                shrinkage_target = np.diag(np.diag(emp_cov))
                shrinkage = min(0.2, 1.0 / max(1, len(class_features)))  # Adaptive shrinkage
                class_cov = (1 - shrinkage) * regularized_cov + shrinkage * shrinkage_target
                
                self.class_covariances[class_idx] = class_cov
                
                # Compute inverse with improved numerical stability
                try:
                    # Try Cholesky decomposition first (faster and more stable)
                    L = np.linalg.cholesky(class_cov)
                    class_inv_cov = np.linalg.inv(L.T) @ np.linalg.inv(L)
                except np.linalg.LinAlgError:
                    # If Cholesky fails, use SVD-based pseudo-inverse with better conditioning
                    try:
                        U, s, Vt = np.linalg.svd(class_cov, full_matrices=False)
                        # Filter out very small singular values for stability
                        s_thresh = max(1e-8, s[0] * 1e-12)  # Relative threshold
                        s_inv = np.where(s > s_thresh, 1.0 / s, 0.0)
                        class_inv_cov = (Vt.T * s_inv) @ Vt
                    except np.linalg.LinAlgError:
                        # Final fallback: use identity matrix
                        print(f"Warning: Using identity matrix for class {class_idx} covariance inverse")
                        class_inv_cov = np.eye(feature_dim)
                
                self.class_inv_covariances[class_idx] = class_inv_cov
                
            else:
                # Single sample case
                feature_dim = class_features.shape[1]
                self.class_covariances[class_idx] = np.eye(feature_dim)
                self.class_inv_covariances[class_idx] = np.eye(feature_dim)
        
    def _evaluate_model(self, test_loader):
        """Evaluate model accuracy on active classes"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                # Only evaluate on classes that the model knows
                mask = targets < self.model.num_classes
                inputs = inputs[mask]
                targets = targets[mask]
                
                if len(inputs) == 0:
                    continue
                
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs, _ = self.model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        return 100. * correct / total if total > 0 else 0
    
    def predict_with_class_names(self, images):
        """Predict with class names for all active classes"""
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for img in images:
                if isinstance(img, np.ndarray):
                    img = torch.from_numpy(img)
                
                img = img.unsqueeze(0).to(self.device)
                logits, _ = self.model(img)
                
                # Apply temperature scaling
                scaled_logits = logits / self.temperature
                probs = F.softmax(scaled_logits, dim=1)
                
                max_prob = probs.max().item()
                predicted_class_idx = probs.argmax().item()
                predicted_class_name = self.model.get_class_name(predicted_class_idx)
                
                predictions.append({
                    'class_idx': predicted_class_idx,
                    'class_name': predicted_class_name,
                    'confidence': max_prob,
                    'probabilities': probs.cpu().numpy().squeeze()
                })
        
        return predictions
    def compute_dimensionality_reduction(self, features=None, labels=None, method='both', **kwargs):
        """
        Compute PCA and/or t-SNE on features
        
        Args:
            features: Feature array to reduce. If None, uses stored training features
            labels: Labels for coloring. If None, uses stored training labels
            method: 'pca', 'tsne', or 'both'
        
        Returns:
            dict with reduced features and metadata
        """
        if features is None:
            if self.training_features is None:
                raise ValueError("No features available. Train model first or provide features.")
            features = self.training_features
            labels = self.training_labels
        
        results = {}
        
        if method in ['pca', 'both']:
            # PCA
            self.pca = PCA(n_components=2)
            pca_features = self.pca.fit_transform(features)
            
            results['pca'] = {
                'features': pca_features,
                'explained_variance_ratio': self.pca.explained_variance_ratio_,
                'cumulative_variance': np.cumsum(self.pca.explained_variance_ratio_),
                'labels': labels
            }
        
        if method in ['tsne', 'both']:
            # t-SNE with customizable parameters
            perplexity = kwargs.get('tsne_perplexity', min(50, max(5, len(features) // 3)))
            learning_rate = kwargs.get('tsne_learning_rate', 200.0)
            
            self.tsne = TSNE(
                n_components=2, 
                random_state=42, 
                perplexity=min(perplexity, len(features) - 1),  # Ensure perplexity is valid
                n_iter=3000,  # More iterations for better convergence
                learning_rate=learning_rate,
                init='pca',  # Initialize with PCA for better results
                early_exaggeration=12.0,  # Higher exaggeration for better separation
                min_grad_norm=1e-7  # Better convergence
            )
            tsne_features = self.tsne.fit_transform(features)
            
            results['tsne'] = {
                'features': tsne_features,
                'labels': labels
            }
        
        return results

    def plot_dimensionality_reduction(self, reduction_results, class_names=None, title_suffix="", **viz_options):
        """
        Plot PCA and/or t-SNE results
        
        Args:
            reduction_results: Output from compute_dimensionality_reduction
            class_names: Optional list of class names for legend
            title_suffix: Additional text for plot titles
        
        Returns:
            Plotly figure object
        """
        n_plots = len(reduction_results)
        
        if n_plots == 1:
            fig = go.Figure()
            method = list(reduction_results.keys())[0]
            data = reduction_results[method]
            
            # Add all traces for this method
            traces = self._create_scatter_traces(data, method, class_names, **viz_options)
            for trace in traces:
                fig.add_trace(trace)
            
            plot_height = viz_options.get('plot_height', 500)
            fig.update_layout(
                title=f"{method.upper()} Visualization {title_suffix}",
                xaxis_title=f"{method.upper()}1",
                yaxis_title=f"{method.upper()}2",
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(size=12),
                height=plot_height
            )
            
        else:
            # Create subplots for both PCA and t-SNE
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=[f'{method.upper()} {title_suffix}' for method in reduction_results.keys()],
                horizontal_spacing=0.1  # Better spacing between plots
            )
            
            for i, (method, data) in enumerate(reduction_results.items()):
                traces = self._create_scatter_traces(data, method, class_names, **viz_options)
                for trace in traces:
                    # Only show legend for first subplot to avoid duplicates
                    if i > 0:
                        trace.showlegend = False
                    fig.add_trace(trace, row=1, col=i+1)
            
            # Update axis labels
            methods = list(reduction_results.keys())
            if len(methods) >= 1:
                fig.update_xaxes(title_text=f"{methods[0].upper()}1", row=1, col=1)
                fig.update_yaxes(title_text=f"{methods[0].upper()}2", row=1, col=1)
            if len(methods) >= 2:
                fig.update_xaxes(title_text=f"{methods[1].upper()}1", row=1, col=2)
                fig.update_yaxes(title_text=f"{methods[1].upper()}2", row=1, col=2)
        
        plot_height = viz_options.get('plot_height', 500)
        fig.update_layout(
            height=plot_height, 
            showlegend=True,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(size=12)
        )
        return fig

    def _create_scatter_traces(self, data, method, class_names=None, **viz_options):
        """Helper method to create scatter traces for dimensionality reduction plots"""
        features = data['features']
        labels = data['labels']
        
        # Get visualization options with defaults
        marker_size = viz_options.get('marker_size', 10)
        marker_opacity = viz_options.get('marker_opacity', 0.8)
        use_borders = viz_options.get('use_borders', True)
        color_scheme = viz_options.get('color_scheme', 'Set1')
        
        # Create color mapping based on selected scheme
        color_palettes = {
            'Set1': px.colors.qualitative.Set1,
            'Dark2': px.colors.qualitative.Dark2,
            'Pastel1': px.colors.qualitative.Pastel1,
            'Set3': px.colors.qualitative.Set3,
            'Viridis': px.colors.sequential.Viridis,
            'Plasma': px.colors.sequential.Plasma
        }
        colors = color_palettes.get(color_scheme, px.colors.qualitative.Set1)
        
        # Create traces for each class
        traces = []
        unique_labels = np.unique(labels)
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            class_name = class_names[label] if class_names and label < len(class_names) else f"Class {label}"
            
            # Configure marker style based on options
            marker_config = dict(
                color=colors[i % len(colors)],
                size=marker_size,
                opacity=marker_opacity
            )
            
            if use_borders:
                marker_config['line'] = dict(width=1, color='white')
            
            trace = go.Scatter(
                x=features[mask, 0],
                y=features[mask, 1],
                mode='markers',
                name=class_name,
                marker=marker_config,
                hovertemplate=f'{class_name}<br>' +
                            f'{method.upper()}1: %{{x:.2f}}<br>' +
                            f'{method.upper()}2: %{{y:.2f}}<extra></extra>'
            )
            traces.append(trace)
        
        return traces

    def visualize_feature_space_evolution(self, include_new_classes=True):
        """
        Visualize how the feature space evolves with new classes
        
        Args:
            include_new_classes: Whether to include newly learned class features
        
        Returns:
            Plotly figure showing feature space evolution
        """
        if self.training_features is None:
            raise ValueError("No training features available")
        
        all_features = [self.training_features]
        all_labels = [self.training_labels]
        data_sources = ['Original CIFAR-10']
        
        # Add new class features if available and requested
        if include_new_classes and hasattr(self.new_class_learner, 'new_class_features'):
            for class_name, features in self.new_class_learner.new_class_features.items():
                if len(features) > 0:
                    all_features.append(features)
                    # Create labels for new class (use class index from model)
                    class_idx = self.model.get_class_idx(class_name)
                    class_labels = np.full(len(features), class_idx if class_idx is not None else len(data_sources))
                    all_labels.append(class_labels)
                    data_sources.append(f'New Class: {class_name}')
        
        # Combine all features
        combined_features = np.vstack(all_features)
        combined_labels = np.concatenate(all_labels)
        
        # Create source labels for coloring
        source_labels = []
        for i, source in enumerate(data_sources):
            source_labels.extend([i] * len(all_features[i]))
        source_labels = np.array(source_labels)
        
        # Compute dimensionality reduction
        reduction_results = self.compute_dimensionality_reduction(
            combined_features, combined_labels, method='both'
        )
        
        # Create enhanced visualization
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['PCA: Feature Space Evolution', 't-SNE: Feature Space Evolution']
        )
        
        # Color by data source
        colors = px.colors.qualitative.Set1
        
        for method_idx, (method, data) in enumerate(reduction_results.items()):
            features = data['features']
            
            # Plot by source
            for source_idx, source_name in enumerate(data_sources):
                mask = source_labels == source_idx
                if np.any(mask):
                    trace = go.Scatter(
                        x=features[mask, 0],
                        y=features[mask, 1],
                        mode='markers',
                        name=source_name,
                        marker=dict(
                            color=colors[source_idx % len(colors)],
                            size=6,
                            opacity=0.7
                        ),
                        showlegend=(method_idx == 0),  # Only show legend for first subplot
                        hovertemplate=f'{source_name}<br>' +
                                    f'{method.upper()}1: %{{x:.2f}}<br>' +
                                    f'{method.upper()}2: %{{y:.2f}}<extra></extra>'
                    )
                    fig.add_trace(trace, row=1, col=method_idx + 1)
        
        # Update axis labels
        fig.update_xaxes(title_text="PC1", row=1, col=1)
        fig.update_yaxes(title_text="PC2", row=1, col=1)
        fig.update_xaxes(title_text="t-SNE1", row=1, col=2)
        fig.update_yaxes(title_text="t-SNE2", row=1, col=2)
        
        fig.update_layout(height=500, title="Feature Space Evolution")
        return fig
        



###############################################################################################
                                        ## Visualize ##
###############################################################################################



def plot_bootstrap_results(bootstrap_results):
    """Plot bootstrap training results for all classes"""
    if not bootstrap_results:
        return go.Figure()
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Bootstrap Training Loss', 'Bootstrap Training Accuracy')
    )
    
    colors = px.colors.qualitative.Set1
    
    for i, (class_name, result) in enumerate(bootstrap_results.items()):
        if result:
            epochs = list(range(1, len(result['losses']) + 1))
            color = colors[i % len(colors)]
            
            fig.add_trace(
                go.Scatter(x=epochs, y=result['losses'], 
                          mode='lines+markers', name=f'{class_name} Loss',
                          line=dict(color=color)),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=epochs, y=result['accuracies'], 
                          mode='lines+markers', name=f'{class_name} Acc',
                          line=dict(color=color, dash='dash')),
                row=1, col=2
            )
            fig.update_xaxes(title_text="Epochs", row=1, col=1)
            fig.update_xaxes(title_text="Epochs", row=1, col=2)
            fig.update_yaxes(title_text="Loss", row=1, col=1)
            fig.update_yaxes(title_text="Accuracy (%)", row=1, col=2)
    
    fig.update_layout(height=400, title="Bootstrap Training Progress")
    return fig

def plot_discovery_batch_results(batch_results, target_classes):
    """Plot results of a discovery batch"""
    if not batch_results:
        return go.Figure(), go.Figure()
    
    # Sample scores plot
    fig_scores = go.Figure()
    
    scores = batch_results['all_scores']
    top_10_indices = batch_results['top_10_indices']
    
    # Color samples by type
    colors = []
    labels = []
    sizes = []
    
    for i, (is_target, class_name) in enumerate(zip(
        batch_results['batch_data']['is_target_class'],
        batch_results['batch_data']['class_names']
    )):
        if is_target:
            colors.append('red')
            labels.append(f'Target: {class_name}')
            sizes.append(12 if i in top_10_indices else 8)
        else:
            colors.append('lightblue')
            labels.append(f'Other: {class_name}')
            sizes.append(10 if i in top_10_indices else 6)
    
    # Plot all samples
    fig_scores.add_trace(go.Scatter(
        x=list(range(len(scores))),
        y=scores,
        mode='markers',
        marker=dict(color=colors, size=sizes),
        text=labels,
        name='Samples',
        hovertemplate='%{text}<br>Score: %{y:.3f}<extra></extra>'
    ))
    
    # Highlight top 10
    top_10_scores = [scores[i] for i in top_10_indices]
    fig_scores.add_trace(go.Scatter(
        x=top_10_indices,
        y=top_10_scores,
        mode='markers',
        marker=dict(color='gold', size=15, symbol='star'),
        name='Top 10',
        hovertemplate='Top 10<br>Score: %{y:.3f}<extra></extra>'
    ))
    
    fig_scores.update_layout(
        title=f'Discovery Batch Results (Batch {batch_results["batch_id"]})',
        xaxis_title='Sample Index',
        yaxis_title='Selection Score',
        height=400
    )
    
    # Detection results plot
    detection_data = batch_results['all_detection_results']
    
    # Create detection summary
    detection_summary = {
        'Target Classes Detected': 0,
        'Target Classes Missed': 0,
        'Non-Target Correctly Rejected': 0,
        'Non-Target Incorrectly Flagged': 0
    }
    
    for detection in detection_data:
        if detection['is_target_class']:
            if detection['detected_as_novel']:
                detection_summary['Target Classes Detected'] += 1
            else:
                detection_summary['Target Classes Missed'] += 1
        else:
            if detection['detected_as_novel']:
                detection_summary['Non-Target Incorrectly Flagged'] += 1
            else:
                detection_summary['Non-Target Correctly Rejected'] += 1
    
    fig_detection = go.Figure(data=[
        go.Bar(
            x=list(detection_summary.keys()),
            y=list(detection_summary.values()),
            marker_color=['green', 'red', 'blue', 'orange']
        )
    ])
    
    fig_detection.update_layout(
        title='Detection Performance Summary',
        yaxis_title='Count',
        height=400
    )
    
    return fig_scores, fig_detection

def plot_found_samples_summary(discovery_engine):
    """Plot summary of all found target samples"""
    summary = discovery_engine.get_found_samples_summary()
    
    if not summary:
        return go.Figure()
    
    classes = list(summary.keys())
    total_found = [summary[cls]['total_found'] for cls in classes]
    correctly_detected = [summary[cls]['correctly_detected'] for cls in classes]
    in_top_10 = [summary[cls]['in_top_10'] for cls in classes]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Total Found',
        x=classes,
        y=total_found,
        marker_color='lightblue'
    ))
    
    fig.add_trace(go.Bar(
        name='Correctly Detected',
        x=classes,
        y=correctly_detected,
        marker_color='green'
    ))
    
    fig.add_trace(go.Bar(
        name='In Top 10',
        x=classes,
        y=in_top_10,
        marker_color='gold'
    ))
    
    fig.update_layout(
        title='Found Target Samples Summary',
        xaxis_title='Target Classes',
        yaxis_title='Count',
        barmode='group',
        height=400
    )
    
    return fig

def plot_sample_selection_interface(discovery_engine, target_class):
    """Create interface for manual sample selection"""
    if target_class not in discovery_engine.found_samples:
        return None, pd.DataFrame()
    
    samples = discovery_engine.found_samples[target_class]
    
    if not samples:
        return None, pd.DataFrame()
    
    # Create DataFrame for display
    sample_data = []
    for i, sample in enumerate(samples):
        sample_data.append({
            'Sample ID': i,
            'Batch ID': sample['batch_id'],
            'Score': f"{sample['score']:.3f}",
            'In Top 10': sample['rank_in_top_10'],
            'Top 10 Rank': sample['rank_position'] if sample['rank_position'] else 'N/A',
            'Detected as Novel': sample['detection_result']['detected_as_novel'],
            'Confidence': f"{sample['detection_result']['max_confidence']:.3f}",
            'Correct Detection': sample['detection_result']['correct_detection']
        })
    
    df = pd.DataFrame(sample_data)
    
    return samples, df


def plot_pca_variance_explained(detector):
    """Plot PCA explained variance"""
    if not hasattr(detector, 'pca') or detector.pca is None:
        return go.Figure().add_annotation(text="No PCA data available", 
                                         xref="paper", yref="paper", 
                                         x=0.5, y=0.5, showarrow=False)
    
    # Get all components, not just first 2
    explained_var = detector.pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)
    
    # Limit to first 10 components for readability
    n_components = min(10, len(explained_var))
    explained_var = explained_var[:n_components]
    cumulative_var = cumulative_var[:n_components]
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Individual variance (bar chart)
    fig.add_trace(go.Bar(
        x=list(range(1, n_components + 1)),
        y=explained_var * 100,
        name='Individual Variance',
        marker_color='lightblue',
        opacity=0.7
    ), secondary_y=False)
    
    # Cumulative variance (line chart)
    fig.add_trace(go.Scatter(
        x=list(range(1, n_components + 1)),
        y=cumulative_var * 100,
        mode='lines+markers',
        name='Cumulative Variance',
        line=dict(color='red', width=3),
        marker=dict(size=8)
    ), secondary_y=True)
    
    # Update axes
    fig.update_xaxes(title_text='Principal Component')
    fig.update_yaxes(title_text='Individual Explained Variance (%)', secondary_y=False)
    fig.update_yaxes(title_text='Cumulative Explained Variance (%)', secondary_y=True)
    
    fig.update_layout(
        title='PCA Explained Variance Analysis',
        height=400,
        hovermode='x unified'
    )
    
    return fig

def plot_class_separation_metrics(detector, reduction_results):
    """Plot metrics showing class separation quality"""
    metrics_data = []
    
    for method, data in reduction_results.items():
        features = data['features']
        labels = data['labels']
        
        # Calculate silhouette score (measure of cluster separation)
        try:
            from sklearn.metrics import silhouette_score
            if len(np.unique(labels)) > 1:  # Need at least 2 classes
                silhouette = silhouette_score(features, labels)
                metrics_data.append({
                    'Method': method.upper(),
                    'Silhouette Score': silhouette,
                    'Status': 'Success'
                })
            else:
                metrics_data.append({
                    'Method': method.upper(),
                    'Silhouette Score': 0,
                    'Status': 'Only one class'
                })
        except Exception as e:
            metrics_data.append({
                'Method': method.upper(),
                'Silhouette Score': 0,
                'Status': f'Error: {str(e)[:30]}...'
            })
    
    if not metrics_data:
        return go.Figure().add_annotation(text="No data available", 
                                         xref="paper", yref="paper", 
                                         x=0.5, y=0.5, showarrow=False)
    
    df_metrics = pd.DataFrame(metrics_data)
    
    # Create bar chart with color coding
    colors = ['lightblue' if status == 'Success' else 'lightcoral' 
              for status in df_metrics['Status']]
    
    fig = go.Figure(data=[
        go.Bar(
            x=df_metrics['Method'],
            y=df_metrics['Silhouette Score'],
            marker_color=colors,
            text=[f"Score: {score:.3f}<br>Status: {status}" 
                  for score, status in zip(df_metrics['Silhouette Score'], df_metrics['Status'])],
            textposition="outside",
            hovertemplate='%{x}<br>Silhouette Score: %{y:.3f}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title='Class Separation Quality (Silhouette Score)',
        yaxis_title='Silhouette Score (-1 to 1, higher is better)',
        height=300,
        showlegend=False
    )
    
    # Add reference lines
    fig.add_hline(y=0, line_dash="dash", line_color="gray", 
                  annotation_text="Random clustering")
    
    return fig

###############################################################################################
                                        ## Main ##
###############################################################################################


def main():
    st.title("OOD Analysis")
    #st.markdown("Advanced new class learning with bootstrap training and intelligent sample discovery")
    st.markdown("---")
    
    # Initialize session state
    if 'detector' not in st.session_state:
        st.session_state.detector = EnhancedOODDetector()
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    if 'selected_classes' not in st.session_state:
        st.session_state.selected_classes = []
    if 'bootstrap_results' not in st.session_state:
        st.session_state.bootstrap_results = {}
    if 'discovery_engine' not in st.session_state:
        st.session_state.discovery_engine = None
    if 'discovery_results' not in st.session_state:
        st.session_state.discovery_results = []
    if 'selected_samples_for_retraining' not in st.session_state:
        st.session_state.selected_samples_for_retraining = {}
    
    # Sidebar - Mission Control Panel
    with st.sidebar:
        st.header("Control Panel")
        
        tab1, tab2, tab3, tab4 = st.tabs(["Backbone Training", "Class Selection", "Weights Selection", "Mission"])
        
        with tab1:
            st.subheader("Base Model Training")
            samples_per_class = st.slider("Samples per class", 100, 5000, 5000, 100,
                                        help="Train on this many samples per class, leaving the rest unseen")
            # Remove the num_classes slider since we're training on all 10 classes
            epochs = st.slider("Epochs", 1, 100, 30, 1)
            batch_size = st.selectbox("Batch size", [32, 64, 128], index=1)
            learning_rate = st.selectbox("Learning rate", [0.0001, 0.001, 0.01], index=1)
            loss_type = st.selectbox("Loss type", ["CrossEntropy", "Contrastive"], index=0,
                            help="Choose between standard cross-entropy loss and contrastive loss")


            if st.button("Train Base Model", type="primary"):
                with st.spinner("Training base model..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    try:
                        history = st.session_state.detector.train_model(
                            samples_per_class=samples_per_class,
                            num_classes=10,  # Always train on all 10 CIFAR-10 classes
                            epochs=epochs,
                            batch_size=batch_size,
                            lr=learning_rate,
                            loss_type=loss_type,
                            progress_bar=progress_bar,
                            status_text=status_text
                        )
                        
                        st.session_state.model_trained = True
                        st.session_state.training_history = history
                        st.success(f"Base model trained! Test accuracy: {history['test_accuracy']:.2f}%")
                        
                        # Show info about unseen samples
                        unseen_samples = max(0, 5000 - samples_per_class)  # CIFAR-10 has ~5000 samples per class
                        st.info(f"Training used {samples_per_class} samples per class. "
                            f"~{min(100, unseen_samples)} unseen samples per class available for discovery.")
                        
                    except Exception as e:
                        st.error(f"Training failed: {str(e)}")
        
        with tab2:
            st.subheader("CIFAR-100 Class Selection")
            st.markdown("Select up to 5 classes to work with:")
            
            # Create searchable multiselect
            available_classes = [(i, name) for i, name in enumerate(CIFAR100_CLASSES)]
            
            # Search functionality
            search_term = st.text_input("Search classes:", placeholder="Type to search...")
            
            if search_term:
                filtered_classes = [(i, name) for i, name in available_classes 
                                  if search_term.lower() in name.lower()]
            else:
                filtered_classes = available_classes
            
            # Display classes in a nice format
            if len(st.session_state.selected_classes) < 5:
                st.markdown("**Available Classes:**")
                cols = st.columns(3)
                for i, (class_idx, class_name) in enumerate(filtered_classes):  
                    if class_idx not in [cls['class_idx'] for cls in st.session_state.selected_classes]:
                        with cols[i % 3]:
                            if st.button(f"{class_name}", key=f"add_{class_idx}"):
                                if len(st.session_state.selected_classes) < 5:
                                    st.session_state.selected_classes.append({
                                        'class_idx': class_idx,
                                        'name': class_name
                                    })
                                    st.rerun()
            
            # Show selected classes
            if st.session_state.selected_classes:
                st.markdown("**Selected Classes:**")
                for i, cls_info in enumerate(st.session_state.selected_classes):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"{i+1}. {cls_info['name']} (ID: {cls_info['class_idx']})")
                    with col2:
                        if st.button("Remove", key=f"remove_{cls_info['class_idx']}"):
                            st.session_state.selected_classes.pop(i)
                            st.rerun()
                
        with tab3:
            st.subheader("Selection Criteria Weights")
            st.markdown("Adjust how much each criterion influences sample selection:")
            
            # Weight sliders with mathematical explanations
            with st.expander("Diversity Weight", expanded=False):
                st.latex(r"\text{diversity}(x_i) = \frac{1}{N} \sum_{j=1}^{N} \|x_i - x_j\|_2")
                st.markdown("""
                **Explanation:** This measures how different the current sample's feature vector $x_i$ is from all previously discovered samples. 
                We compute the Euclidean distance between the current sample and each of the $N$ previously found samples, then take the average. 
                Higher values indicate the sample is more unique compared to what we've already collected.
                """)
            # Get default from current_weights if available
            default_diversity = st.session_state.current_weights.get('diversity', 0.2) if 'current_weights' in st.session_state else 0.2
            diversity_weight = st.slider("Diversity", 0.0, 1.0, default_diversity, 0.05)
            
            with st.expander("Euclidean Distance Weight", expanded=False):
                st.latex(r"\text{euclidean}(x) = \min_{c=1}^{C} \|x - \mu_c\|_2")
                st.markdown("""
                **Explanation:** This finds the minimum Euclidean distance from the sample's feature vector $x$ to any of the $C$ known class centroids $\mu_c$. 
                The class centroid $\mu_c$ is the average feature vector of all training samples from class $c$. 
                A larger minimum distance suggests the sample doesn't fit well into any existing class.
                """)
            default_distance = st.session_state.current_weights.get('distance_from_existing', 0.25) if 'current_weights' in st.session_state else 0.25
            distance_weight = st.slider("Euclidean Distance", 0.0, 1.0, default_distance, 0.05)
            
            with st.expander("Mahalanobis Distance Weight", expanded=False):
                st.latex(r"\text{mahalanobis}(x) = \min_{c=1}^{C} \sqrt{(x - \mu_c)^T \Sigma_c^{-1} (x - \mu_c)}")
                st.markdown("Where the covariance matrix is:")
                st.latex(r"\Sigma_c = \frac{1}{n_c-1} \sum_{i=1}^{n_c} (x_i^{(c)} - \mu_c)(x_i^{(c)} - \mu_c)^T")
                st.markdown("""
                **Explanation:** This is a statistically sophisticated distance measure that accounts for the shape and orientation of each class distribution. 
                Unlike Euclidean distance, it considers how much each class naturally varies in different feature dimensions. 
                $\Sigma_c$ is the covariance matrix of class $c$, and $\Sigma_c^{-1}$ is its inverse. 
                This metric is more accurate because it normalizes by the class's natural variability.
                """)
            default_mahalanobis = st.session_state.current_weights.get('mahalanobis_distance', 0.25) if 'current_weights' in st.session_state else 0.25
            mahalanobis_weight = st.slider("Mahalanobis Distance", 0.0, 1.0, default_mahalanobis, 0.05)
            
            with st.expander("Consistency Weight", expanded=False):
                st.latex(r"\text{consistency}(x) = \frac{1}{1 + \text{Var}(x)}")
                st.markdown("Where variance is:")
                st.latex(r"\text{Var}(x) = \frac{1}{d} \sum_{i=1}^{d} (x_i - \bar{x})^2")
                st.markdown("And mean is:")
                st.latex(r"\bar{x} = \frac{1}{d} \sum_{i=1}^{d} x_i")
                st.markdown("""
                **Explanation:** This measures how "stable" or "consistent" the feature representation is. 
                We compute the variance across all $d$ dimensions of the feature vector $x$. 
                Lower variance means the features are more concentrated and reliable. 
                The score is designed so that lower variance (more consistent features) gives a higher score.
                """)
            default_consistency = st.session_state.current_weights.get('consistency', 0.15) if 'current_weights' in st.session_state else 0.15
            consistency_weight = st.slider("Consistency", 0.0, 1.0, default_consistency, 0.05)
            
            with st.expander("Confidence Weight", expanded=False):
                st.latex(r"\text{confidence}(x) = 1 - \max_{i=1}^{K} P(y_i|x)")
                st.markdown("Where softmax probability is:")
                st.latex(r"P(y_i|x) = \frac{\exp(z_i/T)}{\sum_{j=1}^{K} \exp(z_j/T)}")
                st.markdown("""
                **Explanation:** This measures how uncertain the model is about the sample. 
                We pass the sample through the trained model to get logits $z_i$ for each of the $K$ known classes, 
                apply temperature scaling with parameter $T$, then compute softmax probabilities. 
                We take the maximum probability and subtract it from 1, so samples that confuse the model (low confidence) get higher scores.
                """)
            default_confidence = st.session_state.current_weights.get('confidence', 0.15) if 'current_weights' in st.session_state else 0.15
            confidence_weight = st.slider("Confidence", 0.0, 1.0, default_confidence, 0.05)
            
            # Normalize weights
            total_weight = (diversity_weight + distance_weight + mahalanobis_weight + 
                        consistency_weight + confidence_weight)
            if total_weight > 0:
                weights = {
                    'diversity': diversity_weight / total_weight,
                    'distance_from_existing': distance_weight / total_weight,
                    'mahalanobis_distance': mahalanobis_weight / total_weight,
                    'consistency': consistency_weight / total_weight,
                    'confidence': confidence_weight / total_weight
                }
            else:
                weights = {'diversity': 0.2, 'distance_from_existing': 0.25, 
                        'mahalanobis_distance': 0.25, 'consistency': 0.15, 'confidence': 0.15}

            st.session_state.current_weights = weights
            
            # Show normalized weights
            st.markdown("**Normalized Weights:**")
            for name, weight in weights.items():
                st.text(f"{name.replace('_', ' ').title()}: {weight:.3f}")
                
            # Weight Experiments Section
            st.markdown("---")
            st.subheader("Weight Experiments")
            st.markdown("Run experiments with different weight configurations to compare results.")
            
            # Experiment presets
            experiment_presets = {
                "Balanced": {'diversity': 0.2, 'distance_from_existing': 0.2, 'mahalanobis_distance': 0.2, 'consistency': 0.2, 'confidence': 0.2},
                "Distance-Focused": {'diversity': 0.1, 'distance_from_existing': 0.4, 'mahalanobis_distance': 0.4, 'consistency': 0.05, 'confidence': 0.05},
                "Diversity-Focused": {'diversity': 0.6, 'distance_from_existing': 0.1, 'mahalanobis_distance': 0.1, 'consistency': 0.1, 'confidence': 0.1},
                "Confidence-Focused": {'diversity': 0.05, 'distance_from_existing': 0.05, 'mahalanobis_distance': 0.05, 'confidence': 0.7, 'consistency': 0.15},
                "Statistical-Focused": {'diversity': 0.1, 'distance_from_existing': 0.1, 'mahalanobis_distance': 0.5, 'consistency': 0.3, 'confidence': 0.0}
            }
            
            # Initialize experiment tracking
            if 'weight_experiments' not in st.session_state:
                st.session_state.weight_experiments = {}
                
            col1, col2 = st.columns(2)
            with col1:
                experiment_name = st.text_input("Experiment Name:", placeholder="My_Experiment_1")
                preset_choice = st.selectbox("Quick Preset:", ["Custom"] + list(experiment_presets.keys()))
                
            with col2:
                if st.button("Save Current Weights as Experiment"):
                    if experiment_name:
                        st.session_state.weight_experiments[experiment_name] = weights.copy()
                        st.success(f"Saved experiment: {experiment_name}")
                    else:
                        st.warning("Please enter an experiment name")
                        
                if st.button("Load Preset Weights") and preset_choice != "Custom":
                    # Update the sliders by updating session state values
                    preset_weights = experiment_presets[preset_choice]
                    st.session_state.current_weights = preset_weights
                    st.info(f"Loaded preset: {preset_choice}")
                    st.rerun()
            
            # Show saved experiments
            if st.session_state.weight_experiments:
                st.markdown("**Saved Experiments:**")
                for exp_name, exp_weights in st.session_state.weight_experiments.items():
                    with st.expander(f"Experiment: {exp_name}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            for name, weight in exp_weights.items():
                                st.text(f"{name.replace('_', ' ').title()}: {weight:.3f}")
                        with col2:
                            if st.button(f"Load {exp_name}", key=f"load_{exp_name}"):
                                st.session_state.current_weights = exp_weights.copy()
                                st.success(f"Loaded experiment: {exp_name}")
                                st.rerun()
                            if st.button(f"Delete {exp_name}", key=f"delete_{exp_name}"):
                                del st.session_state.weight_experiments[exp_name]
                                st.success(f"Deleted experiment: {exp_name}")
                                st.rerun()
                                
            # Experiment Comparison
            if len(st.session_state.weight_experiments) > 1:
                st.markdown("**Compare Experiments:**")
                comparison_data = []
                for exp_name, exp_weights in st.session_state.weight_experiments.items():
                    row = {"Experiment": exp_name}
                    for weight_name, weight_value in exp_weights.items():
                        row[weight_name.replace('_', ' ').title()] = f"{weight_value:.3f}"
                    comparison_data.append(row)
                
                if comparison_data:
                    comparison_df = pd.DataFrame(comparison_data)
                    st.dataframe(comparison_df, use_container_width=True)
                    
                    # Add notes section
                    st.markdown("**Experiment Notes:**")
                    notes_key = f"experiment_notes_{len(st.session_state.weight_experiments)}"
                    experiment_notes = st.text_area("Add notes about these experiments:", key=notes_key, height=100, 
                                                   placeholder="e.g., Distance-Focused found 15% more target samples than Balanced...")
                    
                    if experiment_notes:
                        st.session_state[f"saved_notes_{len(st.session_state.weight_experiments)}"] = experiment_notes
        
        with tab4:
            st.subheader("New Mission Configuration")
            
            # Discovery configuration
            st.markdown("**Sample Discovery:**")
            st.markdown("""
            **Sampling Strategy:**
            - 100 unseen samples from each CIFAR-10 class (samples not used in training)
            - 50 samples from each selected CIFAR-100 class
            """)

            # Calculate expected batch size
            if st.session_state.model_trained and st.session_state.selected_classes:
                cifar10_unseen_count = 10 * 100  # 100 unseen samples from each of 10 classes
                selected_cifar100_count = len(st.session_state.selected_classes) * 50
                expected_batch_size = cifar10_unseen_count + selected_cifar100_count
                st.info(f"Expected batch size: {expected_batch_size} samples "
                    f"({cifar10_unseen_count} CIFAR-10 + {selected_cifar100_count} CIFAR-100)")

            # Store in session state for access in main area
            st.session_state.batch_size_discovery = st.slider("Random batch size", 50, 200, 100, 10,
                                help="How many random samples to evaluate at once")
            st.session_state.novelty_threshold = st.slider("Novelty threshold", 0.3, 0.9, 0.7, 0.05,
                                help="Confidence threshold for novelty detection")

            # Initialize discovery engine button
            if st.button("Initialize OOD Discovery Engine",
                        disabled=(not st.session_state.model_trained or not st.session_state.selected_classes)):
                
                # No bootstrap training needed - just set up for pure discovery
                target_classes = {cls['name']: cls['class_idx'] for cls in st.session_state.selected_classes}
                
                st.session_state.discovery_engine = SampleDiscoveryEngine(
                    st.session_state.detector,
                    target_classes,
                    st.session_state.current_weights
                )
                
                st.session_state.discovery_results = []
                st.session_state.selected_samples_for_retraining = {}
                
                st.success("OOD Discovery engine initialized!")
                st.info(f"Ready to discover samples from {len(target_classes)} target classes: {', '.join(target_classes.keys())}")

            # Show current status
            if st.session_state.discovery_engine is not None:
                st.success("Discovery engine is ready!")
                target_classes = list(st.session_state.discovery_engine.target_classes.keys())
                st.info(f"Target classes: {', '.join(target_classes)}")
            else:
                if not st.session_state.model_trained:
                    st.warning("Train base model first")
                elif not st.session_state.selected_classes:
                    st.warning("Select CIFAR-100 classes first")
                else:
                    st.info("Click 'Initialize OOD Discovery Engine' to start")

            st.markdown("---")

            # Run discovery batch button
            if st.button("Run Discovery Batch",
                        disabled=(st.session_state.discovery_engine is None)):
                with st.spinner("Running sample discovery..."):
                    # Sample random batch - use session state variable
                    batch_data = st.session_state.discovery_engine.sample_random_batch(
                        st.session_state.get('batch_size_discovery', 100)
                    )
                    
                    # Evaluate batch
                    batch_results = st.session_state.discovery_engine.evaluate_batch(batch_data)
                    
                    # Store results
                    st.session_state.discovery_results.append(batch_results)
                    st.session_state.latest_batch_results = batch_results
                    
                    st.success(f"Discovery batch completed! Found {batch_results['target_samples_in_batch']} target samples")
            
            st.markdown("---")
            
            # Retraining configuration (define the missing variables)
            st.markdown("**Retraining Configuration:**")
            
            col1, col2 = st.columns(2)
            with col1:
                retrain_epochs = st.slider("Retraining epochs", 10, 100, 30, 5)
            with col2:
                retrain_lr = st.selectbox("Retraining learning rate", [0.0001, 0.001, 0.01], index=1)
            
            retrain_loss_type = st.selectbox("Retraining loss type", ["CrossEntropy", "Contrastive"], 
                                index=0, help="Loss function for new class training")


            if st.button("Train New Classes from Selected Samples", type="primary"):
                with st.spinner("Training new classes from selected OOD samples..."):
                    # Use the new multi-class training approach
                    retrain_results = st.session_state.detector.new_class_learner.train_new_classes(
                        st.session_state.selected_samples_for_retraining,
                        epochs=retrain_epochs,
                        lr=retrain_lr
                    )
                    
                    st.session_state.retrain_results = retrain_results
                    st.session_state.selected_samples_for_retraining = {}
                    
                    # Display results
                    if retrain_results:
                        st.success(f"Trained {len(retrain_results)} new classes!")
                        
                        # Show model expansion info
                        total_classes = st.session_state.detector.model.num_classes
                        st.info(f"Model now knows {total_classes} classes total:")
                        
                        # Show class breakdown
                        cifar10_classes = 10
                        new_classes = total_classes - cifar10_classes
                        st.write(f"- Original CIFAR-10 classes: {cifar10_classes}")
                        st.write(f"- New CIFAR-100 classes: {new_classes}")
                        
                        # List all active classes
                        active_classes = []
                        for i in range(total_classes):
                            class_name = st.session_state.detector.model.get_class_name(i)
                            active_classes.append(f"{i}: {class_name}")
                        
                        st.write("**Active Classes:**")
                        for class_info in active_classes:
                            st.write(f"  {class_info}")
                    else:
                        st.warning("No classes were trained. Please select samples first.")
                    
                    st.rerun()
    # Main content area
    if not st.session_state.model_trained:
        st.info("Configure and train the base model first")
        
        # Show helpful information
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            ### Workflow Overview
            This system simulates a simple live learning mission:
            
            1. **Train the backbone model** on CIFAR-10 
            2. **Select CIFAR-100 classes** up to 5, for the mission itself
            3. **Bootstrap train** on X samples from each selected class
            4. **Sample discovery**: from 100 random samples from all CIFAR100
            5. **Evaluation**: Return top K with detection results
            6. **Manual selection**: Choose samples for retraining
            """)
        
        # with col2:
        #     st.markdown("""
        #     ### Key Features
        #     - **Bootstrap training**: Initial training on target classes
        #     - **Random sampling**: From all 100 CIFAR-100 classes
        #     - **Detection tracking**: Whether model found target samples
        #     - **Sample scoring**: See why each sample was selected
        #     - **Manual curation**: Choose which samples to retrain on
        #     - **Performance analytics**: Track discovery effectiveness
        #     """)
    
    else:
        tabs = st.tabs(["Base Model", "Bootstrap Training", "Sample Discovery", "Found Samples", "Retrain Selection", "Feature Space"])        
        with tabs[0]:  # Modify the existing Base Model tab
            st.header("Model Status & Performance")
            
            # Model architecture info
            if st.session_state.model_trained:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    total_classes = st.session_state.detector.model.num_classes
                    st.metric("Total Classes", total_classes)
                
                with col2:
                    cifar10_classes = 10
                    new_classes = total_classes - cifar10_classes
                    st.metric("CIFAR-10 Classes", cifar10_classes)
                
                with col3:
                    st.metric("New Classes", new_classes)
                
                # Show all active classes
                if total_classes > 10:
                    st.subheader("All Active Classes")
                    
                    # Create a nice display of all classes
                    class_data = []
                    for i in range(total_classes):
                        class_name = st.session_state.detector.model.get_class_name(i)
                        class_type = "CIFAR-10" if i < 10 else "CIFAR-100"
                        class_data.append({
                            'Index': i,
                            'Class Name': class_name,
                            'Type': class_type
                        })
                    
                    df_classes = pd.DataFrame(class_data)
                    st.dataframe(df_classes, use_container_width=True)
                
                # Test model on sample images
                if st.button("🧪 Test Model on Random Samples"):
                    st.subheader("Model Predictions on Random Samples")
                    
                    # Get some random samples from different sources
                    transform = st.session_state.detector.get_augmentation_transform(training=False)
                    
                    # CIFAR-10 test samples
                    cifar10_test = torchvision.datasets.CIFAR10(
                        root='./data', train=False, download=True, transform=transform
                    )
                    
                    # CIFAR-100 test samples
                    cifar100_test = torchvision.datasets.CIFAR100(
                        root='./data', train=False, download=True, transform=transform
                    )
                    
                    # Sample 5 images from each
                    test_images = []
                    true_labels = []
                    sources = []
                    
                    # 5 random CIFAR-10 images
                    for _ in range(5):
                        idx = random.randint(0, len(cifar10_test) - 1)
                        img, label = cifar10_test[idx]
                        test_images.append(img)
                        true_labels.append(st.session_state.detector.cifar10_classes[label])
                        sources.append("CIFAR-10")
                    
                    # 5 random CIFAR-100 images
                    for _ in range(5):
                        idx = random.randint(0, len(cifar100_test) - 1)
                        img, label = cifar100_test[idx]
                        test_images.append(img)
                        true_labels.append(CIFAR100_CLASSES[label])
                        sources.append("CIFAR-100")
                    
                    # Get predictions
                    predictions = st.session_state.detector.predict_with_class_names(test_images)
                    
                    # Display results
                    cols = st.columns(5)
                    for i in range(5):
                        with cols[i]:
                            # Denormalize for display
                            img = test_images[i]
                            img_display = img * torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1) + torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
                            img_display = torch.clamp(img_display, 0, 1)
                            img_np = img_display.permute(1, 2, 0).cpu().numpy()
                            
                            st.image(img_np, width=100)
                            st.write(f"**True:** {true_labels[i]}")
                            st.write(f"**Predicted:** {predictions[i]['class_name']}")
                            st.write(f"**Confidence:** {predictions[i]['confidence']:.3f}")
                            st.write(f"**Source:** {sources[i]}")
                            
                            # Check if prediction is correct
                            if predictions[i]['class_name'] == true_labels[i]:
                                st.success("Correct!")
                            else:
                                st.error("Wrong")
                    
                    # Show second row if we have more samples
                    if len(test_images) > 5:
                        cols = st.columns(5)
                        for i in range(5, min(10, len(test_images))):
                            with cols[i-5]:
                                # Denormalize for display
                                img = test_images[i]
                                img_display = img * torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1) + torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
                                img_display = torch.clamp(img_display, 0, 1)
                                img_np = img_display.permute(1, 2, 0).cpu().numpy()
                                
                                st.image(img_np, width=100)
                                st.write(f"**True:** {true_labels[i]}")
                                st.write(f"**Predicted:** {predictions[i]['class_name']}")
                                st.write(f"**Confidence:** {predictions[i]['confidence']:.3f}")
                                st.write(f"**Source:** {sources[i]}")
                                
                                # Check if prediction is correct
                                if predictions[i]['class_name'] == true_labels[i]:
                                    st.success("Correct!")
                                else:
                                    st.error("Wrong")
            
            # Original training history (if available)
            if 'training_history' in st.session_state:
                st.subheader("Original CIFAR-10 Training History")
                history = st.session_state.training_history
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Final Train Accuracy", f"{history['train_accuracies'][-1]:.2f}%")
                with col2:
                    st.metric("Best Validation Accuracy", f"{history['best_accuracy']:.2f}%")
                with col3:
                    st.metric("Test Accuracy", f"{history['test_accuracy']:.2f}%")
                
                # Plot training history
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=('Training Metrics', 'Validation Accuracy')
                )
                
                epochs_list = list(range(1, len(history['train_accuracies']) + 1))
                
                fig.add_trace(
                    go.Scatter(x=epochs_list, y=history['train_accuracies'], 
                            mode='lines+markers', name='Train Acc'),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(x=epochs_list, y=history['val_accuracies'], 
                            mode='lines+markers', name='Val Acc', line=dict(color='green')),
                    row=1, col=2
                )

                fig.update_xaxes(title_text="Epochs", row=1, col=1)
                fig.update_xaxes(title_text="Epochs", row=1, col=2)
                fig.update_yaxes(title_text="Accuracy (%)", row=1, col=1)
                fig.update_yaxes(title_text="Accuracy (%)", row=1, col=2)

                fig.update_layout(height=400, showlegend=True)
                st.plotly_chart(fig, use_container_width=True)
                
        with tabs[1]:
            st.header("Bootstrap Training Results")
            
            if st.session_state.bootstrap_results:
                # Summary metrics
                st.subheader("Bootstrap Training Summary")
                
                summary_data = []
                for class_name, result in st.session_state.bootstrap_results.items():
                    if result:
                        summary_data.append({
                            'Class': class_name,
                            'Final Accuracy': f"{result['final_accuracy']:.3f}",
                            'Final Loss': f"{result['final_loss']:.3f}",
                            'Samples Used': result['samples_used']
                        })
                
                if summary_data:
                    df_summary = pd.DataFrame(summary_data)
                    st.dataframe(df_summary, use_container_width=True)
                
                # Plot training curves
                st.subheader("Training Progress")
                fig_bootstrap = plot_bootstrap_results(st.session_state.bootstrap_results)
                st.plotly_chart(fig_bootstrap, use_container_width=True)
                
                # Show trained classes status
                st.subheader("Class Training Status")
                cols = st.columns(len(st.session_state.selected_classes))
                
                for i, cls_info in enumerate(st.session_state.selected_classes):
                    with cols[i]:
                        class_name = cls_info['name']
                        is_trained = class_name in st.session_state.detector.new_class_learner.bootstrap_trained_classes
                        
                        if is_trained:
                            result = st.session_state.bootstrap_results.get(class_name, {})
                            accuracy = result.get('final_accuracy', 0)
                            st.metric(f"{class_name}", f"{accuracy:.3f}", "Trained")
                        else:
                            st.metric(f"{class_name}", "Not trained", "Pending")
            
            else:
                st.info("👈 Run bootstrap training first in the sidebar")
                
                st.markdown("""
                ### Bootstrap Training Process:
                
                1. **Sample Collection**: Get X samples from each selected CIFAR-100 class
                2. **Head Addition**: Add new classification head for each class
                3. **Binary Training**: Train to distinguish new class from existing ones
                4. **Feature Learning**: Learn class-specific feature representations
                5. **Validation**: Track training progress and final performance
                """)
        
        with tabs[2]:
            st.header("Sample Discovery")
            
            if st.session_state.discovery_engine:
                # Discovery engine status
                st.subheader("Discovery Engine Status")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Target Classes", len(st.session_state.discovery_engine.target_classes))
                with col2:
                    st.metric("Discovery Batches", len(st.session_state.discovery_results))
                with col3:
                    total_found = sum(len(samples) for samples in st.session_state.discovery_engine.found_samples.values())
                    st.metric("Total Found Samples", total_found)
                
                # Latest batch results
                if 'latest_batch_results' in st.session_state:
                    batch_results = st.session_state.latest_batch_results
                    
                    st.subheader(f"Latest Batch Results (Batch {batch_results['batch_id']})")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Evaluated", batch_results['total_samples'])
                    with col2:
                        st.metric("Target Samples Found", batch_results['target_samples_in_batch'])
                    with col3:
                        st.metric("Top 10 Selected", 10)
                    with col4:
                        avg_score = np.mean(batch_results['all_scores'])
                        st.metric("Average Score", f"{avg_score:.3f}")
                    
                    # Plot batch results
                    fig_scores, fig_detection = plot_discovery_batch_results(
                        batch_results, st.session_state.discovery_engine.target_classes
                    )
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(fig_scores, use_container_width=True)
                    with col2:
                        st.plotly_chart(fig_detection, use_container_width=True)
                    
                    # Top 10 samples details
                    st.subheader("Top 10 Selected Samples")
                    
                    top_10_data = []
                    for rank, idx in enumerate(batch_results['top_10_indices']):
                        sample_data = batch_results['batch_data']
                        detection = batch_results['all_detection_results'][idx]
                        
                        top_10_data.append({
                            'Rank': rank + 1,
                            'True Class': sample_data['class_names'][idx],
                            'Is Target': sample_data['is_target_class'][idx],
                            'Score': f"{batch_results['all_scores'][idx]:.3f}",
                            'Detected as Novel': detection['detected_as_novel'],
                            'Confidence': f"{detection['max_confidence']:.3f}",
                            'Correct Detection': detection['correct_detection']
                        })
                    
                    df_top_10 = pd.DataFrame(top_10_data)
                    st.dataframe(df_top_10, use_container_width=True)
                    
                    # Show sample images
                    st.subheader("Top 10 Sample Images")
                    cols = st.columns(5)
                    for i, idx in enumerate(batch_results['top_10_indices'][:5]):
                        with cols[i]:
                            img = batch_results['batch_data']['images'][idx]
                            class_name = batch_results['batch_data']['class_names'][idx]
                            is_target = batch_results['batch_data']['is_target_class'][idx]
                            score = batch_results['all_scores'][idx]
                            
                            # Denormalize for display
                            img_display = img * torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1) + torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
                            img_display = torch.clamp(img_display, 0, 1)
                            img_np = img_display.permute(1, 2, 0).cpu().numpy()
                            
                            color = "🔴" if is_target else "🔵"
                            st.image(img_np, caption=f"{color} {class_name}\nScore: {score:.3f}", width=120)
                    
                    if len(batch_results['top_10_indices']) > 5:
                        cols = st.columns(5)
                        for i, idx in enumerate(batch_results['top_10_indices'][5:]):
                            with cols[i]:
                                img = batch_results['batch_data']['images'][idx]
                                class_name = batch_results['batch_data']['class_names'][idx]
                                is_target = batch_results['batch_data']['is_target_class'][idx]
                                score = batch_results['all_scores'][idx]
                                
                                # Denormalize for display
                                img_display = img * torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1) + torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
                                img_display = torch.clamp(img_display, 0, 1)
                                img_np = img_display.permute(1, 2, 0).cpu().numpy()
                                
                                color = "🔴" if is_target else "🔵"
                                st.image(img_np, caption=f"{color} {class_name}\nScore: {score:.3f}", width=120)
                
                # Discovery history
                if len(st.session_state.discovery_results) > 1:
                    st.subheader("Discovery History")
                    
                    history_data = []
                    for result in st.session_state.discovery_results:
                        history_data.append({
                            'Batch ID': result['batch_id'],
                            'Total Samples': result['total_samples'],
                            'Target Samples': result['target_samples_in_batch'],
                            'Average Score': f"{np.mean(result['all_scores']):.3f}",
                            'Timestamp': time.strftime('%H:%M:%S', time.localtime(result['timestamp']))
                        })
                    
                    df_history = pd.DataFrame(history_data)
                    st.dataframe(df_history, use_container_width=True)
            
            else:
                st.info("👈 Initialize discovery engine first in the sidebar")
                
                st.markdown("""
                ### Sample Discovery Process:
                
                1. **Random Sampling**: Get 100 random samples from all CIFAR-100 classes
                2. **Feature Extraction**: Extract features using the trained model
                3. **Scoring**: Apply selection criteria to rank samples
                4. **Top 10 Selection**: Select highest-scoring samples
                5. **Detection Analysis**: Check if model detected target samples as novel
                6. **Results Storage**: Store all results for manual curation
                """)
        
        with tabs[3]:
            st.header("Found Target Samples")
            
            if st.session_state.discovery_engine and st.session_state.discovery_engine.found_samples:
                # Overall summary
                st.subheader("Overall Discovery Summary")
                
                fig_summary = plot_found_samples_summary(st.session_state.discovery_engine)
                st.plotly_chart(fig_summary, use_container_width=True)
                
                # Detailed breakdown by class
                summary = st.session_state.discovery_engine.get_found_samples_summary()
                
                summary_data = []
                for class_name, stats in summary.items():
                    summary_data.append({
                        'Class': class_name,
                        'Total Found': stats['total_found'],
                        'Correctly Detected': stats['correctly_detected'],
                        'Detection Rate': f"{stats['detection_rate']:.2%}",
                        'In Top 10': stats['in_top_10'],
                        'Top 10 Rate': f"{stats['top_10_rate']:.2%}",
                        'Avg Score': f"{stats['avg_score']:.3f}"
                    })
                
                df_summary = pd.DataFrame(summary_data)
                st.dataframe(df_summary, use_container_width=True)
                
                # Per-class detailed view
                st.subheader("Per-Class Sample Details")
                
                # Select class to view
                available_classes = [cls for cls in st.session_state.discovery_engine.found_samples.keys() 
                                   if st.session_state.discovery_engine.found_samples[cls]]
                
                if available_classes:
                    selected_class = st.selectbox("Select class to view details:", available_classes)
                    
                    samples, df_samples = plot_sample_selection_interface(
                        st.session_state.discovery_engine, selected_class
                    )
                    
                    if not df_samples.empty:
                        st.dataframe(df_samples, use_container_width=True)
                        
                        # Show sample images for this class
                        st.subheader(f"Sample Images: {selected_class}")

                        # Show top 10 samples by score
                        class_samples = st.session_state.discovery_engine.found_samples[selected_class]
                        sorted_samples = sorted(class_samples, key=lambda x: x['score'], reverse=True)

                        # Display samples in rows of 5 without nested columns
                        for row_start in range(0, min(10, len(sorted_samples)), 5):
                            row_samples = sorted_samples[row_start:row_start+5]
                            cols = st.columns(len(row_samples))
                            
                            for col_idx, sample in enumerate(row_samples):
                                with cols[col_idx]:
                                    img = sample['image']
                                    score = sample['score']
                                    detected = sample['detection_result']['detected_as_novel']
                                    confidence = sample['detection_result']['max_confidence']
                                    
                                    # Denormalize for display
                                    img_display = img * torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1) + torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
                                    img_display = torch.clamp(img_display, 0, 1)
                                    img_np = img_display.permute(1, 2, 0).cpu().numpy()
                                    
                                    detection_icon = "[✓]" if detected else "[✗]"
                                    rank_icon = "*" if sample['rank_in_top_10'] else ""
                                    
                                    st.image(img_np, 
                                            caption=f"{rank_icon} {detection_icon}\nScore: {score:.3f}\nConf: {confidence:.3f}", 
                                            width=100)
                else:
                    st.info("No target samples found yet. Run more discovery batches!")
            
            else:
                st.info("No samples found yet. Run discovery batches first!")
        
        with tabs[4]:
            st.header("Multi-Class Training from Selected Samples")
            
            if st.session_state.discovery_engine and st.session_state.discovery_engine.found_samples:
                st.subheader("Select Samples for Multi-Class Training")
                
                # Show current model status
                if st.session_state.model_trained:
                    current_classes = st.session_state.detector.model.num_classes
                    max_classes = st.session_state.detector.model.max_classes
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Current Classes", current_classes)
                    with col2:
                        st.metric("Max Classes", max_classes)
                    with col3:
                        available_slots = max_classes - current_classes
                        st.metric("Available Slots", available_slots)
                    
                    if available_slots <= 0:
                        st.error("Maximum number of classes reached!")
                        st.stop()
                
                # Initialize selection state
                if 'selected_samples_for_retraining' not in st.session_state:
                    st.session_state.selected_samples_for_retraining = {}
                
                # Create selection interface for each class

                for class_name in st.session_state.discovery_engine.found_samples.keys():
                    if st.session_state.discovery_engine.found_samples[class_name]:
                        with st.expander(f"{class_name} Samples", expanded=True):
                            samples = st.session_state.discovery_engine.found_samples[class_name]
                            
                            # Initialize selection for this class
                            if class_name not in st.session_state.selected_samples_for_retraining:
                                st.session_state.selected_samples_for_retraining[class_name] = []
                            
                            # Quick selection buttons
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                if st.button(f"Select All {class_name}", key=f"select_all_{class_name}"):
                                    st.session_state.selected_samples_for_retraining[class_name] = samples.copy()
                                    st.rerun()
                            with col2:
                                if st.button(f"Select Top 5 {class_name}", key=f"select_top5_{class_name}"):
                                    sorted_samples = sorted(samples, key=lambda x: x['score'], reverse=True)
                                    st.session_state.selected_samples_for_retraining[class_name] = sorted_samples[:5]
                                    st.rerun()
                            with col3:
                                if st.button(f"Clear All {class_name}", key=f"clear_all_{class_name}"):
                                    st.session_state.selected_samples_for_retraining[class_name] = []
                                    st.rerun()
                            
                            # Show current selection count
                            selected_count = len(st.session_state.selected_samples_for_retraining[class_name])
                            st.info(f"Currently selected: {selected_count}/{len(samples)} samples")
                            
                            # Display samples in a grid with selection checkboxes
                            for row_start in range(0, len(samples), 5):
                                row_samples = samples[row_start:row_start+5]
                                cols = st.columns(len(row_samples))
                                
                                for col_idx, sample in enumerate(row_samples):
                                    with cols[col_idx]:
                                        sample_idx = row_start + col_idx
                                        sample_unique_id = f"{sample['batch_id']}_{sample.get('original_index', sample_idx)}"
                                        
                                        # Check if this sample is selected
                                        is_selected = any(
                                            s.get('batch_id') == sample['batch_id'] and 
                                            s.get('original_index', -1) == sample.get('original_index', sample_idx)
                                            for s in st.session_state.selected_samples_for_retraining[class_name]
                                        )
                                        
                                        # Image display
                                        img = sample['image']
                                        img_display = img * torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1) + torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
                                        img_display = torch.clamp(img_display, 0, 1)
                                        img_np = img_display.permute(1, 2, 0).cpu().numpy()
                                        
                                        # Show selection status visually
                                        if is_selected:
                                            st.image(img_np, width=100, caption="SELECTED")
                                        else:
                                            st.image(img_np, width=100)
                                        
                                        # Toggle button
                                        if is_selected:
                                            if st.button(f"Remove", key=f"remove_{class_name}_{sample_unique_id}"):
                                                st.session_state.selected_samples_for_retraining[class_name] = [
                                                    s for s in st.session_state.selected_samples_for_retraining[class_name]
                                                    if not (s.get('batch_id') == sample['batch_id'] and 
                                                        s.get('original_index', -1) == sample.get('original_index', sample_idx))
                                                ]
                                                st.rerun()
                                        else:
                                            if st.button(f"Select", key=f"add_{class_name}_{sample_unique_id}"):
                                                st.session_state.selected_samples_for_retraining[class_name].append(sample)
                                                st.rerun()
                                        
                                        # Sample info
                                        score = sample['score']
                                        detected = sample['detection_result']['detected_as_novel']
                                        confidence = sample['detection_result']['max_confidence']
                                        
                                        detection_icon = "[✓]" if detected else "[✗]"
                                        rank_icon = "*" if sample['rank_in_top_10'] else ""
                                        
                                        st.write(f"{rank_icon} {detection_icon}")
                                        st.write(f"Score: {score:.3f}")
                                        st.write(f"Conf: {confidence:.3f}")
                            
                            # Show selection summary for this class
                            selected_count = len(st.session_state.selected_samples_for_retraining[class_name])
                            st.write(f"**Selected: {selected_count} samples**")
                
                # Multi-class training configuration
                st.subheader("Multi-Class Training Configuration")

                total_selected = sum(len(samples) for samples in st.session_state.selected_samples_for_retraining.values())

                if total_selected > 0:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Selected", total_selected)
                    with col2:
                        classes_to_train = len([k for k, v in st.session_state.selected_samples_for_retraining.items() if v])
                        st.metric("Classes to Train", classes_to_train)
                    with col3:
                        current_classes = st.session_state.detector.model.num_classes
                        final_classes = current_classes + classes_to_train
                        st.metric("Final Total Classes", final_classes)
                    
                    # Training parameters
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        multi_epochs = st.slider("Training epochs", 1, 100, 10, 1, key="multi_epochs")
                    with col2:
                        multi_lr = st.selectbox("Learning rate", [0.0001, 0.001, 0.01], index=1, key="multi_lr")
                    with col3:
                        freeze_backbone = st.checkbox("🔒 Freeze backbone", value=True, 
                                                    help="If checked, only trains classification head. If unchecked, trains entire model.")
                    
                    # Show training strategy info
                    if freeze_backbone:
                        st.info("🔒 **Frozen Backbone**: Only the classification head will be updated. "
                                "Preserves CIFAR-10 features but may limit new class learning.")
                    else:
                        st.warning("🔓 **Unfrozen Backbone**: Entire model will be updated. "
                                "Better new class learning but risk of catastrophic forgetting.")
                    
                    # Train button
                    if st.button("Train Multi-Class Model", type="primary"):
                        with st.spinner("Training multi-class model with selected samples..."):
                            # Filter out empty selections
                            filtered_selections = {k: v for k, v in st.session_state.selected_samples_for_retraining.items() if v}
                            
                            if filtered_selections:
                                # Train all classes simultaneously
                                retrain_results = st.session_state.detector.new_class_learner.train_new_classes(
                                    filtered_selections,
                                    epochs=multi_epochs,
                                    lr=multi_lr,
                                    freeze_backbone=freeze_backbone  # Pass the freeze option
                                )
                                
                                st.session_state.retrain_results = retrain_results
                                #st.session_state.selected_samples_for_retraining = {}
                                
                                if retrain_results:
                                    st.success(f"Successfully trained {len(retrain_results)} new classes!")
                                    
                                    st.subheader("Training Progress")
            
                                    # Plot training curves (combined for all classes since they're trained together)
                                    fig = make_subplots(
                                        rows=1, cols=2,
                                        subplot_titles=('Training Loss', 'Training Accuracy')
                                    )
                                    
                                    # Get the training curves from any class (they're all identical since trained together)
                                    first_result = next(iter(retrain_results.values()))
                                    epochs = list(range(1, len(first_result['losses']) + 1))
                                    trained_class_names = list(retrain_results.keys())
                                    
                                    # Plot single combined curve for all classes
                                    fig.add_trace(
                                        go.Scatter(
                                            x=epochs, y=first_result['losses'], 
                                            mode='lines+markers', 
                                            name=f'Combined Loss ({len(trained_class_names)} classes)',
                                            line=dict(color='blue', width=3)
                                        ),
                                        row=1, col=1
                                    )
                                    
                                    # Accuracy curve  
                                    fig.add_trace(
                                        go.Scatter(
                                            x=epochs, y=first_result['accuracies'], 
                                            mode='lines+markers', 
                                            name=f'Combined Accuracy ({len(trained_class_names)} classes)',
                                            line=dict(color='green', width=3)
                                        ),
                                        row=1, col=2
                                    )
                                    
                                    fig.update_xaxes(title_text="Epochs", row=1, col=1)
                                    fig.update_xaxes(title_text="Epochs", row=1, col=2) 
                                    fig.update_yaxes(title_text="Loss", row=1, col=1)
                                    fig.update_yaxes(title_text="Accuracy (%)", row=1, col=2)
                                    class_list = ', '.join(trained_class_names)
                                    fig.update_layout(height=400, title=f"Multi-Class Training Progress: {class_list}")
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Show final metrics table
                                    st.subheader("Final Training Metrics")
                                    metrics_data = []
                                    for class_name, result in retrain_results.items():
                                        metrics_data.append({
                                            'Class': class_name,
                                            'Final Loss': f"{result['final_loss']:.4f}",
                                            'Final Accuracy': f"{result['final_accuracy']:.2f}%",
                                            'Samples Used': result['samples_used'],
                                            'Backbone': 'Frozen' if result['backbone_frozen'] else 'Unfrozen'
                                        })
                                    
                                    df_metrics = pd.DataFrame(metrics_data)
                                    st.dataframe(df_metrics, use_container_width=True)

                                    # Show training strategy used
                                    backbone_status = "FROZEN" if freeze_backbone else "UNFROZEN"
                                    st.info(f"Training completed with {backbone_status} backbone")
                                    
                                    # Show final model status
                                    total_classes = st.session_state.detector.model.num_classes
                                    st.info(f"Model now classifies {total_classes} classes total!")
                                    
                                    # Show what was trained
                                    trained_classes = list(retrain_results.keys())
                                    st.write(f"**Newly trained classes:** {', '.join(trained_classes)}")
                                else:
                                    st.error("Training failed!")
                            else:
                                st.warning("No samples selected for training!")

                if 'retrain_results' in st.session_state:
                    st.markdown("---")
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Clear Training Results", help="Clear training results but keep selected samples"):
                            del st.session_state.retrain_results
                            st.success("Training results cleared!")
                            st.rerun()
                    with col2:
                        if st.button("Clear All Selections", help="Clear all selected samples for retraining"):
                            st.session_state.selected_samples_for_retraining = {}
                            st.success("All selections cleared!")
                            st.rerun()
                
                else:
                    st.info("👈 Select samples from the classes above to enable multi-class training")
                    
        
            else:
                st.info("No samples available for selection. Run discovery batches first!")
                
                st.markdown("""
                ### Multi-Class Training:
                
                This approach trains a **true multi-class classifier** that can distinguish between:
                - All 10 original CIFAR-10 classes
                - Up to 5 new CIFAR-100 classes
                
                **Key Benefits:**
                - Single unified model with up to 15 classes
                - Proper softmax classification across all classes
                - No need for separate binary classifiers
                - Clean, interpretable predictions
                
                **Training Process:**
                1. Select high-quality samples from discovered OOD samples
                2. Add new classes to the model architecture
                3. Train the expanded classifier on selected samples
                4. Evaluate performance across all classes
                """)
            # else:
            #     st.info("No samples available for selection. Run discovery batches first!")
                
            #     st.markdown("""
            #     ### Manual Sample Selection:
                
            #     1. **Visual Inspection**: See all found target samples with their images
            #     2. **Quality Assessment**: Check scores, detection status, and confidence
            #     3. **Manual Curation**: Select only the best samples for retraining
            #     4. **Batch Retraining**: Retrain model with curated samples
            #     5. **Performance Tracking**: Monitor improvement after retraining
            #     """)

        with tabs[5]:
            st.header("Feature Space Visualization")
        
            if not st.session_state.model_trained:
                st.info("Train the base model first to visualize feature space")
                return
            
            # Visualization controls
            col1, col2, col3 = st.columns(3)
            
            with col1:
                viz_method = st.selectbox(
                    "Visualization Method",
                    ["Both PCA & t-SNE", "PCA only", "t-SNE only"],
                    index=0
                )
            
            with col2:
                include_new = st.checkbox(
                    "Include New Classes",
                    value=True,
                    help="Include newly learned classes in visualization"
                )
            
            with col3:
                feature_source = st.selectbox(
                    "Feature Source",
                    ["Training Features", "All Discovered", "Selected Samples"],
                    index=0
                )
            
            # Map selection to method parameter
            method_map = {
                "Both PCA & t-SNE": "both",
                "PCA only": "pca", 
                "t-SNE only": "tsne"
            }
            method = method_map[viz_method]
            
            # Advanced options
            with st.expander("Advanced Visualization Options"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**t-SNE Parameters:**")
                    tsne_perplexity = st.slider("Perplexity", 5, 100, 50, 5,
                                              help="Controls local vs global structure balance")
                    tsne_learning_rate = st.slider("Learning Rate", 50, 1000, 200, 50,
                                                  help="Controls how fast t-SNE learns")
                    
                with col2:
                    st.markdown("**Visual Options:**")
                    marker_size = st.slider("Marker Size", 4, 20, 10, 1)
                    marker_opacity = st.slider("Marker Opacity", 0.3, 1.0, 0.8, 0.1)
                    use_borders = st.checkbox("White Borders", value=True,
                                            help="Add white borders around markers")
                    
                with col3:
                    st.markdown("**Color Schemes:**")
                    color_scheme = st.selectbox("Color Palette", 
                                               ["Set1", "Dark2", "Pastel1", "Set3", "Viridis", "Plasma"],
                                               index=0)
                    plot_height = st.slider("Plot Height", 400, 800, 500, 50)
                    
                # Store advanced options in session state
                st.session_state.viz_options = {
                    'tsne_perplexity': tsne_perplexity,
                    'tsne_learning_rate': tsne_learning_rate,
                    'marker_size': marker_size,
                    'marker_opacity': marker_opacity,
                    'use_borders': use_borders,
                    'color_scheme': color_scheme,
                    'plot_height': plot_height
                }
            
            if st.button("Generate Visualization", type="primary"):
                with st.spinner("Computing dimensionality reduction..."):
                    try:
                        detector = st.session_state.detector
                        
                        # Get features and labels based on selection
                        if feature_source == "Training Features":
                            features = detector.training_features
                            labels = detector.training_labels
                            class_names = detector.cifar10_classes
                            title_suffix = "(Training Data)"
                        
                        elif feature_source == "All Discovered" and st.session_state.discovery_engine:
                            # Combine all discovered samples
                            all_images = []
                            all_labels = []
                            all_class_names = []
                            
                            for result in st.session_state.discovery_results:
                                all_images.extend(result['batch_data']['images'])
                                all_labels.extend(result['batch_data']['class_names'])
                            
                            if all_images:
                                features = detector.new_class_learner.extract_features(all_images)
                                # Create numeric labels
                                unique_names = list(set(all_labels))
                                label_map = {name: idx for idx, name in enumerate(unique_names)}
                                labels = np.array([label_map[name] for name in all_labels])
                                class_names = unique_names
                                title_suffix = "(All Discovered Samples)"
                            else:
                                st.warning("No discovered samples available")
                                return
                        
                        elif feature_source == "Selected Samples":
                            # Use selected samples for retraining
                            if not st.session_state.selected_samples_for_retraining:
                                st.warning("No samples selected for retraining")
                                return
                            
                            all_images = []
                            all_labels = []
                            all_class_names = []
                            
                            for class_name, samples in st.session_state.selected_samples_for_retraining.items():
                                for sample in samples:
                                    all_images.append(sample['image'])
                                    all_labels.append(class_name)
                            
                            if all_images:
                                features = detector.new_class_learner.extract_features(all_images)
                                unique_names = list(set(all_labels))
                                label_map = {name: idx for idx, name in enumerate(unique_names)}
                                labels = np.array([label_map[name] for name in all_labels])
                                class_names = unique_names
                                title_suffix = "(Selected Samples)"
                            else:
                                st.warning("No selected samples available")
                                return
                        
                        # Compute dimensionality reduction
                        # Get advanced options if available
                        viz_options = st.session_state.get('viz_options', {})
                        
                        reduction_results = detector.compute_dimensionality_reduction(
                            features, labels, method=method, **viz_options
                        )
                        
                        # Store results for later use
                        st.session_state.reduction_results = reduction_results
                        st.session_state.current_class_names = class_names
                        st.session_state.current_title_suffix = title_suffix
                        
                        st.success("Visualization computed successfully!")
                        
                    except Exception as e:
                        st.error(f"Error computing visualization: {str(e)}")
            
            # Display results if available
            if 'reduction_results' in st.session_state:
                reduction_results = st.session_state.reduction_results
                class_names = st.session_state.get('current_class_names', None)
                title_suffix = st.session_state.get('current_title_suffix', "")
                
                # Main visualization
                viz_options = st.session_state.get('viz_options', {})
                fig = st.session_state.detector.plot_dimensionality_reduction(
                    reduction_results, class_names, title_suffix, **viz_options
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Additional metrics and info
                col1, col2 = st.columns(2)
                
                with col1:
                    if 'pca' in reduction_results:
                        st.subheader("PCA Analysis")
                        pca_data = reduction_results['pca']
                        
                        st.write(f"**Explained Variance:**")
                        st.write(f"- PC1: {pca_data['explained_variance_ratio'][0]:.2%}")
                        st.write(f"- PC2: {pca_data['explained_variance_ratio'][1]:.2%}")
                        st.write(f"- Total: {sum(pca_data['explained_variance_ratio']):.2%}")
                        
                        # Plot explained variance
                        fig_var = plot_pca_variance_explained(st.session_state.detector)
                        st.plotly_chart(fig_var, use_container_width=True)
                
                with col2:
                    # Class separation metrics
                    st.subheader("Separation Quality")
                    fig_sep = plot_class_separation_metrics(st.session_state.detector, reduction_results)
                    st.plotly_chart(fig_sep, use_container_width=True)
                    
                    # Feature space statistics
                    if 'tsne' in reduction_results:
                        st.write("**t-SNE Info:**")
                        st.write("- Perplexity: 30")
                        st.write("- Iterations: 1000")
                        st.write("- Initialization: PCA")
                
                # Evolution visualization if new classes exist
                if (st.session_state.detector.model.num_classes > 10 and 
                    hasattr(st.session_state.detector, 'new_class_learner')):
                    
                    st.subheader("Feature Space Evolution")
                    
                    if st.button("Show Evolution Visualization"):
                        with st.spinner("Computing evolution visualization..."):
                            try:
                                fig_evolution = st.session_state.detector.visualize_feature_space_evolution()
                                st.plotly_chart(fig_evolution, use_container_width=True)
                                
                                st.info("This shows how the feature space changes as new classes are added. "
                                    "Different colors represent different data sources (original vs new classes).")
                            
                            except Exception as e:
                                st.error(f"Error creating evolution visualization: {str(e)}")
            
            else:
                st.info("Click 'Generate Visualization' to see feature space plots")
                
                st.markdown("""
                ### Feature Space Visualization
                
                **PCA (Principal Component Analysis):**
                - Linear dimensionality reduction
                - Shows directions of maximum variance
                - Preserves global structure
                - Good for understanding overall data distribution
                
                **t-SNE (t-Distributed Stochastic Neighbor Embedding):**
                - Non-linear dimensionality reduction  
                - Preserves local neighborhood structure
                - Better for visualizing clusters and separation
                - Can reveal hidden patterns in data
                
                **Evolution Visualization:**
                - Shows how feature space changes with new classes
                - Helps understand if new classes are well-separated
                - Useful for diagnosing training issues
                """)

    
    # Footer
    st.markdown("---")
    with st.expander("System Guide"):
        st.markdown("""
        ### New Bootstrap & Discovery Workflow:
        
        **Phase 1: Base Training**
        - Train CNN on subset of CIFAR-10 classes
        - Establish baseline performance and feature representations
        
        **Phase 2: Class Selection & Bootstrap Training**  
        - Choose up to 5 CIFAR-100 classes to learn
        - Bootstrap train each class with X initial samples
        - Learn class-specific binary classification heads
        
        **Phase 3: Sample Discovery**
        - Sample 100 random images from all CIFAR-100 classes
        - Score all samples using selection criteria
        - Return top 10 highest-scoring samples
        - Track which samples belong to target classes
        - Analyze whether model detected target samples as novel
        
        **Phase 4: Manual Curation**
        - Review all found target samples with their scores
        - See detection results (found vs missed)
        - Manually select best samples for retraining
        - Visual interface for sample selection
        
        **Phase 5: Targeted Retraining**
        - Retrain only on manually selected samples
        - Track improvement after each retraining cycle
        - Compare performance across different sample selections
        
        ### Key Features:
        - **Bootstrap Training**: Initial learning on target classes
        - **Random Discovery**: Unbiased sampling from all classes
        - **Detection Tracking**: See if model finds target samples
        - **Manual Curation**: Choose quality samples for retraining
        - **Performance Analytics**: Track discovery and training effectiveness
        - **Visual Interface**: Easy sample selection and review
        """)


if __name__ == "__main__":
    main()