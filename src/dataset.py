import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import numpy as np


def rgb_to_luminosity(image):
    """Convert RGB image to luminosity (grayscale) using standard luminosity weights"""
    if isinstance(image, Image.Image):
        # PIL Image - convert to numpy for processing
        img_array = np.array(image).astype(np.float32)
        # Apply luminosity weights: Y = 0.299*R + 0.587*G + 0.114*B
        luminosity = 0.299 * img_array[:, :, 0] + 0.587 * img_array[:, :, 1] + 0.114 * img_array[:, :, 2]
        # Convert back to RGB format (grayscale in all channels)
        luminosity = np.stack([luminosity, luminosity, luminosity], axis=2).astype(np.uint8)
        return Image.fromarray(luminosity)
    elif isinstance(image, torch.Tensor):
        # Tensor format - apply luminosity weights
        if len(image.shape) == 4:  # Batch dimension
            # Weights for RGB channels [0.299, 0.587, 0.114]
            weights = torch.tensor([0.299, 0.587, 0.114], device=image.device).view(1, 3, 1, 1)
            luminosity = torch.sum(image * weights, dim=1, keepdim=True)
            # Expand to 3 channels
            return luminosity.expand(-1, 3, -1, -1)
        elif len(image.shape) == 3:  # Single image
            weights = torch.tensor([0.299, 0.587, 0.114], device=image.device).view(3, 1, 1)
            luminosity = torch.sum(image * weights, dim=0, keepdim=True)
            # Expand to 3 channels
            return luminosity.expand(3, -1, -1)
    return image


def normalize_luminosity_tensor(tensor):
    """Normalize luminosity tensor to have proper distribution"""
    # Luminosity images tend to have different statistics than RGB
    # Use histogram equalization-like normalization for better contrast
    
    if len(tensor.shape) == 4:  # Batch of images [B, C, H, W]
        batch_size, channels, height, width = tensor.shape
        tensor_flat = tensor.view(batch_size, -1)
        
        # Calculate percentile-based normalization per image in batch
        min_vals = torch.quantile(tensor_flat, 0.02, dim=1, keepdim=True)
        max_vals = torch.quantile(tensor_flat, 0.98, dim=1, keepdim=True)
        
        # Reshape for broadcasting
        min_vals = min_vals.view(batch_size, 1, 1, 1)
        max_vals = max_vals.view(batch_size, 1, 1, 1)
        
        # Normalize to [0, 1] range first
        tensor_norm = (tensor - min_vals) / (max_vals - min_vals + 1e-8)
        tensor_norm = torch.clamp(tensor_norm, 0, 1)
        
        # Then normalize to [-1, 1] for GAN training
        return tensor_norm * 2.0 - 1.0
        
    elif len(tensor.shape) == 3:  # Single image [C, H, W]
        channels, height, width = tensor.shape
        tensor_flat = tensor.view(-1)
        
        # Calculate percentile-based normalization for single image
        min_val = torch.quantile(tensor_flat, 0.02)
        max_val = torch.quantile(tensor_flat, 0.98)
        
        # Normalize to [0, 1] range first
        tensor_norm = (tensor - min_val) / (max_val - min_val + 1e-8)
        tensor_norm = torch.clamp(tensor_norm, 0, 1)
        
        # Then normalize to [-1, 1] for GAN training
        return tensor_norm * 2.0 - 1.0
    
    return tensor


class LuminosityTransform:
    """Transform to convert RGB image to luminosity"""
    def __call__(self, image):
        return rgb_to_luminosity(image)


class LuminosityNormalizeTransform:
    """Transform to normalize luminosity images with adaptive normalization"""
    def __call__(self, tensor):
        if isinstance(tensor, torch.Tensor):
            return normalize_luminosity_tensor(tensor)
        return tensor


class AlbedoRoughnessDataset(Dataset):
    """Dataset for paired albedo and roughness images"""
    
    def __init__(self, root_dir, image_size=256, is_training=True):
        self.root_dir = root_dir
        self.albedo_dir = os.path.join(root_dir, 'albedo')
        self.roughness_dir = os.path.join(root_dir, 'roughness')
        self.image_size = image_size
        self.is_training = is_training
        
        # Get list of common files
        albedo_files = set(os.listdir(self.albedo_dir))
        roughness_files = set(os.listdir(self.roughness_dir))
        self.image_files = list(albedo_files.intersection(roughness_files))
        
        # Filter valid image files
        valid_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'}
        self.image_files = [f for f in self.image_files 
                           if os.path.splitext(f.lower())[1] in valid_extensions]
        
        # Define transforms
        self.albedo_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            LuminosityTransform(),  # Convert albedo to luminosity
            transforms.ToTensor(),
            LuminosityNormalizeTransform()  # Luminosity-specific normalization
        ])
        
        self.roughness_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Standard normalization for roughness
        ])
        
        # Additional augmentation for training
        if is_training:
            self.albedo_augment_transform = transforms.Compose([
                transforms.Resize((int(image_size * 1.12), int(image_size * 1.12))),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                LuminosityTransform(),  # Convert albedo to luminosity
                transforms.ToTensor(),
                LuminosityNormalizeTransform()  # Luminosity-specific normalization
            ])
            
            self.roughness_augment_transform = transforms.Compose([
                transforms.Resize((int(image_size * 1.12), int(image_size * 1.12))),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Standard normalization for roughness
            ])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        filename = self.image_files[idx]
        
        # Load albedo image
        albedo_path = os.path.join(self.albedo_dir, filename)
        albedo_image = Image.open(albedo_path).convert('RGB')
        
        # Load roughness image
        roughness_path = os.path.join(self.roughness_dir, filename)
        roughness_image = Image.open(roughness_path).convert('RGB')
        
        # Apply transforms
        if self.is_training and torch.rand(1).item() > 0.5:
            # Apply augmentation
            seed = torch.randint(0, 2**32, (1,)).item()
            
            torch.manual_seed(seed)
            albedo_tensor = self.albedo_augment_transform(albedo_image)
            
            torch.manual_seed(seed)
            roughness_tensor = self.roughness_augment_transform(roughness_image)
        else:
            # Normal transform
            albedo_tensor = self.albedo_transform(albedo_image)
            roughness_tensor = self.roughness_transform(roughness_image)
        
        return {
            'albedo': albedo_tensor,
            'roughness': roughness_tensor,
            'filename': filename
        }


def create_data_loader(root_dir, batch_size=1, image_size=256, is_training=True, num_workers=0):
    """Create a DataLoader for the dataset"""
    dataset = AlbedoRoughnessDataset(root_dir, image_size, is_training)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_training,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=is_training
    )
    
    return dataloader, len(dataset)


def denormalize_tensor(tensor):
    """Denormalize tensor from [-1, 1] to [0, 1]"""
    return (tensor + 1.0) / 2.0


def tensor_to_pil(tensor):
    """Convert a tensor to PIL Image"""
    tensor = denormalize_tensor(tensor)
    tensor = torch.clamp(tensor, 0, 1)
    numpy_image = tensor.cpu().numpy().transpose(1, 2, 0)
    numpy_image = (numpy_image * 255).astype(np.uint8)
    return Image.fromarray(numpy_image)
