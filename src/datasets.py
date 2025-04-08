import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from torchvision import transforms
import random
import glob
from typing import List, Optional, Union
import cv2
from pathlib import Path

class SRDataset(Dataset):
    """Base Super Resolution Dataset"""
    def __init__(
        self,
        data_dir: str,
        scale_factor: int = 2,
        patch_size: int = 96,
        augment: bool = True,
        split: str = 'train',
        extensions: List[str] = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    ):
        self.data_dir = data_dir
        self.scale_factor = scale_factor
        self.patch_size = patch_size
        self.augment = augment
        self.split = split

        # Find all image files
        self.image_files = []
        for ext in extensions:
            self.image_files.extend(glob.glob(os.path.join(data_dir, f'*{ext}')))

        # Sort for reproducibility
        self.image_files.sort()

        # Split dataset if needed
        if split == 'train':
            self.image_files = self.image_files[:int(0.8 * len(self.image_files))]
        elif split == 'val':
            self.image_files = self.image_files[int(0.8 * len(self.image_files)):]

        # Basic transforms
        self.to_tensor = transforms.ToTensor()

        # Augmentation transforms
        self.augment_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(90),
        ])

    def __len__(self):
        return len(self.image_files)

    def _get_patch(self, img, patch_size):
        """Extract a random patch from the image"""
        h, w = img.size

        # If image is smaller than patch size, pad it
        if h < patch_size or w < patch_size:
            padding = max(0, patch_size - h, patch_size - w)
            img = transforms.Pad(padding)(img)
            h, w = img.size

        # Extract random patch
        x = random.randint(0, w - patch_size)
        y = random.randint(0, h - patch_size)

        return img.crop((y, x, y + patch_size, x + patch_size))

    def __getitem__(self, idx):
        img_path = self.image_files[idx]

        try:
            # Load image
            img = Image.open(img_path).convert('L')  # Convert to grayscale

            # Training mode: extract patches
            if self.split == 'train':
                img = self._get_patch(img, self.patch_size)

                # Apply augmentations if enabled
                if self.augment:
                    img = self.augment_transforms(img)

            # Convert to tensor
            hr_tensor = self.to_tensor(img)

            # Create low-resolution version
            lr_size = tuple(dim // self.scale_factor for dim in hr_tensor.shape[-2:])
            lr_tensor = F.resize(hr_tensor, lr_size, interpolation=transforms.InterpolationMode.BICUBIC)

            # Resize LR back to HR size for model input
            lr_tensor = F.resize(lr_tensor, hr_tensor.shape[-2:], interpolation=transforms.InterpolationMode.BICUBIC)

            return {
                'lr': lr_tensor,
                'hr': hr_tensor,
                'filename': os.path.basename(img_path)
            }

        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Create properly initialized dummy tensors
            dummy_size = (1, self.patch_size, self.patch_size)
            dummy_lr = torch.zeros(dummy_size, dtype=torch.float32)
            dummy_hr = torch.zeros(dummy_size, dtype=torch.float32)

            return {
                'lr': dummy_lr,
                'hr': dummy_hr,
                'filename': 'error.jpg'
            }


class MedicalImageDataset(SRDataset):
    """Dataset specifically for medical images with additional preprocessing"""
    def __init__(
        self,
        data_dir: str,
        scale_factor: int = 2,
        patch_size: int = 96,
        augment: bool = True,
        split: str = 'train',
        extensions: List[str] = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.dcm', '.nii', '.nii.gz'],
        normalize: bool = True
    ):
        super().__init__(data_dir, scale_factor, patch_size, augment, split, extensions)
        self.normalize = normalize

    def _preprocess_medical_image(self, img_tensor):
        """Apply medical-specific preprocessing"""
        # Normalize to [0, 1] range if not already
        if self.normalize:
            if torch.min(img_tensor) < 0 or torch.max(img_tensor) > 1:
                img_tensor = (img_tensor - torch.min(img_tensor)) / (torch.max(img_tensor) - torch.min(img_tensor))

        # Apply contrast stretching for better feature visibility
        p_low, p_high = torch.quantile(img_tensor, torch.tensor([0.02, 0.98]))
        img_tensor = torch.clamp((img_tensor - p_low) / (p_high - p_low), 0, 1)

        return img_tensor

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)

        # Apply medical-specific preprocessing
        sample['hr'] = self._preprocess_medical_image(sample['hr'])
        sample['lr'] = self._preprocess_medical_image(sample['lr'])

        return sample


class BatchProcessor:
    """Handles batch processing of images for super-resolution"""
    def __init__(
        self,
        model,
        device,
        batch_size: int = 4,
        scale_factor: int = 2,
        save_dir: Optional[str] = None,
        preserve_color: bool = False
    ):
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.scale_factor = scale_factor
        self.save_dir = save_dir
        self.preserve_color = preserve_color

        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

    def process_directory(self, input_dir: str, output_dir: Optional[str] = None):
        """Process all images in a directory"""
        if output_dir is None:
            output_dir = self.save_dir or os.path.join(input_dir, 'results')

        os.makedirs(output_dir, exist_ok=True)

        # Get all image files in the directory
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
            image_files.extend(glob.glob(os.path.join(input_dir, f'*{ext}')))

        # Process batches
        self.model.eval()
        results = []

        # Process images in batches
        for i in range(0, len(image_files), self.batch_size):
            batch_files = image_files[i:i+self.batch_size]
            batch_images = []
            filenames = []
            original_images = []

            # Load images
            for img_path in batch_files:
                try:
                    img = Image.open(img_path)
                    filenames.append(os.path.basename(img_path))

                    # Store original image for color preservation if needed
                    if self.preserve_color and img.mode == 'RGB':
                        original_images.append(img.copy())
                    else:
                        original_images.append(None)

                    # Convert to tensor
                    if self.model.__class__.__name__.startswith('Color'):
                        # For color models, use RGB
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        img_tensor = transforms.ToTensor()(img)
                    else:
                        # For grayscale models, convert to grayscale
                        if img.mode != 'L':
                            img = img.convert('L')
                        img_tensor = transforms.ToTensor()(img)

                    batch_images.append(img_tensor)
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")
                    continue

            if not batch_images:
                continue

            # Stack into batch
            input_batch = torch.stack(batch_images).to(self.device)

            # Process through model
            with torch.no_grad():
                sr_batch = self.model(input_batch)

            # Save results
            for sr_img, filename, original in zip(sr_batch, filenames, original_images):
                # Convert to PIL image
                sr_pil = transforms.ToPILImage()(sr_img.cpu())

                # Apply color preservation if needed
                if self.preserve_color and original is not None and original.mode == 'RGB' and not self.model.__class__.__name__.startswith('Color'):
                    # Convert original to YCbCr
                    original_ycbcr = original.convert('YCbCr')
                    _, cb, cr = original_ycbcr.split()

                    # Use SR result as Y channel
                    sr_y = sr_pil.convert('L')

                    # Resize Cb and Cr to match the new Y dimensions
                    new_size = sr_y.size
                    cb = cb.resize(new_size, Image.BICUBIC)
                    cr = cr.resize(new_size, Image.BICUBIC)

                    # Merge channels and convert back to RGB
                    sr_pil = Image.merge('YCbCr', [sr_y, cb, cr]).convert('RGB')

                # Save to output directory
                output_path = os.path.join(output_dir, filename)
                sr_pil.save(output_path)

                results.append({
                    'filename': filename,
                    'path': output_path
                })

        return results

    def process_batch(self, images: List[Union[str, Image.Image, torch.Tensor]]) -> List[torch.Tensor]:
        """Process a batch of images and return super-resolved versions"""
        processed_images = []
        original_images = []

        # Convert all images to tensors and ensure they are properly normalized
        for img in images:
            if isinstance(img, str):
                # Load from path
                original_img = Image.open(img)
                if self.preserve_color and original_img.mode == 'RGB':
                    original_images.append(original_img.copy())
                else:
                    original_images.append(None)

                if self.model.__class__.__name__.startswith('Color'):
                    # For color models, use RGB
                    if original_img.mode != 'RGB':
                        img = original_img.convert('RGB')
                    else:
                        img = original_img
                    img = transforms.ToTensor()(img)
                else:
                    # For grayscale models, convert to grayscale
                    img = original_img.convert('L')
                    img = transforms.ToTensor()(img)
            elif isinstance(img, Image.Image):
                if self.preserve_color and img.mode == 'RGB':
                    original_images.append(img.copy())
                else:
                    original_images.append(None)

                if self.model.__class__.__name__.startswith('Color'):
                    # For color models, use RGB
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    img = transforms.ToTensor()(img)
                else:
                    # For grayscale models, convert to grayscale
                    img = transforms.ToTensor()(img.convert('L'))
            elif isinstance(img, np.ndarray):
                original_images.append(None)  # No color preservation for numpy arrays

                if self.model.__class__.__name__.startswith('Color'):
                    # For color models, ensure RGB
                    if len(img.shape) == 2:  # Grayscale
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                    elif len(img.shape) == 3 and img.shape[2] == 1:  # Grayscale with channel
                        img = cv2.cvtColor(img.squeeze(2), cv2.COLOR_GRAY2RGB)
                    img = transforms.ToTensor()(img)
                else:
                    # For grayscale models, convert to grayscale
                    if len(img.shape) == 3:
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                    img = transforms.ToTensor()(img)
            elif isinstance(img, torch.Tensor):
                original_images.append(None)  # No color preservation for tensors

                if self.model.__class__.__name__.startswith('Color'):
                    # For color models, ensure 3 channels
                    if img.dim() == 2:  # Grayscale image
                        img = img.unsqueeze(0).repeat(3, 1, 1)
                    elif img.dim() == 3 and img.shape[0] == 1:  # Grayscale with channel
                        img = img.repeat(3, 1, 1)
                else:
                    # For grayscale models, ensure 1 channel
                    if img.dim() == 2:  # Grayscale image
                        img = img.unsqueeze(0)
                    elif img.dim() == 3 and img.shape[0] == 3:  # RGB image
                        # Convert to grayscale (simple average)
                        img = img.mean(dim=0, keepdim=True)

            # Ensure tensor is in range [0, 1]
            img = torch.clamp(img, 0, 1)
            processed_images.append(img)

        # Find maximum size in batch
        max_h = max(img.shape[1] for img in processed_images)
        max_w = max(img.shape[2] for img in processed_images)

        # Pad all images to same size
        padded_images = []
        for img in processed_images:
            padding = transforms.Pad(
                padding=(0, 0, max_w - img.shape[2], max_h - img.shape[1]),
                padding_mode='reflect'
            )
            img = padding(img)
            padded_images.append(img)

        # Stack into batch
        batch = torch.stack(padded_images).to(self.device)

        # Process through model
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(batch)

        return outputs


class RealTimeProcessor:
    """Handles real-time processing of images for super-resolution"""
    def __init__(
        self,
        model,
        device,
        scale_factor: int = 2,
        max_size: int = 512,
        preserve_color: bool = False
    ):
        self.model = model
        self.device = device
        self.scale_factor = scale_factor
        self.max_size = max_size
        self.preserve_color = preserve_color

        # Set model to evaluation mode
        self.model.eval()

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame for real-time applications"""
        original_frame = frame.copy()
        is_color = len(frame.shape) == 3 and frame.shape[2] == 3
        is_color_model = self.model.__class__.__name__.startswith('Color')

        # Prepare input based on model type and color preservation settings
        if is_color_model:
            # For color models, use RGB
            if not is_color:
                # Convert grayscale to RGB if needed
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            else:
                rgb_frame = frame

            # Resize if too large
            h, w, _ = rgb_frame.shape
            if max(h, w) > self.max_size:
                scale = self.max_size / max(h, w)
                new_size = (int(w * scale), int(h * scale))
                rgb_frame = cv2.resize(rgb_frame, new_size, interpolation=cv2.INTER_AREA)

            # Convert to tensor
            frame_tensor = transforms.ToTensor()(rgb_frame).unsqueeze(0).to(self.device)
        else:
            # For grayscale models, convert to grayscale
            if is_color:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            else:
                gray_frame = frame

            # Resize if too large
            h, w = gray_frame.shape
            if max(h, w) > self.max_size:
                scale = self.max_size / max(h, w)
                new_size = (int(w * scale), int(h * scale))
                gray_frame = cv2.resize(gray_frame, new_size, interpolation=cv2.INTER_AREA)

            # Convert to tensor
            frame_tensor = transforms.ToTensor()(gray_frame).unsqueeze(0).to(self.device)

        # Process through model
        with torch.no_grad():
            output = self.model(frame_tensor)
            output = torch.clamp(output, 0, 1)

        # Handle color preservation for grayscale models
        if self.preserve_color and is_color and not is_color_model:
            # Convert output to PIL Image
            output_pil = transforms.ToPILImage()(output.squeeze(0).cpu())

            # Convert original frame to PIL Image
            if max(original_frame.shape[:2]) > self.max_size:
                scale = self.max_size / max(original_frame.shape[:2])
                new_size = (int(original_frame.shape[1] * scale), int(original_frame.shape[0] * scale))
                original_frame = cv2.resize(original_frame, new_size, interpolation=cv2.INTER_AREA)

            original_pil = Image.fromarray(cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB))

            # Convert to YCbCr color space
            original_ycbcr = original_pil.convert('YCbCr')
            _, cb, cr = original_ycbcr.split()

            # Replace Y channel with super-resolved version
            sr_y = output_pil.convert('L')

            # Resize Cb and Cr to match the new Y dimensions
            new_size = sr_y.size
            cb = cb.resize(new_size, Image.BICUBIC)
            cr = cr.resize(new_size, Image.BICUBIC)

            # Merge channels and convert back to RGB
            result_pil = Image.merge('YCbCr', [sr_y, cb, cr]).convert('RGB')

            # Convert back to numpy array
            result = np.array(result_pil)
        else:
            # Convert output tensor to numpy array
            if is_color_model or (self.preserve_color and is_color):
                # For color output
                result = output.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255
            else:
                # For grayscale output
                result = output.squeeze().cpu().numpy() * 255

            result = result.astype(np.uint8)

        return result

    def process_video(self, input_path: str, output_path: str, fps: int = 30):
        """Process a video file"""
        # Open video
        cap = cv2.VideoCapture(input_path)

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Create output video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=False)

        # Process each frame
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            processed_frame = self.process_frame(gray_frame)

            # Write to output
            out.write(processed_frame)

            frame_count += 1
            if frame_count % 10 == 0:
                print(f"Processed {frame_count}/{total_frames} frames")

        # Release resources
        cap.release()
        out.release()

        return output_path


class LIDCDataset(Dataset):
    """LIDC-IDRI Dataset for lung nodules"""
    def __init__(self, root_dir, transform=None, scale_factor=2, cache_data=True):
        self.root_dir = root_dir
        self.transform = transform
        self.scale_factor = scale_factor
        self.to_tensor = transforms.ToTensor()
        self.cache_data = cache_data
        self.cache = {}

        try:
            # Try to import pydicom
            import pydicom
            self.pydicom = pydicom
        except ImportError:
            print("Warning: pydicom not installed. LIDCDataset will not work properly.")
            self.pydicom = None

        # Find all DICOM files
        self.image_files = []
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith('.dcm'):
                    self.image_files.append(os.path.join(root, file))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        if self.cache_data and idx in self.cache:
            return self.cache[idx]

        if self.pydicom is None:
            raise ImportError("pydicom is required to use LIDCDataset")

        # Read DICOM file
        dcm = self.pydicom.dcmread(self.image_files[idx])
        image = dcm.pixel_array.astype(float)

        # Normalize to [0, 1]
        image = (image - image.min()) / (image.max() - image.min())

        # Convert to PIL Image
        image = Image.fromarray((image * 255).astype(np.uint8))

        # Create high-resolution image
        hr_image = self.to_tensor(image)

        # Create low-resolution image
        lr_size = tuple(dim // self.scale_factor for dim in hr_image.shape[-2:])
        lr_image = F.resize(hr_image, lr_size, interpolation=transforms.InterpolationMode.BICUBIC)
        lr_image = F.resize(lr_image, hr_image.shape[-2:], interpolation=transforms.InterpolationMode.BICUBIC)

        if self.transform:
            hr_image = self.transform(hr_image)
            lr_image = self.transform(lr_image)

        # Cache the processed images
        if self.cache_data:
            self.cache[idx] = (lr_image, hr_image)

        return lr_image, hr_image

class BraTSDataset(Dataset):
    """BraTS Dataset for brain tumor segmentation"""
    def __init__(self, root_dir, transform=None, scale_factor=2, modality='t1'):
        self.root_dir = root_dir
        self.transform = transform
        self.scale_factor = scale_factor
        self.modality = modality
        self.to_tensor = transforms.ToTensor()

        try:
            # Try to import nibabel
            import nibabel as nib
            self.nib = nib
        except ImportError:
            print("Warning: nibabel not installed. BraTSDataset will not work properly.")
            self.nib = None

        # Find all NIfTI files for the specified modality
        self.image_files = []
        try:
            for path in Path(root_dir).rglob(f'*{modality}.nii.gz'):
                self.image_files.append(str(path))
        except Exception as e:
            print(f"Error finding NIfTI files: {e}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        if self.nib is None:
            raise ImportError("nibabel is required to use BraTSDataset")

        # Load NIfTI file
        nifti_img = self.nib.load(self.image_files[idx])
        image_data = nifti_img.get_fdata()

        # Take middle slice for 2D
        middle_slice = image_data[:, :, image_data.shape[2]//2]

        # Normalize to [0, 1]
        slice_norm = (middle_slice - middle_slice.min()) / (middle_slice.max() - middle_slice.min())

        # Convert to PIL Image
        image = Image.fromarray((slice_norm * 255).astype(np.uint8))

        # Create high-resolution image
        hr_image = self.to_tensor(image)

        # Create low-resolution image
        lr_size = tuple(dim // self.scale_factor for dim in hr_image.shape[-2:])
        lr_image = F.resize(hr_image, lr_size, interpolation=transforms.InterpolationMode.BICUBIC)
        lr_image = F.resize(lr_image, hr_image.shape[-2:], interpolation=transforms.InterpolationMode.BICUBIC)

        if self.transform:
            hr_image = self.transform(hr_image)
            lr_image = self.transform(lr_image)

        return lr_image, hr_image

class COVIDxDataset(Dataset):
    """COVIDx Dataset for COVID-19 chest X-rays"""
    def __init__(self, root_dir, metadata_path, transform=None, scale_factor=2):
        self.root_dir = root_dir
        self.transform = transform
        self.scale_factor = scale_factor
        self.to_tensor = transforms.ToTensor()

        try:
            # Try to import pandas
            import pandas as pd
            self.pd = pd

            # Read metadata file
            self.metadata = pd.read_csv(metadata_path)
            self.image_files = self.metadata['filename'].tolist()
        except ImportError:
            print("Warning: pandas not installed. COVIDxDataset will not work properly.")
            self.pd = None
            self.metadata = None
            self.image_files = []

    def __len__(self):
        return len(self.image_files) if self.image_files else 0

    def __getitem__(self, idx):
        if self.pd is None:
            raise ImportError("pandas is required to use COVIDxDataset")

        # Load image
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert('L')  # Convert to grayscale

        # Create high-resolution image
        hr_image = self.to_tensor(image)

        # Create low-resolution image
        lr_size = tuple(dim // self.scale_factor for dim in hr_image.shape[-2:])
        lr_image = F.resize(hr_image, lr_size, interpolation=transforms.InterpolationMode.BICUBIC)
        lr_image = F.resize(lr_image, hr_image.shape[-2:], interpolation=transforms.InterpolationMode.BICUBIC)

        if self.transform:
            hr_image = self.transform(hr_image)
            lr_image = self.transform(lr_image)

        return lr_image, hr_image

class SampleDataset(Dataset):
    """Sample dataset for testing"""
    def __init__(self, root_dir, transform=None, scale_factor=2):
        self.root_dir = root_dir
        self.transform = transform
        self.scale_factor = scale_factor

        # Find all tensor files
        self.image_files = []
        for file in os.listdir(root_dir):
            if file.endswith('.pt'):
                self.image_files.append(os.path.join(root_dir, file))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load tensor
        hr_image = torch.load(self.image_files[idx])

        # Create low-resolution image
        lr_size = tuple(dim // self.scale_factor for dim in hr_image.shape[-2:])
        lr_image = F.resize(hr_image.unsqueeze(0), lr_size, interpolation=transforms.InterpolationMode.BICUBIC).squeeze(0)
        lr_image = F.resize(lr_image.unsqueeze(0), hr_image.shape[-2:], interpolation=transforms.InterpolationMode.BICUBIC).squeeze(0)

        if self.transform:
            hr_image = self.transform(hr_image)
            lr_image = self.transform(lr_image)

        return lr_image, hr_image

def get_dataset(dataset_name, root_dir, **kwargs):
    """Factory function to get the appropriate dataset"""
    datasets = {
        'lidc': LIDCDataset,
        'brats': BraTSDataset,
        'covidx': COVIDxDataset,
        'sample': SampleDataset
    }

    if dataset_name.lower() not in datasets:
        raise ValueError(f"Dataset {dataset_name} not supported. Available datasets: {list(datasets.keys())}")

    return datasets[dataset_name.lower()](root_dir, **kwargs)
