import numpy as np
import torch
from PIL import Image
import math
from skimage.metrics import structural_similarity
import cv2

def calculate_psnr(img1, img2, max_value=255.0):
    """
    Calculate PSNR (Peak Signal-to-Noise Ratio) between two images.
    Higher values indicate better quality.

    Args:
        img1: First image (original/reference)
        img2: Second image (processed/distorted)
        max_value: Maximum possible pixel value (default: 255.0)

    Returns:
        PSNR value in dB (higher is better)
    """
    try:
        # Convert PIL images to numpy arrays if needed
        if isinstance(img1, Image.Image):
            img1 = np.array(img1).astype(np.float32)
        if isinstance(img2, Image.Image):
            img2 = np.array(img2).astype(np.float32)

        # Convert torch tensors to numpy arrays if needed
        if isinstance(img1, torch.Tensor):
            img1 = img1.cpu().numpy().astype(np.float32)
        if isinstance(img2, torch.Tensor):
            img2 = img2.cpu().numpy().astype(np.float32)

        # Ensure images have the same shape
        if img1.shape != img2.shape:
            # Resize the second image to match the first
            if len(img1.shape) == 3:  # Color images
                img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_CUBIC)
            else:  # Grayscale images
                img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_CUBIC)

        # Handle different color channels
        if len(img1.shape) == 3 and img1.shape[2] == 3:
            # Convert to grayscale for comparison
            if img1.dtype != np.uint8:
                img1_gray = cv2.cvtColor((img1 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
                img2_gray = cv2.cvtColor((img2 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
            else:
                img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY).astype(np.float32)
                img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY).astype(np.float32)

            mse = np.mean((img1_gray - img2_gray) ** 2)
            if mse == 0:
                psnr = float('inf')  # Perfect similarity
            else:
                psnr = 20 * math.log10(max_value / math.sqrt(mse))
        else:
            # Grayscale images
            mse = np.mean((img1 - img2) ** 2)
            if mse == 0:
                psnr = float('inf')  # Perfect similarity
            else:
                psnr = 20 * math.log10(max_value / math.sqrt(mse))

        return psnr

    except Exception as e:
        print(f"Error calculating PSNR: {str(e)}")
        return 0.0  # Return low value on error

def calculate_ssim(img1, img2):
    """
    Calculate SSIM (Structural Similarity Index) between two images.
    Values range from -1 to 1, where 1 indicates perfect similarity.

    Args:
        img1: First image (original/reference)
        img2: Second image (processed/distorted)

    Returns:
        SSIM value in range [-1, 1]
    """
    try:
        # Convert PIL images to numpy arrays if needed
        if isinstance(img1, Image.Image):
            img1 = np.array(img1)
        if isinstance(img2, Image.Image):
            img2 = np.array(img2)

        # Convert torch tensors to numpy arrays if needed
        if isinstance(img1, torch.Tensor):
            img1 = img1.cpu().numpy()
        if isinstance(img2, torch.Tensor):
            img2 = img2.cpu().numpy()

        # Ensure images have the same shape
        if img1.shape != img2.shape:
            # Resize the second image to match the first
            if len(img1.shape) == 3:  # Color images
                img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_CUBIC)
            else:  # Grayscale images
                img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_CUBIC)

        # Handle different color channels
        if len(img1.shape) == 3 and img1.shape[2] == 3:
            # Convert to uint8 if needed
            if img1.dtype != np.uint8:
                img1 = (img1 * 255).astype(np.uint8)
                img2 = (img2 * 255).astype(np.uint8)

            # Convert RGB to grayscale for SSIM calculation
            img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
            img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

            # Calculate SSIM
            ssim_value, _ = structural_similarity(img1_gray, img2_gray, full=True, data_range=255)
        else:
            # Grayscale images
            if img1.dtype != np.uint8:
                img1 = (img1 * 255).astype(np.uint8)
                img2 = (img2 * 255).astype(np.uint8)

            # Calculate SSIM with proper data range
            ssim_value, _ = structural_similarity(
                img1,
                img2,
                full=True,
                data_range=255
            )

        # Ensure SSIM is in the range [-1, 1]
        ssim_value = max(-1.0, min(1.0, ssim_value))

        return ssim_value

    except Exception as e:
        print(f"Error calculating SSIM: {str(e)}")
        return 0.0  # Return neutral value on error

def evaluate_image_quality(original_img, processed_img):
    """
    Evaluate image quality using PSNR and SSIM metrics.

    Args:
        original_img: Original/reference image
        processed_img: Processed/distorted image

    Returns:
        Dictionary containing PSNR and SSIM values and quality assessment
    """
    try:
        psnr_value = calculate_psnr(original_img, processed_img)
        ssim_value = calculate_ssim(original_img, processed_img)

        # Determine quality assessment based on metrics
        if psnr_value > 30 and ssim_value > 0.9:
            quality = "excellent"
        elif psnr_value > 25 and ssim_value > 0.8:
            quality = "good"
        elif psnr_value > 20 and ssim_value > 0.7:
            quality = "acceptable"
        else:
            quality = "poor"

        return {
            'psnr': psnr_value,
            'ssim': ssim_value,
            'quality': quality
        }
    except Exception as e:
        print(f"Error evaluating image quality: {str(e)}")
        return {
            'psnr': 0.0,
            'ssim': 0.0,
            'quality': 'error',
            'error': str(e)
        }
