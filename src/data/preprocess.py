import os
import cv2
import numpy as np
from PIL import Image

def load_image(image_path):
    """
    Load an image from the specified path.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        PIL.Image: Loaded image
    """
    try:
        return Image.open(image_path)
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def resize_image(image, target_size=(224, 224)):
    """
    Resize an image to the target size.
    
    Args:
        image (PIL.Image): Input image
        target_size (tuple): Target size (width, height)
        
    Returns:
        PIL.Image: Resized image
    """
    return image.resize(target_size, Image.LANCZOS)

def normalize_image(image_array):
    """
    Normalize image pixel values to [0, 1] range.
    
    Args:
        image_array (numpy.ndarray): Input image as numpy array
        
    Returns:
        numpy.ndarray: Normalized image array
    """
    return image_array / 255.0

def apply_augmentation(image, augmentation_type=None):
    """
    Apply data augmentation to the image.
    
    Args:
        image (PIL.Image): Input image
        augmentation_type (str): Type of augmentation to apply
            Options: 'rotate', 'flip', 'brightness', None
            
    Returns:
        PIL.Image: Augmented image
    """
    if augmentation_type is None:
        return image
    
    if augmentation_type == 'rotate':
        angle = np.random.randint(-30, 30)
        return image.rotate(angle)
    
    elif augmentation_type == 'flip':
        return image.transpose(Image.FLIP_LEFT_RIGHT)
    
    elif augmentation_type == 'brightness':
        enhancer = Image.ImageEnhance.Brightness(image)
        factor = np.random.uniform(0.8, 1.2)
        return enhancer.enhance(factor)
    
    return image

def extract_features(image):
    """
    Extract basic features from the image using OpenCV.
    
    Args:
        image (PIL.Image): Input image
        
    Returns:
        dict: Dictionary containing extracted features
    """
    # Convert PIL image to OpenCV format
    img = np.array(image)
    img = img[:, :, ::-1].copy()  # RGB to BGR
    
    # Convert to grayscale for feature extraction
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Extract color features
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    color_features = {
        'mean_hue': np.mean(hsv[:,:,0]),
        'mean_saturation': np.mean(hsv[:,:,1]),
        'mean_value': np.mean(hsv[:,:,2])
    }
    
    # Extract texture features
    texture_features = {
        'mean': np.mean(gray),
        'std': np.std(gray),
        'entropy': entropy(gray)
    }
    
    return {**color_features, **texture_features}

def entropy(image):
    """
    Calculate the entropy of an image as a texture feature.
    
    Args:
        image (numpy.ndarray): Grayscale image
        
    Returns:
        float: Entropy value
    """
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    return -np.sum(hist * np.log2(hist))

def preprocess_image_for_prediction(image_path, target_size=(224, 224)):
    """
    Preprocess a single image for prediction.
    
    Args:
        image_path (str): Path to the image file
        target_size (tuple): Target size for resizing
        
    Returns:
        numpy.ndarray: Preprocessed image ready for model input
    """
    # Load and resize image
    image = load_image(image_path)
    if image is None:
        return None
    
    image = resize_image(image, target_size)
    
    # Convert to numpy array and normalize
    image_array = np.array(image)
    image_array = normalize_image(image_array)
    
    # Add batch dimension
    return np.expand_dims(image_array, axis=0)

def batch_preprocess(image_dir, target_size=(224, 224), augment=False):
    """
    Preprocess all images in a directory.
    
    Args:
        image_dir (str): Directory containing images
        target_size (tuple): Target size for resizing
        augment (bool): Whether to apply data augmentation
        
    Returns:
        tuple: (preprocessed_images, file_paths)
    """
    preprocessed_images = []
    file_paths = []
    
    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(image_dir, filename)
            image = load_image(file_path)
            
            if image is not None:
                # Resize image
                image = resize_image(image, target_size)
                
                # Apply augmentation if requested
                if augment:
                    aug_types = [None, 'rotate', 'flip', 'brightness']
                    for aug_type in aug_types:
                        augmented = apply_augmentation(image, aug_type)
                        img_array = np.array(augmented)
                        img_array = normalize_image(img_array)
                        preprocessed_images.append(img_array)
                        file_paths.append(file_path)
                else:
                    # Just normalize without augmentation
                    img_array = np.array(image)
                    img_array = normalize_image(img_array)
                    preprocessed_images.append(img_array)
                    file_paths.append(file_path)
    
    return np.array(preprocessed_images), file_paths