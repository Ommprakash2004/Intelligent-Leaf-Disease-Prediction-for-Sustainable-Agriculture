import numpy as np
import cv2
from tensorflow.keras.applications import VGG16, ResNet50, MobileNetV2
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.models import Model

class FeatureExtractor:
    """
    Class for extracting features from leaf images using various methods:
    1. Traditional computer vision features
    2. Deep learning features using pre-trained models
    """
    
    def __init__(self, method='deep', model_name='vgg16'):
        """
        Initialize the feature extractor.
        
        Args:
            method (str): Feature extraction method ('traditional' or 'deep')
            model_name (str): Pre-trained model name for deep features
                             ('vgg16', 'resnet50', or 'mobilenet')
        """
        self.method = method
        self.model_name = model_name
        
        # Initialize deep learning model if needed
        if method == 'deep':
            self._init_deep_model(model_name)
    
    def _init_deep_model(self, model_name):
        """
        Initialize a pre-trained deep learning model for feature extraction.
        
        Args:
            model_name (str): Name of the pre-trained model
        """
        if model_name == 'vgg16':
            base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
            self.preprocess_func = vgg_preprocess
        elif model_name == 'resnet50':
            base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
            self.preprocess_func = resnet_preprocess
        elif model_name == 'mobilenet':
            base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
            self.preprocess_func = mobilenet_preprocess
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Use the output of the last convolutional layer as features
        self.model = Model(inputs=base_model.input, outputs=base_model.layers[-1].output)
    
    def extract_traditional_features(self, image):
        """
        Extract traditional computer vision features from an image.
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            numpy.ndarray: Feature vector
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        features = []
        
        # Color features (if image is color)
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Convert to HSV color space
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            
            # Calculate color histograms
            h_hist = cv2.calcHist([hsv], [0], None, [32], [0, 180])
            s_hist = cv2.calcHist([hsv], [1], None, [32], [0, 256])
            v_hist = cv2.calcHist([hsv], [2], None, [32], [0, 256])
            
            # Normalize histograms
            h_hist = cv2.normalize(h_hist, h_hist).flatten()
            s_hist = cv2.normalize(s_hist, s_hist).flatten()
            v_hist = cv2.normalize(v_hist, v_hist).flatten()
            
            features.extend(h_hist)
            features.extend(s_hist)
            features.extend(v_hist)
            
            # Color moments
            for i in range(3):
                features.append(np.mean(image[:,:,i]))  # Mean
                features.append(np.std(image[:,:,i]))   # Standard deviation
                features.append(np.cbrt(np.mean(np.power(image[:,:,i] - np.mean(image[:,:,i]), 3))))  # Skewness
        
        # Texture features
        # GLCM (Gray-Level Co-occurrence Matrix) features
        glcm = self._compute_glcm(gray)
        features.append(self._compute_contrast(glcm))
        features.append(self._compute_homogeneity(glcm))
        features.append(self._compute_energy(glcm))
        features.append(self._compute_correlation(glcm))
        
        # Shape features
        # Edge detection
        edges = cv2.Canny(gray, 100, 200)
        features.append(np.sum(edges) / (edges.shape[0] * edges.shape[1]))  # Edge density
        
        return np.array(features)
    
    def extract_deep_features(self, image):
        """
        Extract deep learning features from an image using a pre-trained model.
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            numpy.ndarray: Feature vector
        """
        # Ensure image has correct dimensions
        if image.shape[:2] != (224, 224):
            image = cv2.resize(image, (224, 224))
        
        # Ensure image has 3 channels
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Preprocess the image according to the model requirements
        x = self.preprocess_func(image.astype(np.float32))
        x = np.expand_dims(x, axis=0)
        
        # Extract features
        features = self.model.predict(x)
        
        # Flatten features to a 1D vector
        return features.flatten()
    
    def extract_features(self, image):
        """
        Extract features from an image based on the selected method.
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            numpy.ndarray: Feature vector
        """
        if self.method == 'traditional':
            return self.extract_traditional_features(image)
        elif self.method == 'deep':
            return self.extract_deep_features(image)
        else:
            raise ValueError(f"Unsupported method: {self.method}")
    
    def _compute_glcm(self, image, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
        """
        Compute Gray-Level Co-occurrence Matrix for texture analysis.
        
        Args:
            image (numpy.ndarray): Grayscale image
            distances (list): List of distances
            angles (list): List of angles
            
        Returns:
            numpy.ndarray: GLCM matrix
        """
        # Normalize image to 8 levels to reduce computation
        bins = 8
        img_normalized = np.floor(image / (256 / bins)).astype(np.uint8)
        
        # Compute GLCM
        glcm = np.zeros((bins, bins))
        for distance in distances:
            for angle in angles:
                dx = int(round(distance * np.cos(angle)))
                dy = int(round(distance * np.sin(angle)))
                
                # Create shifted image
                rows, cols = img_normalized.shape
                shifted_img = np.zeros_like(img_normalized)
                
                if dx >= 0:
                    x_start, x_end = 0, cols - dx
                    x_shift_start, x_shift_end = dx, cols
                else:
                    x_start, x_end = -dx, cols
                    x_shift_start, x_shift_end = 0, cols + dx
                
                if dy >= 0:
                    y_start, y_end = 0, rows - dy
                    y_shift_start, y_shift_end = dy, rows
                else:
                    y_start, y_end = -dy, rows
                    y_shift_start, y_shift_end = 0, rows + dy
                
                shifted_img[y_shift_start:y_shift_end, x_shift_start:x_shift_end] = \
                    img_normalized[y_start:y_end, x_start:x_end]
                
                # Compute co-occurrence matrix
                for i in range(bins):
                    for j in range(bins):
                        glcm[i, j] += np.sum((img_normalized == i) & (shifted_img == j))
        
        # Normalize GLCM
        if glcm.sum() > 0:
            glcm = glcm / glcm.sum()
        
        return glcm
    
    def _compute_contrast(self, glcm):
        """
        Compute contrast from GLCM.
        
        Args:
            glcm (numpy.ndarray): GLCM matrix
            
        Returns:
            float: Contrast value
        """
        rows, cols = glcm.shape
        contrast = 0
        for i in range(rows):
            for j in range(cols):
                contrast += glcm[i, j] * (i - j) ** 2
        return contrast
    
    def _compute_homogeneity(self, glcm):
        """
        Compute homogeneity from GLCM.
        
        Args:
            glcm (numpy.ndarray): GLCM matrix
            
        Returns:
            float: Homogeneity value
        """
        rows, cols = glcm.shape
        homogeneity = 0
        for i in range(rows):
            for j in range(cols):
                homogeneity += glcm[i, j] / (1 + (i - j) ** 2)
        return homogeneity
    
    def _compute_energy(self, glcm):
        """
        Compute energy from GLCM.
        
        Args:
            glcm (numpy.ndarray): GLCM matrix
            
        Returns:
            float: Energy value
        """
        return np.sum(glcm ** 2)
    
    def _compute_correlation(self, glcm):
        """
        Compute correlation from GLCM.
        
        Args:
            glcm (numpy.ndarray): GLCM matrix
            
        Returns:
            float: Correlation value
        """
        rows, cols = glcm.shape
        i_indices = np.arange(rows).reshape(-1, 1)
        j_indices = np.arange(cols).reshape(1, -1)
        
        # Calculate means and standard deviations
        mean_i = np.sum(i_indices * glcm)
        mean_j = np.sum(j_indices * glcm)
        std_i = np.sqrt(np.sum(((i_indices - mean_i) ** 2) * glcm))
        std_j = np.sqrt(np.sum(((j_indices - mean_j) ** 2) * glcm))
        
        # Calculate correlation
        correlation = 0
        if std_i > 0 and std_j > 0:
            for i in range(rows):
                for j in range(cols):
                    correlation += glcm[i, j] * ((i - mean_i) * (j - mean_j)) / (std_i * std_j)
        
        return correlation