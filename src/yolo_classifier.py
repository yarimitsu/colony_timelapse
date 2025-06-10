"""
YOLO11 classification module for colony timelapse analysis.
Handles model loading and batch inference on processed images.
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

import pandas as pd
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
import torch

logger = logging.getLogger(__name__)


class YOLOClassifier:
    """Handles YOLO11 classification for processed images."""
    
    def __init__(self, config: Dict):
        """Initialize the YOLO classifier with configuration."""
        self.config = config
        self.model_path = config['paths']['yolo_weights']
        self.confidence_threshold = config['yolo']['confidence_threshold']
        self.device = config['yolo']['device']
        self.batch_size = config['yolo']['batch_size']
        self.top_predictions = config['output']['top_predictions']
        
        # Initialize model
        self.model = None
        self.class_names = None
        self._load_model()
        
    def _load_model(self):
        """Load the YOLO model from weights file."""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model weights not found: {self.model_path}")
            
            logger.info(f"Loading YOLO model from: {self.model_path}")
            self.model = YOLO(self.model_path)
            
            # Set device
            if self.device == 'auto':
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            else:
                device = self.device
            
            self.model.to(device)
            logger.info(f"Model loaded on device: {device}")
            
            # Get class names
            self.class_names = self.model.names
            logger.info(f"Model classes: {list(self.class_names.values())}")
            
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise
    
    def predict_single(self, image_path: Path) -> Dict:
        """Run prediction on a single image."""
        try:
            # Run inference
            results = self.model(str(image_path), verbose=False)
            
            if not results:
                return self._empty_prediction()
            
            result = results[0]
            
            # Extract predictions
            if hasattr(result, 'probs') and result.probs is not None:
                # Classification results
                probs = result.probs.data.cpu().numpy()
                
                # Get top predictions
                top_indices = np.argsort(probs)[::-1][:self.top_predictions]
                
                predictions = {}
                for i, idx in enumerate(top_indices):
                    class_name = self.class_names[idx]
                    confidence = float(probs[idx])
                    
                    # Only include predictions above threshold
                    if confidence >= self.confidence_threshold:
                        predictions[f'class_{i+1}'] = class_name
                        predictions[f'confidence_{i+1}'] = confidence
                    else:
                        predictions[f'class_{i+1}'] = None
                        predictions[f'confidence_{i+1}'] = 0.0
                
                # Add top prediction info
                if predictions.get('class_1'):
                    predictions['top_class'] = predictions['class_1']
                    predictions['top_confidence'] = predictions['confidence_1']
                else:
                    predictions['top_class'] = 'unknown'
                    predictions['top_confidence'] = 0.0
                
                return predictions
            else:
                logger.warning(f"No classification results for: {image_path}")
                return self._empty_prediction()
                
        except Exception as e:
            logger.error(f"Error predicting {image_path}: {e}")
            return self._empty_prediction()
    
    def _empty_prediction(self) -> Dict:
        """Return empty prediction structure."""
        predictions = {'top_class': 'error', 'top_confidence': 0.0}
        
        for i in range(1, self.top_predictions + 1):
            predictions[f'class_{i}'] = None
            predictions[f'confidence_{i}'] = 0.0
            
        return predictions
    
    def predict_batch(self, image_paths: List[Path]) -> List[Dict]:
        """Run predictions on a batch of images."""
        if not image_paths:
            return []
        
        try:
            # Convert paths to strings
            path_strings = [str(path) for path in image_paths]
            
            # Run batch inference
            results = self.model(path_strings, verbose=False)
            
            predictions = []
            for i, result in enumerate(results):
                try:
                    if hasattr(result, 'probs') and result.probs is not None:
                        # Extract classification results
                        probs = result.probs.data.cpu().numpy()
                        
                        # Get top predictions
                        top_indices = np.argsort(probs)[::-1][:self.top_predictions]
                        
                        pred_dict = {}
                        for j, idx in enumerate(top_indices):
                            class_name = self.class_names[idx]
                            confidence = float(probs[idx])
                            
                            if confidence >= self.confidence_threshold:
                                pred_dict[f'class_{j+1}'] = class_name
                                pred_dict[f'confidence_{j+1}'] = confidence
                            else:
                                pred_dict[f'class_{j+1}'] = None
                                pred_dict[f'confidence_{j+1}'] = 0.0
                        
                        # Add top prediction info
                        if pred_dict.get('class_1'):
                            pred_dict['top_class'] = pred_dict['class_1']
                            pred_dict['top_confidence'] = pred_dict['confidence_1']
                        else:
                            pred_dict['top_class'] = 'unknown'
                            pred_dict['top_confidence'] = 0.0
                        
                        predictions.append(pred_dict)
                    else:
                        predictions.append(self._empty_prediction())
                        
                except Exception as e:
                    logger.error(f"Error processing result {i}: {e}")
                    predictions.append(self._empty_prediction())
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error in batch prediction: {e}")
            return [self._empty_prediction() for _ in image_paths]
    
    def classify_images(self, processed_df: pd.DataFrame) -> pd.DataFrame:
        """Run classification on all processed images."""
        if processed_df.empty:
            logger.warning("No processed images to classify")
            return processed_df
        
        logger.info(f"Running YOLO classification on {len(processed_df)} images")
        
        # Get image paths
        image_paths = [Path(path) for path in processed_df['processed_path']]
        
        # Filter existing images
        existing_paths = []
        existing_indices = []
        for i, path in enumerate(image_paths):
            if path.exists():
                existing_paths.append(path)
                existing_indices.append(i)
            else:
                logger.warning(f"Processed image not found: {path}")
        
        if not existing_paths:
            logger.error("No existing processed images found for classification")
            return processed_df
        
        # Run predictions in batches
        all_predictions = []
        
        for i in tqdm(range(0, len(existing_paths), self.batch_size), 
                     desc="Classifying images"):
            batch_paths = existing_paths[i:i + self.batch_size]
            batch_predictions = self.predict_batch(batch_paths)
            all_predictions.extend(batch_predictions)
        
        # Create results dataframe
        results_df = processed_df.copy()
        
        # Initialize prediction columns
        pred_columns = ['top_class', 'top_confidence']
        for i in range(1, self.top_predictions + 1):
            pred_columns.extend([f'class_{i}', f'confidence_{i}'])
        
        for col in pred_columns:
            results_df[col] = None
        
        # Fill in predictions
        for idx, pred in zip(existing_indices, all_predictions):
            for col in pred_columns:
                results_df.at[idx, col] = pred.get(col)
        
        logger.info(f"Classification completed. Found {len(all_predictions)} valid results")
        
        return results_df
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        if self.model is None:
            return {}
        
        return {
            'model_path': self.model_path,
            'device': str(self.model.device),
            'class_names': list(self.class_names.values()),
            'num_classes': len(self.class_names),
            'confidence_threshold': self.confidence_threshold,
            'batch_size': self.batch_size
        } 