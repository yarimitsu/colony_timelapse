"""
Colony Timelapse Analysis Package

This package provides tools for processing camera images from colony monitoring
systems and running YOLO11 classification analysis.
"""

from .pipeline import ColonyTimelapseProcessor
from .image_processor import ImageProcessor
from .yolo_classifier import YOLOClassifier

__version__ = "1.0.0"
__author__ = "Colony Timelapse Team"

__all__ = [
    'ColonyTimelapseProcessor',
    'ImageProcessor', 
    'YOLOClassifier'
] 