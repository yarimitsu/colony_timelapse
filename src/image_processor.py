"""
Image processing module for colony timelapse analysis.
Handles image resizing, renaming, and metadata extraction.
"""

import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import logging
from datetime import datetime

from PIL import Image, ExifTags
import pandas as pd
from tqdm import tqdm
import exifread
import piexif

logger = logging.getLogger(__name__)


class ImageProcessor:
    """Handles image processing operations for colony timelapse data."""
    
    def __init__(self, config: Dict):
        """Initialize the image processor with configuration."""
        self.config = config
        self.target_size = config['image_processing']['target_size']
        self.output_format = config['image_processing']['output_format']
        self.jpg_quality = config['image_processing']['jpg_quality']
        self.processed_dir = Path(config['paths']['processed_images'])
        
        # Create output directory if it doesn't exist
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
    def _detect_file_pattern(self, source_dir: str) -> str:
        """Detect the appropriate file pattern based on directory contents."""
        source_path = Path(source_dir)
        if not source_path.exists():
            raise FileNotFoundError(f"Source directory not found: {source_dir}")
            
        # Check for MLB pattern first
        mlb_pattern = "MLB_*.JPG"
        mlb_files = list(source_path.rglob(mlb_pattern))
        
        # If no MLB files found, check for RCNX pattern
        if not mlb_files:
            rcnx_pattern = "RCNX*.JPG"
            rcnx_files = list(source_path.rglob(rcnx_pattern))
            if rcnx_files:
                logger.info(f"Using RCNX file pattern: {rcnx_pattern}")
                return rcnx_pattern
            else:
                # If neither pattern is found, try any JPG files
                jpg_pattern = "*.JPG"
                jpg_files = list(source_path.rglob(jpg_pattern))
                if jpg_files:
                    logger.warning("No MLB_ or RCNX files found. Using generic JPG pattern.")
                    return jpg_pattern
                else:
                    raise ValueError(f"No supported image files found in {source_dir}")
        else:
            logger.info(f"Using MLB file pattern: {mlb_pattern}")
            return mlb_pattern
    
    def find_images(self, source_dir: str) -> List[Path]:
        """Find all images matching the detected pattern in source directory."""
        source_path = Path(source_dir)
        if not source_path.exists():
            raise FileNotFoundError(f"Source directory not found: {source_dir}")
        
        # Detect the appropriate file pattern
        file_pattern = self._detect_file_pattern(source_dir)
        
        images = []
        # Search recursively for images matching pattern
        for image_path in source_path.rglob(file_pattern):
            if image_path.is_file():
                images.append(image_path)
        
        logger.info(f"Found {len(images)} images matching pattern '{file_pattern}'")
        return sorted(images)
    
    def extract_datetime(self, image_path: Path) -> Optional[datetime]:
        """Extract datetime from image EXIF data."""
        try:
            with open(image_path, 'rb') as f:
                tags = exifread.process_file(f, stop_tag="EXIF DateTimeOriginal")
                
            if 'EXIF DateTimeOriginal' in tags:
                date_str = str(tags['EXIF DateTimeOriginal'])
                return datetime.strptime(date_str, '%Y:%m:%d %H:%M:%S')
            else:
                # Fallback to file modification time
                return datetime.fromtimestamp(image_path.stat().st_mtime)
                
        except Exception as e:
            logger.warning(f"Could not extract datetime from {image_path}: {e}")
            # Fallback to file modification time
            return datetime.fromtimestamp(image_path.stat().st_mtime)
    
    def generate_new_filename(self, original_path: Path) -> str:
        """Generate new filename with parent folder prefix."""
        parent_folder = original_path.parent.name
        original_name = original_path.stem  # filename without extension
        
        # Use template from config
        template = self.config['naming']['processed_name_template']
        new_name = template.format(
            parent_folder=parent_folder,
            original_name=original_name
        )
        
        return f"{new_name}.{self.output_format}"
    
    def _copy_exif(self, source_image: Image.Image, target_image: Image.Image) -> Image.Image:
        """Copy EXIF data from source to target image."""
        try:
            # Get EXIF data from source
            if 'exif' in source_image.info:
                # If using JPEG, preserve EXIF directly
                if self.output_format.upper() in ['JPEG', 'JPG']:
                    target_image.info['exif'] = source_image.info['exif']
                else:
                    # For other formats, use piexif to convert EXIF data
                    exif_dict = piexif.load(source_image.info['exif'])
                    exif_bytes = piexif.dump(exif_dict)
                    target_image.info['exif'] = exif_bytes
            return target_image
        except Exception as e:
            logger.warning(f"Could not copy EXIF data: {e}")
            return target_image
    
    def resize_image(self, image_path: Path, output_path: Path) -> bool:
        """Resize image to target size while maintaining aspect ratio."""
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Calculate resize dimensions maintaining aspect ratio
                original_size = img.size
                ratio = min(self.target_size / original_size[0], 
                           self.target_size / original_size[1])
                new_size = (int(original_size[0] * ratio), 
                           int(original_size[1] * ratio))
                
                # Resize image
                img_resized = img.resize(new_size, Image.Resampling.LANCZOS)
                
                # Create square image with padding if needed
                if new_size[0] != self.target_size or new_size[1] != self.target_size:
                    # Create new square image with white background
                    square_img = Image.new('RGB', (self.target_size, self.target_size), (255, 255, 255))
                    
                    # Center the resized image
                    offset = ((self.target_size - new_size[0]) // 2,
                             (self.target_size - new_size[1]) // 2)
                    square_img.paste(img_resized, offset)
                    img_resized = square_img
                
                # Copy EXIF data to resized image
                img_resized = self._copy_exif(img, img_resized)
                
                # Save the processed image
                save_kwargs = {'format': 'JPEG' if self.output_format.upper() == 'JPG' else self.output_format}
                if self.output_format.upper() in ['JPEG', 'JPG']:
                    save_kwargs['quality'] = self.jpg_quality
                    save_kwargs['optimize'] = True
                    save_kwargs['exif'] = img_resized.info.get('exif', b'')
                
                img_resized.save(output_path, **save_kwargs)
                return True
                
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            return False
    
    def process_images(self, source_dir: str, year_name: str) -> pd.DataFrame:
        """Process all images in source directory."""
        logger.info(f"Starting image processing for {year_name}")
        
        # Try to load existing image metadata to avoid redundant processing
        existing_metadata = self._load_existing_metadata(year_name)
        
        # Find all images
        images = self.find_images(source_dir)
        if not images:
            logger.warning(f"No images found in {source_dir}")
            return pd.DataFrame()
        
        # Create year-specific subdirectory
        year_output_dir = self.processed_dir / year_name
        year_output_dir.mkdir(exist_ok=True)
        
        # Filter images if we have existing metadata
        if existing_metadata is not None:
            existing_paths = set(existing_metadata['original_path'])
            new_images = [img for img in images if str(img) not in existing_paths]
            logger.info(f"Found {len(existing_metadata)} existing metadata entries")
            logger.info(f"Processing {len(new_images)} new images (skipping {len(images) - len(new_images)} with existing metadata)")
            images_to_process = new_images
            processed_data = existing_metadata.to_dict('records')
        else:
            logger.info(f"No existing metadata found, processing all {len(images)} images")
            images_to_process = images
            processed_data = []
        
        # Process only new images
        success_count = len(processed_data)  # Count existing as successful
        new_processed = 0
        save_interval = 1000  # Save metadata every 1000 images
        
        # Extract location from year_name (assuming format like "Marker2023")
        location = ''.join(filter(str.isalpha, year_name))  # Get only letters
        year = ''.join(filter(str.isdigit, year_name))      # Get only numbers
        
        if images_to_process:
            for i, image_path in enumerate(tqdm(images_to_process, desc="Processing new images")):
                try:
                    # Generate new filename
                    new_filename = self.generate_new_filename(image_path)
                    output_path = year_output_dir / new_filename
                    
                    # Check if already processed (double-check)
                    if output_path.exists():
                        logger.debug(f"Image already exists: {new_filename}")
                        # Extract datetime and add to data
                        image_datetime = self.extract_datetime(image_path)
                        processed_data.append({
                            'original_path': str(image_path),
                            'processed_path': str(output_path),
                            'processed_filename': new_filename,
                            'parent_folder': image_path.parent.name,
                            'datetime_taken': image_datetime,
                            'file_size_original': image_path.stat().st_size,
                            'file_size_processed': output_path.stat().st_size,
                            'location': location,
                            'year': year,
                            'dataset': year_name
                        })
                        success_count += 1
                        continue
                    
                    # Process new image
                    if self.resize_image(image_path, output_path):
                        success_count += 1
                        new_processed += 1
                    else:
                        logger.error(f"Failed to process: {image_path}")
                        continue
                    
                    # Extract metadata for new images
                    image_datetime = self.extract_datetime(image_path)
                    
                    # Store processing information
                    processed_data.append({
                        'original_path': str(image_path),
                        'processed_path': str(output_path),
                        'processed_filename': new_filename,
                        'parent_folder': image_path.parent.name,
                        'datetime_taken': image_datetime,
                        'file_size_original': image_path.stat().st_size,
                        'file_size_processed': output_path.stat().st_size if output_path.exists() else None,
                        'location': location,
                        'year': year,
                        'dataset': year_name
                    })
                    
                    # Save metadata periodically to avoid losing progress
                    if (i + 1) % save_interval == 0:
                        df_temp = pd.DataFrame(processed_data)
                        if not df_temp.empty:
                            df_temp['datetime_taken'] = pd.to_datetime(df_temp['datetime_taken'])
                            self._save_metadata(df_temp, year_name)
                            logger.info(f"Saved progress: {len(processed_data)} images processed")
                    
                except Exception as e:
                    logger.error(f"Error processing {image_path}: {e}")
                    continue
        
        logger.info(f"Successfully processed {success_count} total images ({new_processed} newly processed)")
        
        # Create DataFrame
        df = pd.DataFrame(processed_data)
        if not df.empty:
            df['datetime_taken'] = pd.to_datetime(df['datetime_taken'])
            df = df.sort_values('datetime_taken').reset_index(drop=True)
            
            # Save metadata for future resume operations
            self._save_metadata(df, year_name)
        
        return df
    
    def _load_existing_metadata(self, year_name: str) -> Optional[pd.DataFrame]:
        """Load existing image processing metadata if available."""
        try:
            metadata_path = self.processed_dir / f"{year_name}_metadata.csv"
            if metadata_path.exists():
                logger.info(f"Loading existing image metadata from: {metadata_path}")
                df = pd.read_csv(metadata_path, low_memory=False)
                df['datetime_taken'] = pd.to_datetime(df['datetime_taken'])
                return df
            return None
        except Exception as e:
            logger.warning(f"Error loading existing metadata: {e}")
            return None
    
    def _save_metadata(self, df: pd.DataFrame, year_name: str):
        """Save image processing metadata for future resume operations."""
        try:
            metadata_path = self.processed_dir / f"{year_name}_metadata.csv"
            df.to_csv(metadata_path, index=False)
            logger.info(f"Saved image processing metadata to: {metadata_path}")
        except Exception as e:
            logger.warning(f"Error saving metadata: {e}") 