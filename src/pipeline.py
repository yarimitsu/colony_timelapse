"""
Main pipeline for colony timelapse analysis.
Orchestrates image processing, YOLO classification, and data analysis.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Optional
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler

import pandas as pd
import yaml

from image_processor import ImageProcessor
from yolo_classifier import YOLOClassifier

# Setup logging with rotation
log_file = 'colony_timelapse.log'
max_bytes = 10 * 1024 * 1024  # 10MB
backup_count = 5  # Keep 5 backup files

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class ColonyTimelapseProcessor:
    """Main processor for colony timelapse analysis pipeline."""
    
    def __init__(self, config_path: str = 'config/config.yaml'):
        """Initialize the processor with configuration."""
        self.config_path = config_path
        self.config = self._load_config()
        
        # Initialize components
        self.image_processor = ImageProcessor(self.config)
        self.yolo_classifier = YOLOClassifier(self.config)
        
        # Create results directory
        self.results_dir = Path(self.config['paths']['results_output'])
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Colony Timelapse Processor initialized")
    
    def _load_config(self) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from: {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    def process_year(self, year_name: str, source_dir: Optional[str] = None) -> pd.DataFrame:
        """Process a complete year of data."""
        logger.info(f"Starting processing for year: {year_name}")
        
        # Get source directory
        if source_dir is None:
            source_dir = input(f"Enter source directory for {year_name}: ").strip()
        
        if not source_dir or not os.path.exists(source_dir):
            raise ValueError(f"Invalid source directory: {source_dir}")
        
        try:
            # Extract location from year_name (assuming format like "Marker2023")
            location = ''.join(filter(str.isalpha, year_name))  # Get only letters
            year = ''.join(filter(str.isdigit, year_name))      # Get only numbers
            
            # Check for existing results to enable resume functionality
            existing_results = self._load_existing_results(year_name)
            
            # Step 1: Process images (resize and rename)
            logger.info("Step 1: Processing images...")
            processed_df = self.image_processor.process_images(source_dir, year_name)
            
            if processed_df.empty:
                logger.warning(f"No images processed for {year_name}")
                return processed_df
            
            # Step 2: Filter out already classified images
            if existing_results is not None and not existing_results.empty:
                logger.info(f"Found existing results with {len(existing_results)} images")
                processed_df = self._filter_unprocessed_images(processed_df, existing_results)
                logger.info(f"After filtering, {len(processed_df)} new images need classification")
            
            # Step 3: Run YOLO classification on new images only
            if not processed_df.empty:
                logger.info("Step 2: Running YOLO classification...")
                new_results_df = self.yolo_classifier.classify_images(processed_df)
                
                # Ensure location and year columns are preserved
                if 'location' not in new_results_df.columns:
                    new_results_df['location'] = location
                if 'year' not in new_results_df.columns:
                    new_results_df['year'] = year
                if 'dataset' not in new_results_df.columns:
                    new_results_df['dataset'] = year_name
                
                # Merge with existing results
                if existing_results is not None and not existing_results.empty:
                    logger.info("Merging new results with existing results...")
                    results_df = pd.concat([existing_results, new_results_df], ignore_index=True)
                    results_df = results_df.sort_values('datetime_taken').reset_index(drop=True)
                else:
                    results_df = new_results_df
            else:
                logger.info("All images already processed. Using existing results.")
                results_df = existing_results if existing_results is not None else pd.DataFrame()
            
            # Step 3: Save results
            logger.info("Step 3: Saving results...")
            self._save_results(results_df, year_name)
            
            # Step 4: Generate summary (optional, for logging)
            self._generate_summary(results_df, year_name)
            
            return results_df
            
        except Exception as e:
            logger.error(f"Error processing year {year_name}: {e}")
            raise
    
    def _load_existing_results(self, year_name: str) -> Optional[pd.DataFrame]:
        """Load existing results for a year if they exist."""
        try:
            # Look for existing results files for this year
            pattern = f"{year_name}_results_*.csv"
            existing_files = list(self.results_dir.glob(pattern))
            
            if not existing_files:
                logger.info(f"No existing results found for {year_name}")
                return None
            
            # Load the most recent results file
            latest_file = max(existing_files, key=lambda f: f.stat().st_mtime)
            logger.info(f"Loading existing results from: {latest_file}")
            
            df = pd.read_csv(latest_file)
            df['datetime_taken'] = pd.to_datetime(df['datetime_taken'])
            
            logger.info(f"Loaded {len(df)} existing results")
            return df
            
        except Exception as e:
            logger.warning(f"Error loading existing results: {e}")
            return None
    
    def _filter_unprocessed_images(self, processed_df: pd.DataFrame, existing_results: pd.DataFrame) -> pd.DataFrame:
        """Filter out images that have already been classified."""
        if existing_results.empty:
            return processed_df
        
        # Create set of already processed original paths for fast lookup
        processed_paths = set(existing_results['original_path'])
        
        # Filter to only include images not yet classified
        unprocessed_mask = ~processed_df['original_path'].isin(processed_paths)
        filtered_df = processed_df[unprocessed_mask].copy()
        
        skipped_count = len(processed_df) - len(filtered_df)
        if skipped_count > 0:
            logger.info(f"Skipping {skipped_count} already classified images")
        
        return filtered_df
    
    def _save_results(self, results_df: pd.DataFrame, year_name: str):
        """Save results to CSV and generate reports."""
        if results_df.empty:
            logger.warning("No results to save")
            return
        
        # Save main results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"{year_name}_results_{timestamp}.csv"
        csv_path = self.results_dir / csv_filename
        
        results_df.to_csv(csv_path, index=False)
        logger.info(f"Results saved to: {csv_path}")
        
        # Save summary statistics
        summary_path = self.results_dir / f"{year_name}_summary_{timestamp}.txt"
        with open(summary_path, 'w') as f:
            f.write(f"Colony Timelapse Analysis Summary - {year_name}\n")
            f.write("=" * 50 + "\n\n")
            
            # Basic statistics
            f.write(f"Total images processed: {len(results_df)}\n")
            f.write(f"Date range: {results_df['datetime_taken'].min()} to {results_df['datetime_taken'].max()}\n")
            f.write(f"Source directory: {results_df['original_path'].iloc[0] if not results_df.empty else 'N/A'}\n\n")
            
            # Classification results
            if 'top_class' in results_df.columns:
                class_counts = results_df['top_class'].value_counts()
                f.write("Classification Results:\n")
                f.write("-" * 20 + "\n")
                for class_name, count in class_counts.items():
                    percentage = (count / len(results_df)) * 100
                    f.write(f"{class_name}: {count} ({percentage:.1f}%)\n")
                f.write("\n")
            
            # Model information
            model_info = self.yolo_classifier.get_model_info()
            f.write("Model Information:\n")
            f.write("-" * 20 + "\n")
            for key, value in model_info.items():
                f.write(f"{key}: {value}\n")
        
        logger.info(f"Summary saved to: {summary_path}")
    
    def _generate_summary(self, results_df: pd.DataFrame, year_name: str) -> Dict:
        """Generate processing summary."""
        if results_df.empty:
            return {'year': year_name, 'total_images': 0, 'error': 'No results'}
        
        summary = {
            'year': year_name,
            'total_images': len(results_df),
            'date_range': {
                'start': results_df['datetime_taken'].min(),
                'end': results_df['datetime_taken'].max()
            },
            'unique_folders': results_df['parent_folder'].nunique(),
            'total_file_size_mb': results_df['file_size_original'].sum() / (1024 * 1024),
        }
        
        # Classification summary
        if 'top_class' in results_df.columns:
            summary['classification'] = {
                'unique_classes': results_df['top_class'].nunique(),
                'most_common_class': results_df['top_class'].mode().iloc[0] if not results_df['top_class'].mode().empty else 'N/A',
                'average_confidence': results_df['top_confidence'].mean()
            }
        
        logger.info(f"Processing summary for {year_name}: {summary}")
        return summary
    
    def batch_process_years(self, years_config: Dict[str, str]) -> Dict[str, pd.DataFrame]:
        """Process multiple years in batch."""
        results = {}
        
        for year_name, source_dir in years_config.items():
            try:
                logger.info(f"Processing year: {year_name}")
                results[year_name] = self.process_year(year_name, source_dir)
            except Exception as e:
                logger.error(f"Failed to process {year_name}: {e}")
                results[year_name] = pd.DataFrame()
        
        return results
    
    def get_processing_stats(self) -> Dict:
        """Get statistics about processed data."""
        stats = {
            'processed_images_dir': str(self.image_processor.processed_dir),
            'results_dir': str(self.results_dir),
            'config_path': self.config_path,
            'yolo_model_info': self.yolo_classifier.get_model_info()
        }
        
        # Count processed images
        if self.image_processor.processed_dir.exists():
            processed_count = sum(1 for _ in self.image_processor.processed_dir.rglob('*.jpg'))
            stats['total_processed_images'] = processed_count
        
        return stats


def main():
    """Main entry point for command line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Colony Timelapse Analysis Pipeline')
    parser.add_argument('--year', required=True, help='Year name for processing')
    parser.add_argument('--source', help='Source directory path')
    parser.add_argument('--config', default='config/config.yaml', help='Configuration file path')
    
    args = parser.parse_args()
    
    try:
        processor = ColonyTimelapseProcessor(args.config)
        results_df = processor.process_year(args.year, args.source)
        
        print(f"\nProcessing completed for {args.year}")
        print(f"Total images processed: {len(results_df)}")
        
        if not results_df.empty and 'top_class' in results_df.columns:
            print("\nTop classifications:")
            print(results_df['top_class'].value_counts().head())
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 