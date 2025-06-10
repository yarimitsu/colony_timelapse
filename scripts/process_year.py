#!/usr/bin/env python3
"""
Standalone script to process a single year of colony timelapse data.
Can be run from command line or imported as a module.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from pipeline import ColonyTimelapseProcessor


def main():
    """Main function for processing a year of data."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Process colony timelapse data for a specific year'
    )
    parser.add_argument(
        '--year', 
        required=True, 
        help='Year identifier (e.g., "Marker2023")'
    )
    parser.add_argument(
        '--source', 
        help='Source directory path (if not provided, will be prompted)'
    )
    parser.add_argument(
        '--config', 
        default='config/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set up logging level
    if args.verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        print(f"Colony Timelapse Processor - Processing year: {args.year}")
        print("=" * 60)
        
        # Initialize processor
        processor = ColonyTimelapseProcessor(args.config)
        
        # Process the year
        results_df = processor.process_year(args.year, args.source)
        
        # Display results summary
        if not results_df.empty:
            print(f"\n✅ Processing completed successfully!")
            print(f"📊 Total images processed: {len(results_df)}")
            
            if 'datetime_taken' in results_df.columns:
                date_range = f"{results_df['datetime_taken'].min()} to {results_df['datetime_taken'].max()}"
                print(f"📅 Date range: {date_range}")
            
            if 'top_class' in results_df.columns:
                print(f"\n🏷️  Top classifications:")
                class_counts = results_df['top_class'].value_counts().head()
                for class_name, count in class_counts.items():
                    percentage = (count / len(results_df)) * 100
                    print(f"   {class_name}: {count} ({percentage:.1f}%)")
            
            print(f"\n💾 Results saved to: {processor.results_dir}")
            
        else:
            print(f"⚠️  No images were processed for {args.year}")
            
    except KeyboardInterrupt:
        print(f"\n❌ Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 