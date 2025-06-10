# Colony Timelapse Image Analysis Pipeline

This project processes time lapse camera images, resizes them, and runs YOLO11 image classification to analyze temporal patterns in Common Murre colony attendance.

## Features

- **Batch Image Processing**: Resize images to 224x224px with EXIF preservation
- **YOLO11 Classification**: Run inference using trained weights  
- **Resume Functionality**: Skip already processed images for efficiency
- **Multi-Year Support**: Process and visualize data across multiple years
- **Automated Visualization**: Generate attendance pattern plots
- **Robust Pipeline**: End-to-end processing from raw images to final plots

## Project Structure

```
colony_timelapse/
├── scripts/
│   ├── process_year.py         # Main processing script
│   └── make_plot.py           # Generate attendance plots
├── src/
│   ├── pipeline.py            # Main pipeline orchestrator
│   ├── image_processor.py     # Image resizing and preprocessing
│   └── yolo_classifier.py     # YOLO11 classification
├── config/
│   └── config.yaml            # Configuration settings
├── models/
│   └── yolo_weights.pt        # Pre-trained YOLO11 weights
├── requirements.txt           # Python dependencies
├── setup.py                  # Package installation
└── README.md                 # This file
```

## Quick Start

**📋 Requirements:**
- Python 3.8+ (download from [python.org](https://www.python.org/downloads/))
- At least 20GB free disk space for processed images
- CUDA GPU recommended for faster processing

**🚀 Installation:**
1. Clone or download this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Process a Year of Data
```bash
python scripts/process_year.py --year YourDataset2024 --source "path/to/your/images"
```

### Generate Attendance Plot
```bash
python scripts/make_plot.py
```

## Configuration

Key settings in `config/config.yaml`:
- **Image size**: 224x224 pixels (optimized for YOLO11)
- **File patterns**: Automatically detects MLB_*.JPG or RCNX*.JPG
- **Output directories**: `processed_images/` and `results/`
- **YOLO settings**: Batch size, confidence threshold, device selection

## Resume Functionality

The pipeline automatically resumes interrupted processing:
- **Image Processing**: Skips already resized images (newly implemented)
- **Classification**: Loads existing results and processes only new images  
- **Efficiency**: Perfect for large datasets (50,000+ images)
- **Logging**: Shows exactly how many images are skipped vs. newly processed

## Output

The pipeline produces:
- **Processed Images**: Resized (224x224) in `processed_images/[Year]/`
- **Results CSV**: Classifications with metadata in `results/`
- **Attendance Plot**: Visualization in `figs/colony_attendance.png`
- **Processing Logs**: Detailed logs in `colony_timelapse.log`

## Model Information

**✅ Pre-trained Model Included!**
- YOLO11 classification model: `models/yolo_weights.pt`
- Classes: `Zero`, `Few_Half` (<50%), `Many_All` (>50%)
- Input: 224x224 pixel images
- Optimized for Common Murre colony analysis

### Model Training Details

The included YOLO11 weights were trained specifically on **Gull Island Common Murre colony data**:

- **Training Dataset**: 6,054 hand-labeled images from timelapse cameras
- **Class Distribution**:
  - `Zero`: 4,197 images (3,357 train, 840 validation)
  - `Few_Half`: 648 images (518 train, 130 validation) 
  - `Many_All`: 1,209 images (967 train, 242 validation)
- **Model Performance**:
  - Overall Accuracy: **99.67%**
  - `Few_Half` Class: Precision 99.22%, Recall 97.69%, F1-Score 98.45%
  - `Many_All` Class: Precision 99.59%, Recall 99.59%, F1-Score 99.59%
  - `Zero` Class: Precision 99.76%, Recall 100.00%, F1-Score 99.88%

The model was trained using Timelapse software for image annotation and achieves excellent performance across all attendance categories.

## File Patterns

Automatically detects and processes:
- **MLB_*.JPG** (older camera format)
- **RCNX*.JPG** (newer camera format)  
- Supports numbered camera folders (e.g., `100RECNX`, `101RECNX`, etc.)
- Automatically ignores backup and system directories

## Example Workflow

1. **Process your data:**
   ```bash
   python scripts/process_year.py --year MyColony2024 --source "/path/to/camera/images"
   ```

2. **Generate updated plot:**
   ```bash
   python scripts/make_plot.py
   ```

3. **Results:** Check `figs/colony_attendance.png` for attendance patterns across all years 