# Face Recognition Attendance System

A professional face recognition-based attendance management system built with Python, OpenCV, and Streamlit.

## Features

- **Student Registration**: Register students with ID, name, and contact details
- **Face Capture**: Capture multiple face images for training
- **Model Training**: Train recognition models using Dlib (high accuracy) and LBPH (lightweight)
- **Real-time Recognition**: Mark attendance through live face recognition
- **Attendance Reports**: View and export reports (Excel, CSV, PDF)
- **Liveness Detection**: Blink and movement detection to prevent spoofing
- **Multi-face Detection**: Recognize multiple faces simultaneously
- **Duplicate Prevention**: Prevents marking attendance more than once per day

## Project Structure

```
Face Recognition Attendance System/
├── app.py                      # Main application entry point
├── requirements.txt            # Python dependencies
├── config/
│   ├── __init__.py
│   └── settings.py            # Configuration settings
├── database/
│   ├── __init__.py
│   ├── models.py              # Database models (SQLite)
│   └── operations.py          # CRUD operations
├── utils/
│   ├── __init__.py
│   ├── face_detector.py       # Face detection module
│   ├── face_recognizer.py     # Face recognition (Dlib + LBPH)
│   ├── camera.py              # Camera management
│   ├── helpers.py             # Utility functions
│   └── export.py              # Export to Excel/PDF
├── pages/
│   ├── 1_Student_Registration.py
│   ├── 2_Face_Capture.py
│   ├── 3_Model_Training.py
│   ├── 4_Mark_Attendance.py
│   └── 5_Attendance_Reports.py
├── dataset/                    # Face images storage
├── trained_models/             # Trained recognition models
├── exports/                    # Exported reports
└── logs/                       # Application logs
```

## Installation

1. **Clone or download the project**

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate     # Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   Note: Installing `dlib` may require additional steps:
   - **Linux**: `sudo apt-get install cmake libboost-all-dev`
   - **Windows**: Install Visual Studio Build Tools
   - **Mac**: `brew install cmake boost`

4. **Run the application**:
   ```bash
   streamlit run app.py
   ```

## Usage

### 1. Register Students
- Navigate to "Student Registration"
- Fill in student details (ID, name, email, etc.)
- Submit the form

### 2. Capture Face Images
- Go to "Face Capture"
- Select a registered student
- Click "Start Capture" to capture face images
- Move your head slowly to capture different angles
- Minimum 10 images required per student

### 3. Train the Model
- Navigate to "Model Training"
- Select recognition models (Dlib and/or LBPH)
- Click "Start Training"
- Wait for training to complete

### 4. Mark Attendance
- Go to "Mark Attendance"
- Click "Start Recognition"
- Face the camera
- Attendance is marked automatically when recognized

### 5. View Reports
- Navigate to "Reports"
- Select date range and filters
- Export to Excel, CSV, or PDF

## Configuration

Edit `config/settings.py` to customize:

- **Face Recognition Settings**: Tolerance, model type, encoding parameters
- **Camera Settings**: Resolution, FPS, camera index
- **Capture Settings**: Number of images, interval, image size
- **Attendance Settings**: Duplicate check period, confidence threshold
- **Liveness Settings**: Enable/disable, blink threshold

## Database

The system uses SQLite for data storage with the following tables:
- `students`: Student registration data
- `attendance`: Attendance records
- `training_logs`: Model training history
- `system_logs`: Activity logs

## Recognition Models

### Dlib (Recommended)
- Uses deep learning face embeddings
- Higher accuracy
- Requires more computational resources

### LBPH (Local Binary Patterns Histogram)
- Lightweight, works offline
- Faster processing
- Good for resource-constrained systems

## Security Features

- **Liveness Detection**: Detects blinks and facial movement to prevent photo spoofing
- **Confidence Scoring**: Only marks attendance above confidence threshold
- **Duplicate Prevention**: One attendance record per student per day
- **Unknown Face Handling**: Displays "Unknown" and doesn't mark attendance

## Troubleshooting

### Camera not working
- Check camera permissions
- Try different camera index in settings
- Ensure no other application is using the camera

### Face not detected
- Improve lighting conditions
- Face the camera directly
- Remove obstructions (glasses, masks)

### Low recognition accuracy
- Capture more face images (50+ recommended)
- Capture in varied lighting conditions
- Retrain the model after adding new images

### Installation issues with dlib
```bash
# Ubuntu/Debian
sudo apt-get install build-essential cmake
sudo apt-get install libgtk-3-dev libboost-all-dev

# Then install dlib
pip install dlib
```

## License

This project is for educational purposes.

## Support

For issues or questions, please check the logs in the `logs/` directory.
