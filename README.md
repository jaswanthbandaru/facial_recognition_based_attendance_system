# Face Recognition Attendance System

A comprehensive attendance management system using facial recognition technology built with Python, OpenCV, and Flask. This system provides both a command-line interface and a web-based interface for managing student attendance through real-time face recognition.

## Features

### Core Functionality
- **Real-time Face Recognition**: Uses advanced face recognition algorithms to identify students
- **Automated Attendance Marking**: Automatically marks attendance when a recognized face is detected
- **Duplicate Prevention**: Prevents marking attendance multiple times for the same student on the same day
- **Student Management**: Add, view, and manage student records with their photos
- **Attendance Reports**: Generate detailed attendance reports for specific dates or all records

### Interfaces
- **Command Line Interface**: Terminal-based system for basic operations
- **Web Interface**: Modern web application with live video streaming and interactive dashboard
- **Live Camera Feed**: Real-time video streaming with face detection overlay
- **Responsive Design**: Works on desktop and mobile devices

## Technology Stack

- **Backend**: Python 3.7+, Flask
- **Computer Vision**: OpenCV, face_recognition library
- **Data Storage**: JSON (student database), CSV (attendance records), Pickle (face encodings)
- **Frontend**: HTML, CSS, JavaScript, Bootstrap
- **Image Processing**: PIL/Pillow, NumPy

## Installation

### Prerequisites
- Python 3.7 or higher
- Webcam or camera device
- Good lighting conditions for optimal face recognition

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/face-recognition-attendance-system.git
cd face-recognition-attendance-system
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Create Required Directories
```bash
mkdir student_images
mkdir attendance_data
```

## Usage

### Command Line Interface

Run the main attendance system:
```bash
python attendance_system.py
```

**Available Options:**
1. **Add New Student**: Register a new student with their photo
2. **Start Attendance Session**: Begin live face recognition for attendance
3. **Generate Attendance Report**: View attendance records for specific dates
4. **List All Students**: Display all registered students
5. **Test Camera**: Check if camera is working properly

### Web Interface

Start the Flask web application:
```bash
python web_app.py
```

Visit `http://localhost:5000` in your browser.

**Web Features:**
- **Dashboard**: Overview of system status and quick access to features
- **Student Management**: Add new students and view existing records
- **Live Attendance**: Real-time face recognition with video feed
- **Attendance Reports**: View and filter attendance records
- **Camera Controls**: Start/stop camera and attendance sessions

## Configuration

### Camera Settings
The system automatically configures camera settings for optimal performance:
- Resolution: 640x480 (adjustable)
- Frame rate: 30 FPS
- Processing: Every other frame for better performance

### Face Recognition Parameters
- **Tolerance**: 0.6 (lower = more strict)
- **Model**: HOG (faster) or CNN (more accurate)
- **Jitters**: 1 (for encoding stability)

### File Storage
- **Student Database**: JSON format for easy reading/writing
- **Attendance Records**: CSV format for easy analysis
- **Face Encodings**: Pickle format for efficient storage

## API Endpoints (Web Interface)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Dashboard home page |
| GET | `/students` | View all students |
| GET/POST | `/add_student` | Add new student |
| GET | `/attendance` | View attendance records |
| GET | `/live_attendance` | Live attendance page |
| GET | `/start_camera` | Start camera streaming |
| GET | `/stop_camera` | Stop camera streaming |
| GET | `/start_attendance` | Start attendance session |
| GET | `/stop_attendance` | Stop attendance session |
| GET | `/video_feed` | Video streaming endpoint |
| GET | `/attendance_status` | Get system status |

## Database Schema

### Student Database (JSON)
```json
{
  "student_name": {
    "student_id": "unique_id",
    "image_path": "path/to/image.jpg",
    "added_date": "2024-01-01T12:00:00"
  }
}
```

### Attendance Records (CSV)
```csv
Name,Student_ID,Date,Time,Status
John Doe,12345,2024-01-01,09:30:15,Present
```

## Performance Optimization

### Face Recognition
- Processes every other frame for better performance
- Resizes frames to 1/4 size for faster processing
- Uses HOG model for real-time recognition
- Configurable tolerance for accuracy vs speed

### Web Application
- Threaded Flask application for concurrent requests
- Efficient video streaming with frame buffering
- Minimal CPU usage during idle periods

## Troubleshooting

### Common Issues

**Camera Not Working:**
- Check if camera is connected and not used by other applications
- Try different camera indices (0, 1, 2...)
- Ensure proper permissions for camera access

**Face Not Recognized:**
- Ensure good lighting conditions
- Check image quality when adding student
- Adjust tolerance settings if needed
- Verify face is clearly visible and not obscured

**Installation Issues:**
- Update pip: `pip install --upgrade pip`
- Install cmake: `pip install cmake`
- For dlib issues on Windows: Install Visual Studio Build Tools

**Memory Issues:**
- Reduce camera resolution in settings
- Process fewer frames per second
- Clear face encodings cache periodically

### Performance Tips
- Use good lighting for better recognition accuracy
- Keep student photos clear and well-lit
- Regularly clean camera lens
- Close unnecessary applications when running

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit a pull request

