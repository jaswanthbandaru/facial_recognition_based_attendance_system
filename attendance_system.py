import cv2
import numpy as np
import face_recognition
import os
import csv
import pandas as pd
from datetime import datetime
import json
import pickle
from pathlib import Path

class FaceRecognitionAttendanceSystem:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.attendance_records = []
        self.attendance_file = "attendance_records.csv"
        self.encodings_file = "face_encodings.pkl"
        self.students_db = "students_database.json"
        
        # Create necessary directories
        Path("student_images").mkdir(exist_ok=True)
        Path("attendance_data").mkdir(exist_ok=True)
        
        # Load existing data
        self.load_student_database()
        self.load_face_encodings()
        self.setup_attendance_file()
    
    def load_student_database(self):
        """Load student database from JSON file"""
        try:
            with open(self.students_db, 'r') as f:
                self.students_data = json.load(f)
        except FileNotFoundError:
            self.students_data = {}
            print("No existing student database found. Starting fresh.")
    
    def save_student_database(self):
        """Save student database to JSON file"""
        with open(self.students_db, 'w') as f:
            json.dump(self.students_data, f, indent=2)
    
    def add_student(self, name, student_id, image_path):
        """Add a new student to the database"""
        try:
            # Load and encode the face
            image = face_recognition.load_image_file(image_path)
            face_encodings = face_recognition.face_encodings(image)
            
            if len(face_encodings) == 0:
                print(f"No face detected in {image_path}")
                return False
            
            # Store student data
            self.students_data[name] = {
                'student_id': student_id,
                'image_path': image_path,
                'added_date': datetime.now().isoformat()
            }
            
            # Store face encoding
            self.known_face_encodings.append(face_encodings[0])
            self.known_face_names.append(name)
            
            # Save to files
            self.save_student_database()
            self.save_face_encodings()
            
            print(f"Student {name} (ID: {student_id}) added successfully!")
            return True
            
        except Exception as e:
            print(f"Error adding student: {str(e)}")
            return False
    
    def load_face_encodings(self):
        """Load face encodings from pickle file"""
        try:
            with open(self.encodings_file, 'rb') as f:
                data = pickle.load(f)
                self.known_face_encodings = data['encodings']
                self.known_face_names = data['names']
            print(f"Loaded {len(self.known_face_encodings)} face encodings")
        except FileNotFoundError:
            print("No existing face encodings found.")
    
    def save_face_encodings(self):
        """Save face encodings to pickle file"""
        data = {
            'encodings': self.known_face_encodings,
            'names': self.known_face_names
        }
        with open(self.encodings_file, 'wb') as f:
            pickle.dump(data, f)
    
    def setup_attendance_file(self):
        """Setup attendance CSV file with headers"""
        if not os.path.exists(self.attendance_file):
            with open(self.attendance_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Name', 'Student_ID', 'Date', 'Time', 'Status'])
    
    def mark_attendance(self, name):
        """Mark attendance for a student"""
        if name in self.students_data:
            now = datetime.now()
            date_str = now.strftime("%Y-%m-%d")
            time_str = now.strftime("%H:%M:%S")
            
            # Check if already marked today
            if self.is_already_marked_today(name, date_str):
                print(f"Attendance already marked for {name} today")
                return False
            
            # Mark attendance
            with open(self.attendance_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    name,
                    self.students_data[name]['student_id'],
                    date_str,
                    time_str,
                    'Present'
                ])
            
            print(f"Attendance marked for {name} at {time_str}")
            return True
        return False
    
    def is_already_marked_today(self, name, date):
        """Check if attendance is already marked for today"""
        try:
            df = pd.read_csv(self.attendance_file)
            today_records = df[(df['Name'] == name) & (df['Date'] == date)]
            return len(today_records) > 0
        except:
            return False
    
    def start_attendance_session(self):
        """Start live attendance session using webcam"""
        video_capture = cv2.VideoCapture(0)
        
        # Set camera properties for better performance
        video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        video_capture.set(cv2.CAP_PROP_FPS, 30)
        
        print("Starting attendance session... Press 'q' to quit")
        
        # Process every nth frame for better performance
        frame_count = 0
        process_this_frame = True
        
        while True:
            ret, frame = video_capture.read()
            if not ret:
                print("Failed to grab frame from camera")
                break
            
            # Only process every other frame to save time
            if process_this_frame:
                # Resize frame for faster processing
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                
                # Convert BGR to RGB (face_recognition uses RGB)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                
                # Find faces in current frame
                try:
                    face_locations = face_recognition.face_locations(rgb_small_frame)
                    
                    # Only proceed if faces are found
                    if face_locations:
                        # Get face encodings - using a more robust approach
                        face_encodings = []
                        for face_location in face_locations:
                            try:
                                # Get face encoding for each face location
                                encoding = face_recognition.face_encodings(rgb_small_frame, [face_location], num_jitters=1)
                                if encoding:
                                    face_encodings.append(encoding[0])
                                else:
                                    face_encodings.append(None)
                            except Exception as e:
                                print(f"Error encoding face: {e}")
                                face_encodings.append(None)
                        
                        # Process each face
                        for i, (face_encoding, face_location) in enumerate(zip(face_encodings, face_locations)):
                            if face_encoding is None:
                                continue
                                
                            # Compare with known faces
                            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.6)
                            name = "Unknown"
                            
                            # Find best match
                            if True in matches:
                                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                                best_match_index = np.argmin(face_distances)
                                
                                if matches[best_match_index] and face_distances[best_match_index] < 0.6:
                                    name = self.known_face_names[best_match_index]
                                    
                                    # Mark attendance
                                    if self.mark_attendance(name):
                                        print(f"✓ Attendance marked for {name}")
                            
                            # Draw rectangle and label (scale back up)
                            top, right, bottom, left = face_location
                            top *= 4
                            right *= 4
                            bottom *= 4
                            left *= 4
                            
                            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                            
                            font = cv2.FONT_HERSHEY_DUPLEX
                            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.6, (255, 255, 255), 1)
                    
                except Exception as e:
                    print(f"Error processing frame: {e}")
                    # Continue to next frame
                    pass
            
            process_this_frame = not process_this_frame
            
            # Display frame
            cv2.imshow('Face Recognition Attendance', frame)
            
            # Check for quit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        video_capture.release()
        cv2.destroyAllWindows()
        print("Attendance session ended.")
    
    def test_camera(self):
        """Test camera functionality"""
        print("Testing camera... Press 'q' to quit")
        video_capture = cv2.VideoCapture(0)
        
        if not video_capture.isOpened():
            print("Error: Could not open camera")
            return False
        
        while True:
            ret, frame = video_capture.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            cv2.imshow('Camera Test', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        video_capture.release()
        cv2.destroyAllWindows()
        return True
    
    def generate_attendance_report(self, date=None):
        """Generate attendance report for a specific date"""
        try:
            df = pd.read_csv(self.attendance_file)
            
            if date:
                df = df[df['Date'] == date]
                print(f"\n=== Attendance Report for {date} ===")
            else:
                print(f"\n=== Complete Attendance Report ===")
            
            if df.empty:
                print("No attendance records found!")
                return
            
            print(df.to_string(index=False))
            
            # Summary statistics
            total_students = len(self.students_data)
            present_today = len(df[df['Date'] == datetime.now().strftime("%Y-%m-%d")]) if not date else len(df)
            
            print(f"\nSummary:")
            print(f"Total Registered Students: {total_students}")
            print(f"Present Today: {present_today}")
            if total_students > 0:
                print(f"Attendance Percentage: {(present_today/total_students)*100:.1f}%")
            
        except Exception as e:
            print(f"Error generating report: {str(e)}")
    
    def list_students(self):
        """List all registered students"""
        print("\n=== Registered Students ===")
        if not self.students_data:
            print("No students registered yet.")
            return
        
        for name, data in self.students_data.items():
            print(f"Name: {name}, ID: {data['student_id']}")
    
    def menu(self):
        """Main menu for the attendance system"""
        while True:
            print("\n" + "="*50)
            print("FACE RECOGNITION ATTENDANCE SYSTEM")
            print("="*50)
            print("1. Add New Student")
            print("2. Start Attendance Session")
            print("3. Generate Attendance Report")
            print("4. List All Students")
            print("5. Test Camera")
            print("6. Exit")
            print("="*50)
            
            choice = input("Enter your choice (1-6): ")
            
            if choice == '1':
                name = input("Enter student name: ")
                student_id = input("Enter student ID: ")
                image_path = input("Enter image path: ")
                
                if not os.path.exists(image_path):
                    print("Error: Image file not found!")
                    continue
                
                self.add_student(name, student_id, image_path)
            
            elif choice == '2':
                if not self.known_face_encodings:
                    print("No students registered yet! Please add students first.")
                    continue
                self.start_attendance_session()
            
            elif choice == '3':
                date = input("Enter date (YYYY-MM-DD) or press Enter for all records: ")
                if date.strip() == "":
                    date = None
                self.generate_attendance_report(date)
            
            elif choice == '4':
                self.list_students()
            
            elif choice == '5':
                self.test_camera()
            
            elif choice == '6':
                print("Goodbye!")
                break
            
            else:
                print("Invalid choice! Please try again.")

def main():
    # Check if required packages are installed
    try:
        import cv2
        import face_recognition
        import numpy as np
        import pandas as pd
        print("All required packages are installed.")
    except ImportError as e:
        print(f"Missing package: {e}")
        print("Please install required packages:")
        print("pip install opencv-python face-recognition pandas numpy")
        return
    
    # Initialize the attendance system
    attendance_system = FaceRecognitionAttendanceSystem()
    
    # Start the menu
    attendance_system.menu()

if __name__ == "__main__":
    main()