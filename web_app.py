from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, Response
import os
import json
import pandas as pd
from datetime import datetime
from werkzeug.utils import secure_filename
import base64
import cv2
import numpy as np
import face_recognition
import pickle
import threading
import time
import csv

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'student_images'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

class AttendanceWebSystem:
    def __init__(self):
        self.students_db = "students_database.json"
        self.attendance_file = "attendance_records.csv"
        self.encodings_file = "face_encodings.pkl"
        self.camera = None
        self.is_streaming = False
        self.attendance_session_active = False
        self.load_data()
        self.setup_attendance_file()
    
    def load_data(self):
        """Load existing data"""
        try:
            with open(self.students_db, 'r') as f:
                self.students_data = json.load(f)
        except FileNotFoundError:
            self.students_data = {}
        
        try:
            with open(self.encodings_file, 'rb') as f:
                data = pickle.load(f)
                self.known_face_encodings = data['encodings']
                self.known_face_names = data['names']
        except FileNotFoundError:
            self.known_face_encodings = []
            self.known_face_names = []
    
    def save_data(self):
        """Save data to files"""
        with open(self.students_db, 'w') as f:
            json.dump(self.students_data, f, indent=2)
        
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
    
    def add_student(self, name, student_id, image_path):
        """Add new student"""
        try:
            image = face_recognition.load_image_file(image_path)
            face_encodings = face_recognition.face_encodings(image)
            
            if len(face_encodings) == 0:
                return False, "No face detected in image"
            
            self.students_data[name] = {
                'student_id': student_id,
                'image_path': image_path,
                'added_date': datetime.now().isoformat()
            }
            
            self.known_face_encodings.append(face_encodings[0])
            self.known_face_names.append(name)
            self.save_data()
            
            return True, "Student added successfully"
        except Exception as e:
            return False, str(e)
    
    def mark_attendance(self, name):
        """Mark attendance for a student"""
        if name in self.students_data:
            now = datetime.now()
            date_str = now.strftime("%Y-%m-%d")
            time_str = now.strftime("%H:%M:%S")
            
            # Check if already marked today
            if self.is_already_marked_today(name, date_str):
                return False, "Already marked today"
            
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
            
            return True, f"Attendance marked for {name} at {time_str}"
        return False, "Student not found"
    
    def is_already_marked_today(self, name, date):
        """Check if attendance is already marked for today"""
        try:
            df = pd.read_csv(self.attendance_file)
            today_records = df[(df['Name'] == name) & (df['Date'] == date)]
            return len(today_records) > 0
        except:
            return False
    
    def get_attendance_data(self, date=None):
        """Get attendance data"""
        try:
            df = pd.read_csv(self.attendance_file)
            if date:
                df = df[df['Date'] == date]
            return df.to_dict('records')
        except FileNotFoundError:
            return []
    
    def get_students_list(self):
        """Get list of all students"""
        return self.students_data
    
    def start_camera(self):
        """Start camera for streaming"""
        if not self.is_streaming:
            self.camera = cv2.VideoCapture(0)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.is_streaming = True
            return True
        return False
    
    def stop_camera(self):
        """Stop camera streaming"""
        if self.is_streaming and self.camera:
            self.camera.release()
            self.is_streaming = False
            self.attendance_session_active = False
            return True
        return False
    
    def generate_frames(self):
        """Generate video frames for streaming"""
        frame_count = 0
        process_this_frame = True
        
        while self.is_streaming:
            if not self.camera:
                break
                
            ret, frame = self.camera.read()
            if not ret:
                break
            
            # Process face recognition if attendance session is active
            if self.attendance_session_active and process_this_frame:
                # Resize frame for faster processing
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                
                try:
                    # Find faces in current frame
                    face_locations = face_recognition.face_locations(rgb_small_frame)
                    
                    if face_locations:
                        # Get face encodings
                        face_encodings = []
                        for face_location in face_locations:
                            try:
                                encoding = face_recognition.face_encodings(rgb_small_frame, [face_location], num_jitters=1)
                                if encoding:
                                    face_encodings.append(encoding[0])
                                else:
                                    face_encodings.append(None)
                            except:
                                face_encodings.append(None)
                        
                        # Process each face
                        for face_encoding, face_location in zip(face_encodings, face_locations):
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
                                    success, message = self.mark_attendance(name)
                                    if success:
                                        print(f"✓ {message}")
                            
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
            
            process_this_frame = not process_this_frame
            
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            time.sleep(0.1)  # Small delay to prevent excessive CPU usage

# Initialize the system
attendance_system = AttendanceWebSystem()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/students')
def students():
    students_list = attendance_system.get_students_list()
    return render_template('students.html', students=students_list)

@app.route('/add_student', methods=['GET', 'POST'])
def add_student():
    if request.method == 'POST':
        name = request.form['name']
        student_id = request.form['student_id']
        
        if 'image' not in request.files:
            flash('No image file selected')
            return redirect(request.url)
        
        file = request.files['image']
        if file.filename == '':
            flash('No image file selected')
            return redirect(request.url)
        
        if file:
            filename = secure_filename(f"{name}_{student_id}.jpg")
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            success, message = attendance_system.add_student(name, student_id, filepath)
            
            if success:
                flash(f'Student {name} added successfully!', 'success')
                return redirect(url_for('students'))
            else:
                flash(f'Error: {message}', 'error')
    
    return render_template('add_student.html')

@app.route('/attendance')
def attendance():
    date = request.args.get('date')
    attendance_data = attendance_system.get_attendance_data(date)
    return render_template('attendance.html', attendance=attendance_data, date=date)

@app.route('/live_attendance')
def live_attendance():
    return render_template('live_attendance.html')

@app.route('/start_camera')
def start_camera():
    """Start camera for live streaming"""
    if attendance_system.start_camera():
        return jsonify({'status': 'success', 'message': 'Camera started'})
    return jsonify({'status': 'error', 'message': 'Camera already running'})

@app.route('/stop_camera')
def stop_camera():
    """Stop camera streaming"""
    if attendance_system.stop_camera():
        return jsonify({'status': 'success', 'message': 'Camera stopped'})
    return jsonify({'status': 'error', 'message': 'Camera not running'})

@app.route('/start_attendance')
def start_attendance():
    """Start attendance session"""
    if attendance_system.is_streaming:
        attendance_system.attendance_session_active = True
        return jsonify({'status': 'success', 'message': 'Attendance session started'})
    return jsonify({'status': 'error', 'message': 'Camera not running'})

@app.route('/stop_attendance')
def stop_attendance():
    """Stop attendance session"""
    attendance_system.attendance_session_active = False
    return jsonify({'status': 'success', 'message': 'Attendance session stopped'})

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    if attendance_system.is_streaming:
        return Response(attendance_system.generate_frames(),
                       mimetype='multipart/x-mixed-replace; boundary=frame')
    return "Camera not started", 404

@app.route('/attendance_status')
def attendance_status():
    """Get current attendance session status"""
    return jsonify({
        'camera_running': attendance_system.is_streaming,
        'attendance_active': attendance_system.attendance_session_active,
        'students_registered': len(attendance_system.known_face_encodings)
    })

if __name__ == '__main__':
    app.run(debug=True, threaded=True)