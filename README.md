# Face Recognition Application with Real-time Face Detection

This repository contains a Python script for a face recognition application with real-time face detection using the YOLOv4 model. The application provides a graphical user interface (GUI) built using tkinter and utilizes the face_recognition library for face recognition tasks.

https://github.com/alhydrym7/Face_Recognition_With_Yolov8/assets/50909741/c7d9fce1-69fd-4fc4-ada9-3c9f8c0e347b


# Features
  1. Real-time face recognition using YOLOv4 model for face detection
  2. Comparison of detected faces with known face encodings
  3. Display of recognized student information in a GUI text area
  4. Ability to save recognized student information to an Excel file

# Dependencies
  Before running the face recognition application, make sure you have the following libraries installed:

  1. OpenCV (cv2)
  2. NumPy (numpy)
  3. face_recognition
  4. pandas (pd)
  5. datetime
  6. tkinter (tk)
  7. Pillow (PIL)
  8. threading
  9. ultralytics
  10. pickle
  To install the required libraries, you can use pip in your terminal or command prompt:

    pip install opencv-python numpy face-recognition pandas pillow
    pip install tkinter ultralytics

# How to Use
  1. Clone the repository or download the Python script face_recognition_app.py to your local machine.

  2. In the script, update the persons_folder variable with the path to the folder containing the images of known persons. Each image should be named with the person's name (e.g., "john.jpg").

  3. Set the output_file variable to the desired name of the Excel file where recognized student information will be saved.

  4. To run the face recognition application, execute the script using the following command:

         python face_withe_YOLOV8_GUI.py
  5. The application window will open, displaying the real-time camera feed along with the recognized student information.

  6. Click on the "Start Recognition" button to initiate the face recognition process.

  7. As faces are detected in the camera feed, the application will compare them with the known face encodings and display the name of the most similar image above the bounding box of the recognized face.

  8. If a recognized face is found, the student's name, timestamp, and a captured image of the recognized student will be added to the student information display.

  9. Click on the "Save to Excel" button to save the recognized student information to the specified Excel file.

# How Faces Are Compared
  The comparison of faces is performed using the euclidean_distance method, which calculates the Euclidean distance between the face encodings of the detected face and the known face encodings stored in the application. The smaller the Euclidean distance, the more similar the faces are considered.
  
  The compare_faces method is used to compare a list of known face encodings against the detected face encoding to determine if they match. The method takes the list of known face encodings, a single face encoding to compare against the list, and a tolerance parameter that sets the threshold for considering a match. A lower tolerance value (e.g., 0.6) ensures stricter matching.
  
  If a match is found within the tolerance, the detected face is recognized as a known person, and their information is displayed and stored.


# Note
  For the face recognition to work properly, you need to generate face encodings for known persons and save them in a pickle file. The script assumes that the face encodings are stored in the encodings.pickle file, which is loaded during the application's initialization.
  
  For face encoding generation, consider using a separate script or tool that uses the face_recognition library to encode the faces of known persons and save the encodings to the encodings.pickle file.
  
  Please make sure to provide the necessary face encodings for known persons before running the application to achieve accurate face recognition results.
  
  Enjoy using the face recognition application with real-time face detection! If you encounter any issues or have suggestions for improvements, feel free to contribute or contact us.
