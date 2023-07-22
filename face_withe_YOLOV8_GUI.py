import cv2
import numpy as np
import face_recognition
import os
import pandas as pd
import datetime
import tkinter as tk
from tkinter import messagebox, filedialog
from PIL import Image, ImageTk
import threading
from ultralytics import YOLO
import pickle


def euclidean_distance(face_encodings, face_to_compare):
    if len(face_encodings) == 0:
        return np.empty((0))
    return np.linalg.norm(face_encodings - face_to_compare, axis=1)


def convert_image_numpy_array(file, mode='RGB'):
    im = Image.open(file)
    if mode:
        im = im.convert(mode)
    return np.array(im)
def compare_faces(known_face_encodings, face_encoding_to_check, tolerance=0.6):
    return list(euclidean_distance(known_face_encodings, face_encoding_to_check) <= tolerance)

class FaceRecognitionGUI:
    def __init__(self, persons_folder, output_file):
        self.persons_folder = persons_folder
        self.output_file = output_file
        self.student_info = pd.DataFrame(columns=['Name', 'Time', 'Image'])
        self.images = []
        self.classNames = []
        self.encodeListKnown = []

        self.window = tk.Tk()
        self.window.title("Face Recognition")

        # Create a frame for the camera feed
        self.camera_frame = tk.Frame(self.window)
        self.camera_frame.pack(side=tk.LEFT, padx=10, pady=10)

        # Create a frame for the student info
        self.info_frame = tk.Frame(self.window)
        self.info_frame.pack(side=tk.LEFT, padx=10, pady=10)

        # Create a scrollable text area for student info
        self.scrollbar = tk.Scrollbar(self.info_frame)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.text_area = tk.Text(self.info_frame, yscrollcommand=self.scrollbar.set)
        self.text_area.pack()

        self.scrollbar.config(command=self.text_area.yview)

        # Start Recognition button
        self.start_button = tk.Button(self.window, text="Start Recognition", command=self.start_recognition)
        self.start_button.pack(pady=10)

        # Save to Excel button
        self.save_button = tk.Button(self.window, text="Save to Excel", command=self.save_to_excel)
        self.save_button.pack(pady=10)

        self.window.protocol("WM_DELETE_WINDOW", self.on_close)

        self.camera_thread = None
        self.stop_camera = False

    def on_close(self):
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.stop_camera = True
            self.window.destroy()

    def load_encodings_from_pickle(self):
        # Load encodeListKnown and classNames from the pickle file
        with open('encodings.pickle', 'rb') as f:
            self.encodeListKnown = pickle.load(f)
            self.classNames = pickle.load(f)

    def recognize_faces(self):
        self.load_encodings_from_pickle()
        model = YOLO(r"C:\Users\asus\Downloads\yolov8_face.pt")

        cap = cv2.VideoCapture(0)

        while True:
            if self.stop_camera:
                break

            _, img = cap.read()
            if img is None:
                continue

            imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
            imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

            detect_params = model.predict(source=[img], conf=0.45, save=False)

            DP = detect_params[0].numpy()

            if len(DP) != 0:
                for i in range(len(detect_params[0])):
                    boxes = detect_params[0].boxes
                    box = boxes[i]
                    bb = box.xyxy.numpy()[0]

                    # Display class name and confidence
                    font = cv2.FONT_HERSHEY_COMPLEX

                    # Clip the shape using the bounding box coordinates
                    clipped_shape = img[int(bb[1]):int(bb[3]), int(bb[0]):int(bb[2])]

                    # Compute face encoding for the clipped shape
                    face_encodings = face_recognition.face_encodings(clipped_shape)

                    if len(face_encodings) > 0:
                        face_encoding = face_encodings[0]  # Assuming there's only one face in each frame
                        for encodeFace, faceLoc in zip(face_encodings, clipped_shape):
                            matches = compare_faces(self.encodeListKnown, encodeFace,tolerance=0.55)
                            

                        # Compare face encoding with data encodings
                        face_distances = euclidean_distance(self.encodeListKnown, face_encoding)
                        most_similar_index = np.argmin(face_distances)
                        most_similar_image_name = self.classNames[most_similar_index]
                        if np.any(matches):
                            new_name = most_similar_image_name.replace('.jpg', '')

                            # Display the name of the most similar image above the bounding box
                            cv2.putText(img,new_name,(int(bb[0]), int(bb[1]) - 30),font,1,(255, 255, 255),1)

                            # Draw rectangle around the face
                            cv2.rectangle(img, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), (0, 255, 0), 2)

                            # Capture an image of the recognized student
                            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                            image_filename = f"recognized_{most_similar_image_name}_{timestamp}.jpg"
                            image_path = os.path.join(self.persons_folder, image_filename)
                            # cv2.imwrite(image_path, clipped_shape)

                            # Add student info to DataFrame
                            self.add_student_info(new_name,
                                                  datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                                  image_filename)
                        else:
                            # Draw rectangle around the clipped shape
                            cv2.putText(img, "Uknown", (int(bb[0]), int(bb[1]) - 30), font, 1, (255, 0, 0), 1)
                            cv2.rectangle(img, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), (255, 0, 0), 2)

            # Convert the image to PIL format and resize it
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = img.resize((400, 300), Image.LANCZOS)
            img = ImageTk.PhotoImage(image=img)

            # Display the image in a label
            self.camera_label.config(image=img)
            self.camera_label.image = img

            # Update the student info in the text area
            self.update_student_info()

    def add_student_info(self, name, time, image_filename):
        if name not in self.student_info['Name'].values:
            new_row = {'Name': name, 'Time': time, 'Image': image_filename}
            self.student_info = pd.concat([self.student_info, pd.DataFrame(new_row, index=[0])], ignore_index=True)
            self.update_student_info()

    def update_student_info(self):
        self.text_area.delete(1.0, tk.END)
        for _, row in self.student_info.iterrows():
            name = row['Name']
            time = row['Time']
            info_text = f'Name: {name}\tTime: {time}\n'
            self.text_area.insert(tk.END, info_text)

    def start_recognition(self):
        self.start_button.config(state=tk.DISABLED)
        self.load_encodings_from_pickle()
        self.stop_camera = False
        self.camera_thread = threading.Thread(target=self.recognize_faces)
        self.camera_thread.start()

    def save_to_excel(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".xlsx")
        if not file_path:
            return

        self.student_info.to_excel(file_path, index=False)
        messagebox.showinfo("Success", "Student information saved to Excel file!")


    def run(self):
        # Create a label for the camera feed
        self.camera_label = tk.Label(self.camera_frame)
        self.camera_label.pack()

        self.window.mainloop()


# Usage example
def main():
    persons_folder = r'C:\Users\asus\Desktop\Artificial intelligence\Third Year\Summer\EVC\Face-recognition-python-project-main\persons'
    output_file = 'student_info.xlsx'

    face_recognition_gui = FaceRecognitionGUI(persons_folder, output_file)
    face_recognition_gui.run()


if __name__ == '__main__':
    main()
