import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from sklearn.svm import SVC
import joblib
import pandas as pd
import os
from datetime import datetime

# ================= FaceMesh =================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# ================= Data =================
X, y = [], []
names = []
samples = 20

# ================= Camera =================
cap = cv2.VideoCapture(0)

# ================= Utils =================
def extract_vector(landmarks):
    vec = np.array([[lm.x, lm.y] for lm in landmarks.landmark]).flatten()
    return vec / np.linalg.norm(vec)


def mark_attendance(name):
    file = "attendance.csv"
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")

    if os.path.exists(file):
        df = pd.read_csv(file)
        if ((df["Name"] == name) & (df["Date"] == date)).any():
            return
    else:
        df = pd.DataFrame(columns=["Name", "Date", "Time"])

    df.loc[len(df)] = [name, date, time]
    df.to_csv(file, index=False)


# ================= GUI =================
root = tk.Tk()
root.title("Face Login Attendance System")
root.geometry("900x650")

tk.Label(root, text="ML Face Login Attendance",
         font=("Arial", 20, "bold")).pack(pady=10)

video_label = tk.Label(root)
video_label.pack()

status = tk.Label(root, text="Status: Ready",
                  font=("Arial", 14))
status.pack(pady=10)

name_entry = tk.Entry(root, font=("Arial", 14))
name_entry.pack(pady=5)
name_entry.insert(0, "Enter Name")


# ================= Functions =================
def update_frame():
    ret, frame = cap.read()
    if not ret:
        return

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if result.multi_face_landmarks:
        for lm in result.multi_face_landmarks[0].landmark:
            h, w, _ = frame.shape
            cv2.circle(frame,
                       (int(lm.x * w), int(lm.y * h)),
                       1, (0, 255, 0), -1)

    img = ImageTk.PhotoImage(
        Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    )
    video_label.imgtk = img
    video_label.configure(image=img)

    root.after(10, update_frame)


def register_face():
    name = name_entry.get().strip()

    if name == "" or name == "Enter Name":
        messagebox.showerror("Error", "Enter valid name")
        return

    ret, frame = cap.read()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if result.multi_face_landmarks:

        # if new person
        if name not in names:
            names.append(name)

        label = names.index(name)

        for _ in range(samples):
            X.append(extract_vector(result.multi_face_landmarks[0]))
            y.append(label)

        status.config(text=f"Registered: {name}")

    else:
        messagebox.showwarning("Warning", "No face detected")


def train_model():

    if len(X) == 0:
        messagebox.showerror("Error", "No data")
        return

    model = SVC(kernel="linear", probability=True)
    model.fit(np.array(X), np.array(y))

    joblib.dump(model, "face_model.pkl")
    joblib.dump(names, "names.pkl")

    status.config(text="Model trained")
    messagebox.showinfo("Success", "Training done")


def login_face():
    if not os.path.exists("face_model.pkl"):
        messagebox.showerror("Error", "Train first")
        return

    model = joblib.load("face_model.pkl")
    names_list = joblib.load("names.pkl")

    ret, frame = cap.read()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if result.multi_face_landmarks:

        features = extract_vector(
            result.multi_face_landmarks[0]
        )

        pred = model.predict([features])[0]
        conf = np.max(model.predict_proba([features]))

        if conf > 0.4:

            name = names_list[pred]
            mark_attendance(name)

            status.config(
                text=f"Login Success: {name} ({conf:.2f})"
            )

        else:
            status.config(text="Unknown face")

    else:
        status.config(text="No face detected")


# ================= Buttons =================
btn_frame = tk.Frame(root)
btn_frame.pack(pady=15)

tk.Button(
    btn_frame,
    text="Register Face",
    font=("Arial", 12),
    width=15,
    command=register_face
).grid(row=0, column=0, padx=10)

tk.Button(
    btn_frame,
    text="Train Model",
    font=("Arial", 12),
    width=15,
    command=train_model
).grid(row=0, column=1, padx=10)

tk.Button(
    btn_frame,
    text="Login",
    font=("Arial", 12),
    width=15,
    command=login_face
).grid(row=0, column=2, padx=10)


# ================= Start =================
update_frame()
root.mainloop()

cap.release()
cv2.destroyAllWindows()
