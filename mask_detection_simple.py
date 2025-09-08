import cv2
import numpy as np
import os
import time
import winsound   # For beep sound on Windows
from PIL import ImageFont, ImageDraw, Image
from datetime import datetime

# ---------------- Age, Gender Buckets ----------------
AGE_BUCKETS = ["(0-2)", "(4-6)", "(8-12)", "(15-20)",
               "(25-32)", "(38-43)", "(48-53)", "(60-100)"]
GENDERS = ["Male", "Female"]

# ---------------- Object Classes ----------------
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"]

# ---------------- Font Helper (Poppins) ----------------
def putText_PIL(frame, text, position, font_path="Poppins-Regular.ttf",
                font_size=20, color=(255, 255, 255)):
    """ Draws text with Poppins font on OpenCV frames using PIL """
    cv2_im_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(cv2_im_rgb)

    try:
        font = ImageFont.truetype(font_path, font_size)
    except:
        font = ImageFont.load_default()

    draw = ImageDraw.Draw(pil_im)
    draw.text(position, text, font=font, fill=color)

    return cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)

# ---------------- Video Recorder Class ----------------
class VideoRecorder:
    def __init__(self, folder="recordings", fps=20.0, frame_size=(640,480)):
        self.folder = folder
        os.makedirs(self.folder, exist_ok=True)
        self.fps = fps
        self.frame_size = frame_size
        self.writer = None
        self.is_recording = False

    def start(self):
        if not self.is_recording:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.folder, f"session_{timestamp}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(filename, fourcc, self.fps, self.frame_size)
            self.is_recording = True
            print(f"[REC] Started recording: {filename}")

    def stop(self):
        if self.is_recording:
            self.writer.release()
            self.is_recording = False
            print("[REC] Stopped recording.")

    def write(self, frame):
        if self.is_recording and self.writer is not None:
            self.writer.write(frame)

# ---------------- REC Indicator ----------------
def draw_rec_indicator(frame, is_recording):
    if is_recording:
        (h, w) = frame.shape[:2]
        cv2.circle(frame, (w - 90, 35), 10, (0, 0, 255), -1)
        frame = putText_PIL(frame, "REC", (w - 75, 25),
                            font_size=18, color=(255, 255, 255))
    return frame

# ---------------- Face + Mask + Age + Gender ----------------
def detect_and_predict_mask(frame, faceNet, ageNet, genderNet):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()
    
    results = []
    
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            
            face = frame[startY:endY, startX:endX]
            if face.size == 0:
                continue
            
            # -------- Mask Detection (simple HSV color-based) --------
            face_hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
            lower_mask = np.array([90, 50, 50])
            upper_mask = np.array([130, 255, 255])
            mask_region = cv2.inRange(face_hsv, lower_mask, upper_mask)
            lower_half = mask_region[mask_region.shape[0]//2:, :]
            mask_pixel_percentage = np.sum(lower_half > 0) / (lower_half.size) * 100
            mask_score = min(mask_pixel_percentage / 30, 1.0)
            without_mask_score = 1.0 - mask_score
            mask_label = "Mask" if mask_score > without_mask_score else "No Mask"
            
            # -------- Age & Gender Detection --------
            face_blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227),
                                              (78.4263377603, 87.7689143744, 114.895847746),
                                              swapRB=False)
            
            genderNet.setInput(face_blob)
            gender_preds = genderNet.forward()
            gender = GENDERS[gender_preds[0].argmax()]
            
            ageNet.setInput(face_blob)
            age_preds = ageNet.forward()
            age = AGE_BUCKETS[age_preds[0].argmax()]
            
            results.append(((startX, startY, endX, endY), mask_label, gender, age))
    
    return results

# ---------------- Camera Selection Function ----------------
def select_camera():
    selection_window = np.zeros((300, 500, 3), dtype=np.uint8)
    cv2.putText(selection_window, "Select Camera Source", (100, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(selection_window, "1 - Laptop Camera (0)", (50, 120), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(selection_window, "2 - USB Camera (1)", (50, 170), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(selection_window, "3 - Test All Cameras", (50, 220), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(selection_window, "Press 1, 2, or 3 to select", (100, 270), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    cv2.imshow("Camera Selection", selection_window)
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('1'):
            cv2.destroyWindow("Camera Selection")
            return 0
        elif key == ord('2'):
            cv2.destroyWindow("Camera Selection")
            return 1
        elif key == ord('3'):
            cv2.destroyWindow("Camera Selection")
            return -1

# ---------------- Test All Cameras Function ----------------
def test_all_cameras():
    print("[INFO] Testing all available cameras...")
    available = []
    for i in range(0, 10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                available.append(i)
                print(f"Camera found at index {i}")
                preview = cv2.resize(frame, (320, 240))
                cv2.imshow(f"Camera {i}", preview)
                cv2.waitKey(500)
                cv2.destroyWindow(f"Camera {i}")
            cap.release()
    return available if available else [0]

# ---------------- Main ----------------
def main():
    print("[INFO] Loading models...")
    prototxtPath = os.path.join("face_detector", "deploy.prototxt")
    weightsPath = os.path.join("face_detector", "res10_300x300_ssd_iter_140000.caffemodel")
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
    
    ageNet = cv2.dnn.readNet(
        os.path.join("age_gender", "age_deploy.prototxt"),
        os.path.join("age_gender", "age_net.caffemodel")
    )
    genderNet = cv2.dnn.readNet(
        os.path.join("age_gender", "gender_deploy.prototxt"),
        os.path.join("age_gender", "gender_net.caffemodel")
    )

    obj_proto = os.path.join("object_detector", "MobileNetSSD_deploy.prototxt")
    obj_model = os.path.join("object_detector", "MobileNetSSD_deploy.caffemodel")
    objectNet = cv2.dnn.readNetFromCaffe(obj_proto, obj_model)
    
    # --- Default to Laptop Camera ---
    camera_index = 0  
    available_cams = test_all_cameras()
    print(f"[INFO] Available cameras: {available_cams}")
    
    print(f"[INFO] Starting video stream from laptop camera (index {camera_index})...")
    vs = cv2.VideoCapture(camera_index)
    
    width = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
    aspect_ratio = width / height
    fps = vs.get(cv2.CAP_PROP_FPS)
    if fps == 0:  
        fps = 20.0
    
    recorder = VideoRecorder(frame_size=(width, height), fps=fps)
    
    print(f"[INFO] Camera resolution: {width}x{height}")
    
    cv2.namedWindow("Mask-Age-Gender + Object Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Mask-Age-Gender + Object Detection", 800, int(800 / aspect_ratio))

    warning_offset = 800  

    # --- Tracking variables ---
    last_snapshot_time = 0
    snapshot_interval = 10  # seconds
    no_mask_start_time = None
    prev_mask_status = {}

    while True:
        ret, frame = vs.read()
        if not ret:
            print("[WARN] Failed to grab frame. Restarting default cam...")
            vs.release()
            vs = cv2.VideoCapture(0)
            cv2.waitKey(1000)
            continue

        (h, w) = frame.shape[:2]

        # ---- Object Detection ----
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                     0.007843, (300, 300), 127.5)
        objectNet.setInput(blob)
        detections = objectNet.forward()

        object_counts = {}
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.4:
                idx = int(detections[0, 0, i, 1])
                label = CLASSES[idx]
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2)
                frame = putText_PIL(frame, label, (startX, startY - 25),
                                    font_size=18, color=(0,0,255))
                if label != "person":
                    object_counts[label] = object_counts.get(label, 0) + 1

        # ---- Face Detection + Mask/Age/Gender ----
        results = detect_and_predict_mask(frame, faceNet, ageNet, genderNet)
        for (box, mask_label, gender, age) in results:
            (startX, startY, endX, endY) = box
            color = (0, 255, 0) if mask_label == "Mask" else  (255, 215, 0)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            label = f"{mask_label}, {gender}, Age: {age}"
            frame = putText_PIL(frame, label, (startX, startY - 20),
                                font_size=18, color=color)

        # ---- Auto Recording Logic ----
        any_no_mask = any(mask_label == "No Mask" for (_, mask_label, _, _) in results)
        if any_no_mask:
            if no_mask_start_time is None:
                no_mask_start_time = time.time()
            elif time.time() - no_mask_start_time >= 10:  # 10 sec continuous
                if not recorder.is_recording:
                    recorder.start()
        else:
            no_mask_start_time = None

        # ---- Beep + Snapshot when mask is worn ----
        for idx, (_, mask_label, gender, age) in enumerate(results, start=1):
            person_id = idx
            previous_status = prev_mask_status.get(person_id)

            if previous_status == "No Mask" and mask_label == "Mask":
                # Beep sound
                winsound.Beep(1000, 400) # Frequency 1000Hz, Duration 400ms
                print(f"[BEEP] Person {person_id} wore mask")

                # Save snapshot immediately when mask is worn
                os.makedirs("snapshots", exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join("snapshots", f"mask_worn_{timestamp}_p{person_id}.jpg")
                cv2.imwrite(filename, frame)
                print(f"[SNAPSHOT] Saved (Mask Worn): {filename}")

            prev_mask_status[person_id] = mask_label

        # ---- Snapshot saving every interval ----
        if results:
            current_time = time.time()
            if current_time - last_snapshot_time >= snapshot_interval:
                os.makedirs("snapshots", exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                for idx, (box, mask_label, gender, age) in enumerate(results, start=1):
                    filename = os.path.join("snapshots", f"{mask_label}_{timestamp}_p{idx}.jpg")
                    cv2.imwrite(filename, frame)
                    print(f"[SNAPSHOT] Saved: {filename}")
                last_snapshot_time = current_time

        # ---- Warning Banner ----
        if any_no_mask:
            banner_height = 20
            cv2.rectangle(frame, (0, 0), (w, banner_height), (0, 0, 255), -1)
            warning_text = "⚠ WARNING: PLEASE WEAR A MASK! ⚠"
            frame = putText_PIL(frame, warning_text, (warning_offset, 5),  
                                font_size=12, color=(255,255,255))
            warning_offset -= 20
            if warning_offset < -100:
                warning_offset = w
        else:
            warning_offset = w

               # ---- Info Board ----
        board_w, board_h = 200, 260  
        board_x, board_y = 10, h - board_h - 10

        overlay = frame.copy()
        cv2.rectangle(overlay, (board_x, board_y),
                      (board_x + board_w, board_y + board_h),
                      (0, 0, 0), -1)
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        frame = putText_PIL(frame, "Info Board:", (board_x + 10, board_y + 10),
                            font_size=16, color=(0,255,255))

        y_offset = board_y + 35

        # ---- Date / Time / Day ----
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")
        day_str  = now.strftime("%A")

        frame = putText_PIL(frame, f"Date: {date_str}", (board_x + 10, y_offset),
                            font_size=12, color=(255, 255, 255))
        y_offset += 18
        frame = putText_PIL(frame, f"Time: {time_str}", (board_x + 10, y_offset),
                            font_size=12, color=(255, 255, 255))
        y_offset += 18
        frame = putText_PIL(frame, f"Day:  {day_str}", (board_x + 10, y_offset),
                            font_size=12, color=(255, 255, 255))
        y_offset += 25

        if object_counts:
            frame = putText_PIL(frame, "Objects:", (board_x + 10, y_offset),
                                font_size=14, color=(0,200,255))
            y_offset += 20
            for obj, count in object_counts.items():
                frame = putText_PIL(frame, f"{obj}: {count}", (board_x + 20, y_offset),
                                    font_size=12, color=(255, 255, 255))
                y_offset += 18

        if results:
            frame = putText_PIL(frame, f"Persons: {len(results)}", (board_x + 10, y_offset + 5),
                                font_size=14, color=(0,200,255))
            y_offset += 30
            for idx, (_, mask_label, gender, age) in enumerate(results, start=1):
                gender_abbr = "M" if gender == "Male" else "F"
                info_line1 = f"{idx}. {gender_abbr}, Age {age}"
                frame = putText_PIL(frame, info_line1, (board_x + 20, y_offset),
                                    font_size=12, color=(255, 255, 255))
                y_offset += 15
                mask_status = "Mask" if mask_label == "Mask" else "No Mask"
                color = (0, 255, 0) if mask_label == "Mask" else (10, 100, 255)
                frame = putText_PIL(frame, f"Status: {mask_status}", (board_x + 35, y_offset),
                                    font_size=12, color=color)
                y_offset += 20

        # ---- Camera source info ----
        camera_source = "Laptop Cam" if camera_index == 0 else f"USB Cam ({camera_index})"
        frame = putText_PIL(frame, f"Source: {camera_source}", (10, 30),
                            font_size=14, color=(255, 255, 255))

        # ---- Write
        #------- Recording ----
        recorder.write(frame)

        # ---- REC Indicator ----
        frame = draw_rec_indicator(frame, recorder.is_recording)
        
        cv2.imshow("Mask-Age-Gender + Object Detection", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):  # Quit
            break
        elif key == ord("s"):  # Switch camera (menu)
            vs.release()
            cv2.destroyAllWindows()
            camera_index = select_camera()
            vs = cv2.VideoCapture(camera_index)
        elif key == ord("c"):  # Cycle cameras
            current_idx = available_cams.index(camera_index) if camera_index in available_cams else -1
            next_idx = (current_idx + 1) % len(available_cams)
            camera_index = available_cams[next_idx]
            vs.release()
            vs = cv2.VideoCapture(camera_index)
        elif key == ord("r"):  # Manual recording toggle
            if recorder.is_recording:
                recorder.stop()
            else:
                recorder.start()

    # ---- Cleanup ----
    vs.release()
    recorder.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
