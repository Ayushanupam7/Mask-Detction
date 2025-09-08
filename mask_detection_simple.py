import cv2
import numpy as np
import os
from PIL import ImageFont, ImageDraw, Image

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
    """
    Draws text with Poppins font on OpenCV frames using PIL.
    """
    # Convert OpenCV image (BGR) to PIL (RGB)
    cv2_im_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(cv2_im_rgb)

    # Load custom font
    try:
        font = ImageFont.truetype(font_path, font_size)
    except:
        font = ImageFont.load_default()

    draw = ImageDraw.Draw(pil_im)
    draw.text(position, text, font=font, fill=color)

    # Convert back to OpenCV (BGR)
    return cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)

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

# ---------------- Main ----------------
def main():
    print("[INFO] Loading models...")
    
    # Face detector
    prototxtPath = os.path.join("face_detector", "deploy.prototxt")
    weightsPath = os.path.join("face_detector", "res10_300x300_ssd_iter_140000.caffemodel")
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
    
    # Age & gender models
    ageNet = cv2.dnn.readNet(
        os.path.join("age_gender", "age_deploy.prototxt"),
        os.path.join("age_gender", "age_net.caffemodel")
    )
    genderNet = cv2.dnn.readNet(
        os.path.join("age_gender", "gender_deploy.prototxt"),
        os.path.join("age_gender", "gender_net.caffemodel")
    )

    # Object detection model (MobileNet SSD)
    obj_proto = os.path.join("object_detector", "MobileNetSSD_deploy.prototxt")
    obj_model = os.path.join("object_detector", "MobileNetSSD_deploy.caffemodel")
    objectNet = cv2.dnn.readNetFromCaffe(obj_proto, obj_model)
    
    print("[INFO] Starting video stream...")
    vs = cv2.VideoCapture(0)
    
    # Get the original frame dimensions
    width = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
    aspect_ratio = width / height
    
    # Create resizable window
    cv2.namedWindow("Mask-Age-Gender + Object Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Mask-Age-Gender + Object Detection", 800, int(800 / aspect_ratio))

    # For scrolling warning banner
    warning_offset = 800  

    while True:
        ret, frame = vs.read()
        if not ret:
            print("Failed to grab frame")
            break

        (h, w) = frame.shape[:2]

        # ---- Object Detection ----
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                     0.007843, (300, 300), 127.5)
        objectNet.setInput(blob)
        detections = objectNet.forward()

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

        # ---- Face Detection + Mask/Age/Gender ----
        results = detect_and_predict_mask(frame, faceNet, ageNet, genderNet)
        for (box, mask_label, gender, age) in results:
            (startX, startY, endX, endY) = box
            color = (0, 255, 0) if mask_label == "Mask" else  (255, 215, 0)
            
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            label = f"{mask_label}, {gender}, Age: {age}"
            frame = putText_PIL(frame, label, (startX, startY - 20),
                                font_size=18, color=color)

        # ---- Warning Banner if No Mask ----
        any_no_mask = any(mask_label == "No Mask" for (_, mask_label, _, _) in results)
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

        # ---- Information Board (Transparent, Bottom-Left) ----
        # Increased size for the info board to accommodate two lines
        board_w, board_h = 150, 120  # Increased width and height
        board_x, board_y = 10, h - board_h - 10
        radius = 15 

        # Create a transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (board_x, board_y),
                      (board_x + board_w, board_y + board_h),
                      (0, 0, 0), -1)
        alpha = 1  # Transparency factor (0 = fully transparent, 1 = fully opaque)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        frame = putText_PIL(frame, "Info Board:-", (board_x + 10, board_y + 10),
                            font_size=16, color=(0,255,255))  # Smaller font
        frame = putText_PIL(frame, f"Persons: {len(results)}", (board_x + 10, board_y + 30),
                            font_size=14)  # Smaller font

        y_offset = board_y + 50
        for idx, (_, mask_label, gender, age) in enumerate(results, start=1):
            # Use abbreviated labels to save space
            gender_abbr = "Male" if gender == "Male" else "F"
            
            # First line: Person number, gender and age
            info_line1 = f"{idx}. {gender_abbr}, Age{age}"
            frame = putText_PIL(frame, info_line1, (board_x + 10, y_offset),
                                font_size=12, color=(255, 255, 255))  # White color for first line
            
            # Second line: Mask status with appropriate color
            mask_status = "Mask" if mask_label == "Wearing Mask" else "Not Wear Mask"
            color = (0, 255, 0) if mask_label == "Mask" else (10, 100, 255)
            frame = putText_PIL(frame, mask_status, (board_x + 10, y_offset + 15),
                                font_size=12, color=color)
            
            y_offset += 30  # Increased line spacing to accommodate two lines

        # ---- Resize frame to fit window while maintaining aspect ratio ----
        # Get current window size
        win_rect = cv2.getWindowImageRect("Mask-Age-Gender + Object Detection")
        win_width, win_height = win_rect[2], win_rect[3]
        
        # Check if window dimensions are valid (not zero)
        if win_height <= 0 or win_width <= 0:
            # If window is minimized or has invalid dimensions, just show the frame as is
            cv2.imshow("Mask-Age-Gender + Object Detection", frame)
        else:
            # Calculate the aspect ratio of the window
            win_aspect_ratio = win_width / win_height
            
            # Resize frame to fit window while maintaining aspect ratio
            if aspect_ratio > win_aspect_ratio:
                # Window is taller than frame
                new_width = win_width
                new_height = int(win_width / aspect_ratio)
            else:
                # Window is wider than frame
                new_height = win_height
                new_width = int(win_height * aspect_ratio)
                
            resized_frame = cv2.resize(frame, (new_width, new_height))
            
            # Create a black background and center the frame
            display_frame = np.zeros((win_height, win_width, 3), dtype=np.uint8)
            y_offset = (win_height - new_height) // 2
            x_offset = (win_width - new_width) // 2
            display_frame[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_frame
            
            # ---- Show Frame ----
            cv2.imshow("Mask-Age-Gender + Object Detection", display_frame)
        
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
    
    vs.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()