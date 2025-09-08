# 🛡️ Smart Mask, Age & Gender Detection with Object Recognition  

This project is a **real-time AI-based surveillance system** that uses computer vision to:  

- Detect **faces** and classify them as **Mask** 😷 or **No Mask** ❌  
- Estimate **Age** and **Gender** of each detected person  
- Count and display detected **objects** using MobileNet SSD  
- Display an **Info Board** with:
  - Number of persons  
  - Person details (Mask/No Mask, Gender, Age)  
  - Current **Date, Time, and Day**  
- Show a scrolling **warning banner** if someone is not wearing a mask  
- Save **snapshots** every 10 seconds when a person is without a mask  
- Play a **beep sound** when someone wears a mask  
- Start **automatic recording** after 10 seconds of continuous no-mask detection  
- Allow **manual recording toggle** using the `r` key  

---

## 🚀 Features  

- ✅ Face Detection (OpenCV DNN)  
- ✅ Mask Detection (color-based HSV heuristic)  
- ✅ Age & Gender Estimation (pre-trained Caffe models)  
- ✅ Object Detection (MobileNet SSD)  
- ✅ Info Board (Date, Time, Day, People & Objects summary)  
- ✅ Auto Snapshots (No Mask → every 10 seconds)  
- ✅ Auto Recording (after 10s continuous No Mask)  
- ✅ Manual Recording toggle (`r` key)  
- ✅ Beep alert when mask is detected  

---

## 🖼️ Screenshots  

### Main Detection Window ( Not Wearing Mask ) 
![Detection Screenshot](https://github.com/Ayushanupam7/Mask-Detction/blob/main/assets/Screenshot%202025-09-08%20195137.png) 

### Main Detection Window ( Wearing Mask ) 
![Detection Screenshot](https://github.com/Ayushanupam7/Mask-Detction/blob/main/assets/masked.png) 

### Info Board  
![Info Board](https://github.com/Ayushanupam7/Mask-Detction/blob/main/assets/info.png)  

*(Place your actual screenshots in the `screenshots/` folder inside the project repository.  
For example, you can capture images when a mask is detected or not detected, and also while recording is active.)*


---

## 📂 Project Structure

```text
📂 Smart-Mask-Detection/
 ┣ 📂 face_detector
 ┃ ┣ deploy.prototxt
 ┃ ┗ res10_300x300_ssd_iter_140000.caffemodel
 ┣ 📂 age_gender
 ┃ ┣ age_deploy.prototxt
 ┃ ┣ age_net.caffemodel
 ┃ ┣ gender_deploy.prototxt
 ┃ ┗ gender_net.caffemodel
 ┣ 📂 object_detector
 ┃ ┣ MobileNetSSD_deploy.prototxt
 ┃ ┗ MobileNetSSD_deploy.caffemodel
 ┣ 📂 recordings
 ┣ 📂 snapshots
 ┣ mask_detection_simple.py
 ┣ README.md
 ┗ requirements.txt
```

## ⚙️ Installation  

1. Clone the repository:  
   ```bash
   git clone https://github.com/Ayushanupam7/Mask-Detction.git
   cd mask-detection
   ```
2. Install dependencies:
```bash
pip install -r requirements.txt
```

Place the required Caffe model files inside their respective folders:

- face_detector/
- age_gender/
- object_detector/

## ▶️ Usage

Run the program:
```bash
python mask_detection_simple.py
```

## Keyboard Shortcuts:

- q → Quit
- r → Start/Stop manual recording
- s → Switch camera via menu
- c → Cycle through available cameras

## 📦 Requirements

- Python 3.7+
- OpenCV
- NumPy
- Pillow

## Install with:
```bash
pip install opencv-python numpy pillow
```

## 🛠️ Future Enhancements

- Improve mask detection using a deep learning model
- Add attendance logging system
- Enable remote monitoring (Flask/Streamlit dashboard)
  
## 📝 License

This project is released under the MIT License.
