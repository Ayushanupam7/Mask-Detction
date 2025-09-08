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
- Save **snapshots** every 5 seconds when a person is without a mask  
- Play a **beep sound** when someone wears a mask  
- Start **automatic recording** after 5 seconds of continuous no-mask detection  
- Allow **manual recording toggle** using the `r` key  

---

## 🚀 Features  

- ✅ Face Detection (OpenCV DNN)  
- ✅ Mask Detection (color-based HSV heuristic)  
- ✅ Age & Gender Estimation (pre-trained Caffe models)  
- ✅ Object Detection (MobileNet SSD)  
- ✅ Info Board (Date, Time, Day, People & Objects summary)  
- ✅ Auto Snapshots (No Mask → every 5 seconds)  
- ✅ Auto Recording (after 5s continuous No Mask)  
- ✅ Manual Recording toggle (`r` key)  
- ✅ Beep alert when mask is detected  

---

## 🖼️ Screenshots  

### Main Detection Window  
![Detection Screenshot](screenshots/detection.png)  

### Info Board  
![Info Board](screenshots/infoboard.png)  

*(Add your actual screenshots in a `screenshots/` folder inside the project repo.)*  

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
 ┣ main.py
 ┣ README.md
 ┗ requirements.txt
```

## ⚙️ Installation  

1. Clone the repository:  
   ```bash
   git clone https://github.com/your-username/smart-mask-detection.git
   cd smart-mask-detection
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
python main.py
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
