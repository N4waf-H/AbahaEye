# AbhaEye: Smart Traffic Accident Response System

AbhaEye is an open-source project aimed at building an end-to-end system that uses AI and computer vision to analyze CCTV feeds, detect traffic accidents in real-time, classify their severity, and automatically trigger the appropriate emergency response.

---

## The Problem
In busy cities, every second is critical following a traffic accident. Delays in reporting, inaccurate location pinpointing, and a lack of immediate situation assessment lead to wasted precious minutes, which can increase the severity of injuries and cause major traffic congestion.

---

## The Solution
This project offers an AI-driven solution to transform standard surveillance cameras into a smart, vigilant network. The system is designed to automate the following tasks:

1. **Instant Detection** – Identifies accidents the moment they occur.  
2. **Intelligent Assessment** – Classifies the accident's severity (e.g., minor, moderate, severe).  
3. **Automated Dispatch** – Sends an alert to the appropriate entity (e.g., traffic management, police, ambulance) based on the severity level.  

---

## Key Features
- **Vehicle Detection & Tracking** – Utilizes YOLOv8 and the SORT algorithm to track vehicles in the frame.  
- **Accident Detection Model** – A custom-trained classifier to distinguish between accidents and normal traffic flow.  
- **Severity Assessment** – Uses an AI-powered classification model (with optional OpenAI integration) to evaluate accident severity.  
- **Rule-Based Dispatch Logic** – Simple decision rules to determine the correct responding authority.  
- **Graphical User Interface (GUI)** – A PyQt6-based interface (`abhaeye_gui.py`) for real-time monitoring, visualization, and logs.  
- **Dataset Support** – Includes `accident_dataset/` for training and evaluation.  
- **Pipeline Automation** – `run_pipeline.py` integrates detection, severity classification, and alert logic in one workflow.  

---

## Tech Stack
- **Python 3.10+**  
- **OpenCV** – Video feed processing  
- **Ultralytics YOLOv8** – Object detection  
- **SORT** – Vehicle tracking  
- **NumPy & Pandas** – Data handling  
- **PyQt6** – User interface  
- **OpenAI API** – Accident severity analysis (optional)  

---

## Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/N4waf-H/AbhaEye.git
cd AbhaEye
pip install -r requirements.txt
