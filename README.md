
# AbhaEye: Smart Traffic Accident Response System

AbhaEye is an open-source project aimed at building an end-to-end system that uses AI and computer vision to analyze CCTV feeds, detect traffic accidents in real-time, classify their severity, and automatically trigger the appropriate emergency response.


## The Problem


In busy cities, every second is critical following a traffic accident. Delays in reporting, inaccurate location pinpointing, and a lack of immediate situation assessment lead to wasted precious minutes, which can increase the severity of injuries and cause major traffic congestion.




## The Solution
This project offers an AI-driven solution to transform standard surveillance cameras into a smart, vigilant network. The system is designed to automate the following tasks:

1- Instant Detection: Identifies accidents the moment they occur.

2- Intelligent Assessment: Classifies the accident's severity (e.g., minor, moderate, severe).

3- Automated Dispatch: Sends an alert to the appropriate entity (e.g., traffic management, police, ambulance) based on the severity level.
## Key Features
- Vehicle Detection & Tracking: Utilizes YOLOv8 and the SORT algorithm to track every vehicle in the frame.

- Accident Detection Model: A custom-trained classification model to distinguish between accidents and normal traffic flow.

- Severity Assessment Model: A classification model to determine the severity of a detected accident.

- Rule-Based Dispatch Logic: Simple rules to determine the correct responding entity.
## Tech Stack

Python

OpenCV

Ultralytics YOLOv8

NumPy


## Project Status
This project is currently in the prototype development phase. Contributions and ideas are welcome to help improve and expand its capabilities.
