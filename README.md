# Real-Time Sign Language Translator (Flask + TensorFlow)

This project is a real-time sign language recognition and translation system built using **Flask**, **MediaPipe**, **OpenCV**, and a deep learning model trained with **TensorFlow**. It captures hand gestures via webcam, processes them on the server, and displays live predictions through a web interface.

## Features

-  Real-time sign language recognition from webcam
-  Word prediction using a trained TensorFlow model
-  Hand tracking and landmark extraction using MediaPipe
-  Web-based interface built with HTML + Socket.IO
-  Text-to-speech support (reads translated sentences aloud)
-  Live statistics like session time and word count

---

## Project Structure

├── app2.py                  # Flask application entry point
├── main.py                  # Real-time camera handling & server events
├── model.py                 # TensorFlow LSTM model architecture
├── best_model_temp.keras    # Trained model file
├── data_collection.py       # Gesture recording & preprocessing
├── my_functions.py          # Helper utilities
├── actions.npy              # List of trained action labels
├── index.html               # Frontend UI
├── requirements.txt         # Required packages
└── README.md                # Project documentation


## Installation & Setup

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/sign-language-translator.git
cd sign-language-translator

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt



## Running the App

# Run the Flask app
python3 app2.py

	•	Open your browser and navigate to: http://localhost:5000


 How It Works
	1.	Click “Start Camera” on the UI.
	2.	Webcam feed is sent to the backend.
	3.	Each frame is processed using MediaPipe to extract hand landmarks.
	4.	Landmarks are passed to the trained TensorFlow model.
	5.	Predicted words are displayed and spoken in real time.


 Model Info
	•	Architecture: LSTM-based sequence model
	•	Trained on custom sign gesture dataset
	•	Input: Hand landmarks (x, y, z) per frame
	•	Output: Word or character prediction


Requirements
See requirements.txt for full list.


Developed with ❤️ by Özge Berfin Ümmetoglu
This project was built as part of a sign language translation initiative using computer vision and deep learning.
