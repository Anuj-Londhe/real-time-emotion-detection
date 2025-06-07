# Real-time Emotion Detection

This project implements a **real-time emotion detection system** using deep learning and computer vision techniques. It detects faces from your webcam feed and classifies emotions into seven categories:

- Angry
- Disgust
- Fear
- Happy
- Sad
- Surprise
- Neutral

The model is based on a convolutional neural network trained on the FER-2013 dataset.

---

## Features

- Real-time face detection using OpenCV Haar cascades
- Emotion classification with a CNN model (mini_XCEPTION architecture)
- Displays emotion label and confidence score on detected faces
- Shows frames per second (FPS) for performance feedback

---

## Requirements

- Python 3.x
- TensorFlow
- OpenCV
- NumPy
