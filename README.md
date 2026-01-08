## License Plate Number Recognition

An automated system that detects a vehicle’s license plate from an image or video and extracts only the license plate number using OpenCV and OCR.

### About the Project

This project automatically identifies a car’s license plate and converts it into text. It demonstrates how traditional image processing techniques can be applied to real-world problems such as parking systems, security, and vehicle monitoring.

### Features

- Detects license plates from images or videos
- Extracts and displays only the plate number
- Lightweight and easy to understand
- Web interface built using Streamlit

### Tech Stack

- Python
- OpenCV – Image processing & plate detection
- Tesseract OCR – Text extraction
- Streamlit – User interface

### How It Works

- User uploads a car image or video
- Image is preprocessed (grayscale, blur, edges)
- Plate region is detected using contours
- OCR extracts the alphanumeric characters
- Only the license plate number is displayed
