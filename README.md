# MTG Card Detector

A comprehensive Magic: The Gathering card detection system with a React Native frontend and Python backend. The app can detect MTG cards from camera photos or gallery images and overlay bounding boxes showing different card regions.

## Features

- **Camera Integration**: Take photos of MTG cards directly in the app
- **Gallery Selection**: Choose existing images from your device
- **Advanced Detection**: Uses enhanced computer vision algorithms to detect:
  - Card borders and boundaries
  - Title and mana cost areas
  - Type line and text box
  - Set symbol location
- **Real-time Processing**: Fast detection with confidence scores
- **Result Sharing**: Share detection results with others
- **Gallery Saving**: Save results to your photo library

## Screenshots

The app produces results similar to the reference image with colored bounding boxes:
- **Green**: Title area
- **Red**: Type line
- **Blue**: Text box
- **Yellow**: Mana cost
- **Magenta**: Set symbol

## Architecture

- **Frontend**: React Native with Expo
- **Backend**: Python Flask API
- **Computer Vision**: OpenCV with custom MTG detection algorithms
- **Communication**: HTTP API with image upload

## Prerequisites

- Python 3.8+
- Node.js 16+
- Expo CLI
- iOS Simulator or Android Emulator (or physical device)

## Installation & Setup

### 1. Backend Setup (Python Flask API)

```bash
# Navigate to project directory
cd mtg

# Install Python dependencies
pip install -r requirements.txt

# Start the Flask backend
python app.py
```

The backend will be available at `http://localhost:5000`

### 2. Frontend Setup (React Native Expo)

```bash
# Install Node.js dependencies
npm install

# Start Expo development server
npm start
```

### 3. Running the App

1. **Start the backend**: Run `python app.py` in one terminal
2. **Start the frontend**: Run `npm start` in another terminal
3. **Open in Expo Go**: Scan the QR code with Expo Go app on your device
4. **Configure API URL**: Update the API endpoint in `screens/ResultScreen.js` if needed

## API Endpoints

### POST /detect-card
Upload an image to detect MTG card regions.

**Request**: Multipart form data with image file
**Response**: JSON with detection results and visualization

### POST /detect-borders
Upload an image to detect just the card borders.

**Request**: Multipart form data with image file
**Response**: JSON with border coordinates and visualization

### GET /health
Health check endpoint.

## File Structure

```
mtg/
├── app.py                          # Flask backend API
├── requirements.txt                # Python dependencies
├── border_detection_enhanced.py   # Enhanced border detection
├── framers_enhanced.py            # MTG card region detection
├── package.json                   # Node.js dependencies
├── app.json                       # Expo configuration
├── App.js                         # Main React Native app
├── screens/                       # App screens
│   ├── HomeScreen.js             # Main landing page
│   ├── CameraScreen.js           # Camera interface
│   ├── GalleryScreen.js          # Image picker
│   └── ResultScreen.js           # Detection results
└── assets/                        # App assets
    └── mtg-logo.png              # App logo
```

## Usage

### Taking a Photo
1. Tap "Take Photo" on the home screen
2. Position the MTG card within the frame guide
3. Ensure good lighting and avoid glare
4. Tap the capture button
5. Wait for processing and view results

### Selecting from Gallery
1. Tap "Choose from Gallery" on the home screen
2. Select an existing photo from your device
3. Tap "Detect MTG Card"
4. Wait for processing and view results

### Understanding Results
- **Bounding Boxes**: Colored rectangles show detected regions
- **Confidence Scores**: Percentage indicating detection accuracy
- **Card Style**: Simple, normal, or complex card classification
- **Region Details**: Specific coordinates and confidence for each area

## Configuration

### Backend Configuration
- **Port**: Default 5000 (change in `app.py`)
- **Image Processing**: Adjust quality and size limits in the API endpoints
- **CORS**: Configure allowed origins in `app.py`

### Frontend Configuration
- **API URL**: Update endpoint in `screens/ResultScreen.js`
- **Image Quality**: Modify camera and picker settings in respective screens
- **UI Theme**: Customize colors and styles in the StyleSheet objects

## Troubleshooting

### Common Issues

1. **Backend Connection Error**
   - Ensure Flask server is running on port 5000
   - Check firewall settings
   - Verify network connectivity

2. **Camera Permission Denied**
   - Grant camera permissions in device settings
   - Restart the app after granting permissions

3. **Image Processing Fails**
   - Check image format (JPEG/PNG supported)
   - Ensure image size is reasonable (< 10MB)
   - Verify good lighting conditions

4. **Detection Accuracy Issues**
   - Improve lighting conditions
   - Avoid shadows and glare
   - Ensure card fills most of the frame
   - Keep camera steady during capture

### Debug Mode

Enable debug logging in the backend by setting:
```python
app.run(host='0.0.0.0', port=5000, debug=True)
```

## Development

### Adding New Detection Features
1. Extend the Python detection algorithms in `framers_enhanced.py`
2. Update the Flask API endpoints in `app.py`
3. Modify the frontend to display new results
4. Update the color legend and UI components

### Customizing the UI
- Modify StyleSheet objects in each screen component
- Update colors, fonts, and layouts
- Add new UI components as needed

### Performance Optimization
- Implement image caching
- Add loading states and progress indicators
- Optimize image processing pipeline
- Consider offline detection capabilities

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenCV community for computer vision tools
- Expo team for the excellent React Native framework
- Magic: The Gathering community for inspiration

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the code comments
3. Open an issue on GitHub
4. Contact the development team
