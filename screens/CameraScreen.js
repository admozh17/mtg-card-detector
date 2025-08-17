import React, { useState, useRef, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Alert,
  ActivityIndicator,
  Dimensions,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { CameraView, useCameraPermissions } from 'expo-camera';
import { Ionicons } from '@expo/vector-icons';
import * as ImageManipulator from 'expo-image-manipulator';
import * as FileSystem from 'expo-file-system';

const { width, height } = Dimensions.get('window');

export default function CameraScreen({ navigation }) {
  const [permission, requestPermission] = useCameraPermissions();
  const [facing, setFacing] = useState('back');
  const [isProcessing, setIsProcessing] = useState(false);
  const cameraRef = useRef(null);

  const takePicture = async () => {
    if (cameraRef.current) {
      try {
        setIsProcessing(true);
        const photo = await cameraRef.current.takePictureAsync({
          quality: 0.8,
          base64: true,
        });

        // Process the image
        await processImage(photo);
      } catch (error) {
        console.error('Error taking picture:', error);
        Alert.alert('Error', 'Failed to take picture. Please try again.');
        setIsProcessing(false);
      }
    }
  };

  const processImage = async (photo) => {
    try {
      // Calculate frame area for optimal card detection
      const frameWidth = width * 0.7;
      const frameHeight = frameWidth * 1.4; // MTG card ratio
      
      // Calculate crop area to match camera frame (center of image)
      const cropX = (photo.width - frameWidth * (photo.width / width)) / 2;
      const cropY = (photo.height - frameHeight * (photo.height / height)) / 2;
      const cropWidth = frameWidth * (photo.width / width);
      const cropHeight = frameHeight * (photo.height / height);
      
      // Crop to frame area and resize if needed
      const manipulateActions = [
        {
          crop: {
            originX: Math.max(0, cropX),
            originY: Math.max(0, cropY),
            width: Math.min(cropWidth, photo.width),
            height: Math.min(cropHeight, photo.height),
          }
        }
      ];
      
      // Add resize if the cropped image is still too large
      if (cropWidth > 1920 || cropHeight > 1920) {
        manipulateActions.push({ resize: { width: 1920, height: 1920 } });
      }
      
      const processedImage = await ImageManipulator.manipulateAsync(
        photo.uri,
        manipulateActions,
        { compress: 0.8, format: ImageManipulator.SaveFormat.JPEG, base64: true }
      );

      // Save to temporary file
      const tempUri = FileSystem.documentDirectory + 'temp_mtg_photo.jpg';
      await FileSystem.writeAsStringAsync(tempUri, processedImage.base64, {
        encoding: FileSystem.EncodingType.Base64,
      });

      // Navigate to result screen with the image
      navigation.navigate('Result', {
        imageUri: tempUri,
        imageBase64: processedImage.base64,
        source: 'camera',
      });
    } catch (error) {
      console.error('Error processing image:', error);
      Alert.alert('Error', 'Failed to process image. Please try again.');
    } finally {
      setIsProcessing(false);
    }
  };

  const flipCamera = () => {
    setFacing(current => (current === 'back' ? 'front' : 'back'));
  };

  if (!permission) {
    return (
      <View style={styles.container}>
        <ActivityIndicator size="large" color="#3498db" />
        <Text style={styles.permissionText}>Requesting camera permission...</Text>
      </View>
    );
  }

  if (!permission.granted) {
    return (
      <View style={styles.container}>
        <Ionicons name="camera-off" size={64} color="#e74c3c" />
        <Text style={styles.permissionText}>No access to camera</Text>
        <Text style={styles.permissionSubtext}>
          Please enable camera permissions in your device settings to use this feature.
        </Text>
        <TouchableOpacity
          style={styles.permissionButton}
          onPress={requestPermission}
        >
          <Text style={styles.permissionButtonText}>Grant Permission</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[styles.permissionButton, { backgroundColor: '#95a5a6', marginTop: 10 }]}
          onPress={() => navigation.goBack()}
        >
          <Text style={styles.permissionButtonText}>Go Back</Text>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      <CameraView style={styles.camera} facing={facing} ref={cameraRef}>
        {/* Camera Overlay */}
        <View style={styles.overlay}>
          {/* Card Frame Guide */}
          <View style={styles.cardFrame}>
            <View style={styles.cornerGuide} />
            <View style={[styles.cornerGuide, styles.topRight]} />
            <View style={[styles.cornerGuide, styles.bottomLeft]} />
            <View style={[styles.cornerGuide, styles.bottomRight]} />
            
            {/* Center crosshair for alignment */}
            <View style={styles.centerCrosshair}>
              <View style={styles.crosshairHorizontal} />
              <View style={styles.crosshairVertical} />
            </View>
          </View>

          {/* Instructions */}
          <View style={styles.instructionsContainer}>
            <Text style={styles.instructionsText}>
              Fill the green frame with your MTG card
            </Text>
            <Text style={styles.instructionsSubtext}>
              Card should touch all 4 corners • Avoid shadows & glare
            </Text>
          </View>
        </View>

        {/* Camera Controls */}
        <View style={styles.controlsContainer}>
          <TouchableOpacity
            style={styles.flipButton}
            onPress={flipCamera}
            disabled={isProcessing}
          >
            <Ionicons name="camera-reverse" size={24} color="#fff" />
          </TouchableOpacity>

          <TouchableOpacity
            style={[styles.captureButton, isProcessing && styles.captureButtonDisabled]}
            onPress={takePicture}
            disabled={isProcessing}
          >
            {isProcessing ? (
              <ActivityIndicator size="large" color="#fff" />
            ) : (
              <View style={styles.captureButtonInner} />
            )}
          </TouchableOpacity>

          <TouchableOpacity
            style={styles.backButton}
            onPress={() => navigation.goBack()}
            disabled={isProcessing}
          >
            <Ionicons name="close" size={24} color="#fff" />
          </TouchableOpacity>
        </View>
      </CameraView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
  },
  camera: {
    flex: 1,
  },
  overlay: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  cardFrame: {
    width: width * 0.7,
    height: width * 0.7 * 1.4, // MTG card aspect ratio (2.5" x 3.5")
    borderWidth: 3,
    borderColor: '#00ff00',
    borderStyle: 'solid',
    borderRadius: 12,
    position: 'relative',
    backgroundColor: 'rgba(0, 255, 0, 0.1)',
  },
  cornerGuide: {
    position: 'absolute',
    width: 30,
    height: 30,
    borderColor: '#00ff00',
    borderWidth: 4,
    backgroundColor: 'rgba(0, 255, 0, 0.3)',
  },
  topRight: {
    top: -4,
    right: -4,
    borderLeftWidth: 0,
    borderBottomWidth: 0,
  },
  bottomLeft: {
    bottom: -4,
    left: -4,
    borderRightWidth: 0,
    borderTopWidth: 0,
  },
  bottomRight: {
    bottom: -4,
    right: -4,
    borderLeftWidth: 0,
    borderTopWidth: 0,
  },
  centerCrosshair: {
    position: 'absolute',
    top: '50%',
    left: '50%',
    width: 40,
    height: 40,
    marginTop: -20,
    marginLeft: -20,
  },
  crosshairHorizontal: {
    position: 'absolute',
    top: '50%',
    left: 0,
    right: 0,
    height: 2,
    backgroundColor: '#00ff00',
    marginTop: -1,
  },
  crosshairVertical: {
    position: 'absolute',
    left: '50%',
    top: 0,
    bottom: 0,
    width: 2,
    backgroundColor: '#00ff00',
    marginLeft: -1,
  },
  instructionsContainer: {
    position: 'absolute',
    bottom: 120,
    alignItems: 'center',
    backgroundColor: 'rgba(0, 0, 0, 0.7)',
    paddingHorizontal: 20,
    paddingVertical: 10,
    borderRadius: 20,
  },
  instructionsText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
    textAlign: 'center',
  },
  instructionsSubtext: {
    color: '#bdc3c7',
    fontSize: 14,
    textAlign: 'center',
    marginTop: 4,
  },
  controlsContainer: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    flexDirection: 'row',
    justifyContent: 'space-around',
    alignItems: 'center',
    paddingBottom: 40,
    paddingHorizontal: 20,
  },
  flipButton: {
    width: 50,
    height: 50,
    borderRadius: 25,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  captureButton: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: '#3498db',
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 4,
    borderColor: '#fff',
  },
  captureButtonDisabled: {
    backgroundColor: '#95a5a6',
  },
  captureButtonInner: {
    width: 60,
    height: 60,
    borderRadius: 30,
    backgroundColor: '#fff',
  },
  backButton: {
    width: 50,
    height: 50,
    borderRadius: 25,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  permissionText: {
    fontSize: 18,
    color: '#2c3e50',
    textAlign: 'center',
    marginTop: 20,
  },
  permissionSubtext: {
    fontSize: 14,
    color: '#7f8c8d',
    textAlign: 'center',
    marginTop: 10,
    marginHorizontal: 40,
    lineHeight: 20,
  },
  permissionButton: {
    backgroundColor: '#3498db',
    paddingHorizontal: 30,
    paddingVertical: 15,
    borderRadius: 25,
    marginTop: 30,
  },
  permissionButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
});
