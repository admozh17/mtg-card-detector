import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Image,
  Alert,
  ActivityIndicator,
  ScrollView,
  Dimensions,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import * as ImagePicker from 'expo-image-picker';
import * as MediaLibrary from 'expo-media-library';
import * as ImageManipulator from 'expo-image-manipulator';
import * as FileSystem from 'expo-file-system';
import { Ionicons } from '@expo/vector-icons';

const { width, height } = Dimensions.get('window');

export default function GalleryScreen({ navigation }) {
  const [hasPermission, setHasPermission] = useState(null);
  const [selectedImage, setSelectedImage] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);

  useEffect(() => {
    (async () => {
      const { status } = await MediaLibrary.requestPermissionsAsync();
      setHasPermission(status === 'granted');
    })();
  }, []);

  const pickImage = async () => {
    try {
      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Images,
        allowsEditing: true,
        aspect: [1, 1.4], // MTG card aspect ratio
        quality: 0.8,
        base64: true,
      });

      if (!result.canceled && result.assets && result.assets[0]) {
        setSelectedImage(result.assets[0]);
      }
    } catch (error) {
      console.error('Error picking image:', error);
      Alert.alert('Error', 'Failed to pick image. Please try again.');
    }
  };

  const takeNewPhoto = () => {
    navigation.navigate('Camera');
  };

  const processSelectedImage = async () => {
    if (!selectedImage) return;

    try {
      setIsProcessing(true);

      // Resize image if it's too large
      let processedImage = selectedImage;
      if (selectedImage.width > 1920 || selectedImage.height > 1920) {
        const resizeResult = await ImageManipulator.manipulateAsync(
          selectedImage.uri,
          [{ resize: { width: 1920, height: 1920 } }],
          { compress: 0.8, format: ImageManipulator.SaveFormat.JPEG, base64: true }
        );
        processedImage = resizeResult;
      }

      // Save to temporary file
      const tempUri = FileSystem.documentDirectory + 'temp_mtg_gallery.jpg';
      await FileSystem.writeAsStringAsync(tempUri, processedImage.base64, {
        encoding: FileSystem.EncodingType.Base64,
      });

      // Navigate to result screen with the image
      navigation.navigate('Result', {
        imageUri: tempUri,
        imageBase64: processedImage.base64,
        source: 'gallery',
      });
    } catch (error) {
      console.error('Error processing image:', error);
      Alert.alert('Error', 'Failed to process image. Please try again.');
    } finally {
      setIsProcessing(false);
    }
  };

  const clearSelection = () => {
    setSelectedImage(null);
  };

  if (hasPermission === null) {
    return (
      <View style={styles.container}>
        <ActivityIndicator size="large" color="#3498db" />
        <Text style={styles.permissionText}>Requesting media library permission...</Text>
      </View>
    );
  }

  if (hasPermission === false) {
    return (
      <View style={styles.container}>
        <Ionicons name="folder-open" size={64} color="#e74c3c" />
        <Text style={styles.permissionText}>No access to media library</Text>
        <Text style={styles.permissionSubtext}>
          Please enable media library permissions in your device settings to use this feature.
        </Text>
        <TouchableOpacity
          style={styles.permissionButton}
          onPress={() => navigation.goBack()}
        >
          <Text style={styles.permissionButtonText}>Go Back</Text>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView style={styles.scrollView} contentContainerStyle={styles.scrollContent}>
        {/* Header */}
        <View style={styles.header}>
          <Text style={styles.title}>Choose MTG Card Image</Text>
          <Text style={styles.subtitle}>
            Select an image from your gallery or take a new photo
          </Text>
        </View>

        {/* Image Selection */}
        <View style={styles.selectionContainer}>
          {selectedImage ? (
            <View style={styles.selectedImageContainer}>
              <Image source={{ uri: selectedImage.uri }} style={styles.selectedImage} />
              <View style={styles.imageInfo}>
                <Text style={styles.imageInfoText}>
                  Selected: {selectedImage.width} × {selectedImage.height}
                </Text>
                <TouchableOpacity style={styles.clearButton} onPress={clearSelection}>
                  <Ionicons name="close-circle" size={24} color="#e74c3c" />
                </TouchableOpacity>
              </View>
            </View>
          ) : (
            <TouchableOpacity style={styles.pickImageButton} onPress={pickImage}>
              <Ionicons name="images" size={48} color="#9b59b6" />
              <Text style={styles.pickImageText}>Tap to select image</Text>
              <Text style={styles.pickImageSubtext}>
                Choose from your photo library
              </Text>
            </TouchableOpacity>
          )}
        </View>

        {/* Action Buttons */}
        <View style={styles.actionsContainer}>
          {selectedImage ? (
            <TouchableOpacity
              style={[styles.actionButton, styles.processButton]}
              onPress={processSelectedImage}
              disabled={isProcessing}
            >
              {isProcessing ? (
                <ActivityIndicator size="small" color="#fff" />
              ) : (
                <Ionicons name="search" size={20} color="#fff" />
              )}
              <Text style={styles.actionButtonText}>
                {isProcessing ? 'Processing...' : 'Detect MTG Card'}
              </Text>
            </TouchableOpacity>
          ) : (
            <TouchableOpacity
              style={[styles.actionButton, styles.cameraButton]}
              onPress={takeNewPhoto}
            >
              <Ionicons name="camera" size={20} color="#fff" />
              <Text style={styles.actionButtonText}>Take New Photo</Text>
            </TouchableOpacity>
          )}

          <TouchableOpacity
            style={[styles.actionButton, styles.secondaryButton]}
            onPress={pickImage}
          >
            <Ionicons name="folder-open" size={20} color="#9b59b6" />
            <Text style={[styles.actionButtonText, styles.secondaryButtonText]}>
              {selectedImage ? 'Choose Different Image' : 'Browse Gallery'}
            </Text>
          </TouchableOpacity>
        </View>

        {/* Tips */}
        <View style={styles.tipsContainer}>
          <Text style={styles.tipsTitle}>Tips for best results:</Text>
          <View style={styles.tipItem}>
            <Ionicons name="checkmark-circle" size={16} color="#27ae60" />
            <Text style={styles.tipText}>Ensure good lighting on the card</Text>
          </View>
          <View style={styles.tipItem}>
            <Ionicons name="checkmark-circle" size={16} color="#27ae60" />
            <Text style={styles.tipText}>Avoid shadows and glare</Text>
          </View>
          <View style={styles.tipItem}>
            <Ionicons name="checkmark-circle" size={16} color="#27ae60" />
            <Text style={styles.tipText}>Position card to fill most of the frame</Text>
          </View>
          <View style={styles.tipItem}>
            <Ionicons name="checkmark-circle" size={16} color="#27ae60" />
            <Text style={styles.tipText}>Keep camera steady when taking photos</Text>
          </View>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f8f9fa',
  },
  scrollView: {
    flex: 1,
  },
  scrollContent: {
    padding: 20,
  },
  header: {
    alignItems: 'center',
    marginBottom: 30,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#2c3e50',
    textAlign: 'center',
    marginBottom: 8,
  },
  subtitle: {
    fontSize: 16,
    color: '#7f8c8d',
    textAlign: 'center',
    lineHeight: 22,
  },
  selectionContainer: {
    marginBottom: 30,
  },
  pickImageButton: {
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 40,
    alignItems: 'center',
    borderWidth: 2,
    borderColor: '#e0e0e0',
    borderStyle: 'dashed',
    elevation: 2,
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 1,
    },
    shadowOpacity: 0.22,
    shadowRadius: 2.22,
  },
  pickImageText: {
    fontSize: 18,
    fontWeight: '600',
    color: '#9b59b6',
    marginTop: 16,
    marginBottom: 8,
  },
  pickImageSubtext: {
    fontSize: 14,
    color: '#95a5a6',
    textAlign: 'center',
  },
  selectedImageContainer: {
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 16,
    elevation: 4,
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.25,
    shadowRadius: 3.84,
  },
  selectedImage: {
    width: '100%',
    height: 300,
    borderRadius: 12,
    marginBottom: 12,
  },
  imageInfo: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  imageInfoText: {
    fontSize: 14,
    color: '#7f8c8d',
  },
  clearButton: {
    padding: 4,
  },
  actionsContainer: {
    marginBottom: 30,
  },
  actionButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 16,
    paddingHorizontal: 24,
    borderRadius: 12,
    marginBottom: 12,
    elevation: 2,
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 1,
    },
    shadowOpacity: 0.22,
    shadowRadius: 2.22,
  },
  processButton: {
    backgroundColor: '#27ae60',
  },
  cameraButton: {
    backgroundColor: '#3498db',
  },
  secondaryButton: {
    backgroundColor: '#fff',
    borderWidth: 2,
    borderColor: '#9b59b6',
  },
  actionButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
    marginLeft: 8,
  },
  secondaryButtonText: {
    color: '#9b59b6',
  },
  tipsContainer: {
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 20,
    elevation: 2,
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 1,
    },
    shadowOpacity: 0.22,
    shadowRadius: 2.22,
  },
  tipsTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#2c3e50',
    marginBottom: 16,
  },
  tipItem: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 12,
  },
  tipText: {
    fontSize: 14,
    color: '#34495e',
    marginLeft: 12,
    flex: 1,
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
