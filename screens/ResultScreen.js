import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  Image,
  TouchableOpacity,
  Alert,
  ActivityIndicator,
  ScrollView,
  Dimensions,
  Share,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import * as FileSystem from 'expo-file-system';
import * as MediaLibrary from 'expo-media-library';

const { width, height } = Dimensions.get('window');

export default function ResultScreen({ route, navigation }) {
  const { imageUri, imageBase64, source } = route.params;
  const [isProcessing, setIsProcessing] = useState(true);
  const [detectionResult, setDetectionResult] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    processImage();
  }, []);

  const processImage = async () => {
    try {
      setIsProcessing(true);
      setError(null);

      // Create form data for the API
      const formData = new FormData();
      formData.append('image', {
        uri: imageUri,
        type: 'image/jpeg',
        name: 'mtg_card.jpg',
      });

      // Make API call to Flask backend
      const response = await fetch('http://192.168.35.21:5001/detect-card', {
        method: 'POST',
        body: formData,
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      
      if (result.success) {
        setDetectionResult(result);
      } else {
        throw new Error(result.error || 'Detection failed');
      }
    } catch (error) {
      console.error('Error processing image:', error);
      setError(error.message || 'Failed to process image. Please try again.');
    } finally {
      setIsProcessing(false);
    }
  };

  const retakePhoto = () => {
    if (source === 'camera') {
      navigation.navigate('Camera');
    } else {
      navigation.navigate('Gallery');
    }
  };

  const shareResult = async () => {
    try {
      if (detectionResult && detectionResult.visualization) {
        // Save the result image to media library first
        const base64Data = detectionResult.visualization;
        const tempUri = FileSystem.documentDirectory + 'mtg_result.jpg';
        
        await FileSystem.writeAsStringAsync(tempUri, base64Data, {
          encoding: FileSystem.EncodingType.Base64,
        });

        // Share the image
        await Share.share({
          url: tempUri,
          title: 'MTG Card Detection Result',
          message: 'Check out this MTG card detection result!',
        });
      }
    } catch (error) {
      console.error('Error sharing result:', error);
      Alert.alert('Error', 'Failed to share result. Please try again.');
    }
  };

  const saveToGallery = async () => {
    try {
      if (detectionResult && detectionResult.visualization) {
        const base64Data = detectionResult.visualization;
        const tempUri = FileSystem.documentDirectory + 'mtg_result.jpg';
        
        await FileSystem.writeAsStringAsync(tempUri, base64Data, {
          encoding: FileSystem.EncodingType.Base64,
        });

        const asset = await MediaLibrary.createAssetAsync(tempUri);
        await MediaLibrary.createAlbumAsync('MTG Card Detector', asset, false);
        
        Alert.alert('Success', 'Result saved to gallery!');
      }
    } catch (error) {
      console.error('Error saving to gallery:', error);
      Alert.alert('Error', 'Failed to save to gallery. Please try again.');
    }
  };

  if (isProcessing) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color="#3498db" />
          <Text style={styles.loadingText}>Analyzing MTG card...</Text>
          <Text style={styles.loadingSubtext}>
            This may take a few seconds
          </Text>
        </View>
      </SafeAreaView>
    );
  }

  if (error) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.errorContainer}>
          <Ionicons name="alert-circle" size={64} color="#e74c3c" />
          <Text style={styles.errorTitle}>Detection Failed</Text>
          <Text style={styles.errorMessage}>{error}</Text>
          <View style={styles.errorActions}>
            <TouchableOpacity
              style={[styles.actionButton, styles.retryButton]}
              onPress={processImage}
            >
              <Ionicons name="refresh" size={20} color="#fff" />
              <Text style={styles.actionButtonText}>Try Again</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={[styles.actionButton, styles.secondaryButton]}
              onPress={retakePhoto}
            >
              <Ionicons name="camera" size={20} color="#fff" />
              <Text style={styles.actionButtonText}>Retake Photo</Text>
            </TouchableOpacity>
          </View>
        </View>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView style={styles.scrollView} contentContainerStyle={styles.scrollContent}>
        {/* Header */}
        <View style={styles.header}>
          <Text style={styles.title}>Detection Complete!</Text>
          <Text style={styles.subtitle}>
            MTG card elements have been identified
          </Text>
        </View>

        {/* Result Image */}
        {detectionResult && detectionResult.visualization && (
          <View style={styles.resultContainer}>
            <Image
              source={{
                uri: `data:image/jpeg;base64,${detectionResult.visualization}`,
              }}
              style={styles.resultImage}
              resizeMode="contain"
            />
          </View>
        )}

        {/* Detection Details */}
        {detectionResult && (
          <View style={styles.detailsContainer}>
            <Text style={styles.detailsTitle}>Detection Results</Text>
            
            <View style={styles.detailRow}>
              <Text style={styles.detailLabel}>Card Style:</Text>
              <Text style={styles.detailValue}>{detectionResult.card_style}</Text>
            </View>
            
            <View style={styles.detailRow}>
              <Text style={styles.detailLabel}>Confidence:</Text>
              <Text style={styles.detailValue}>
                {(detectionResult.confidence * 100).toFixed(1)}%
              </Text>
            </View>

            {/* Extracted Text Results - Segmented by Frame */}
            {detectionResult.extracted_text && Object.keys(detectionResult.extracted_text).length > 0 && (
              <>
                <Text style={styles.sectionTitle}>📝 Text by Frame Segment:</Text>
                {Object.entries(detectionResult.extracted_text).map(([regionName, data]) => {
                  const hasText = data.text && data.text.trim().length > 0;
                  return (
                    <View key={regionName} style={[styles.extractedItem, !hasText && styles.extractedItemEmpty]}>
                      <View style={styles.segmentHeader}>
                        <Text style={styles.extractedLabel}>
                          🎯 {regionName.charAt(0).toUpperCase() + regionName.slice(1).replace('_', ' ')} Frame
                        </Text>
                        <Text style={styles.regionSize}>
                          {data.raw_region_size || 'Unknown size'}
                        </Text>
                      </View>
                      <Text style={styles.extractedText}>
                        {hasText ? `"${data.text}"` : 'No text detected in this frame segment'}
                      </Text>
                      <View style={styles.metadataRow}>
                        <Text style={styles.extractedConfidence}>
                          Confidence: {(data.confidence * 100).toFixed(1)}%
                        </Text>
                        {data.error && (
                          <Text style={styles.errorText}>
                            Error: {data.error}
                          </Text>
                        )}
                      </View>
                    </View>
                  );
                })}
              </>
            )}

            {/* Extracted Mana Results */}
            {detectionResult.extracted_mana && Object.keys(detectionResult.extracted_mana).length > 0 && (
              <>
                <Text style={styles.sectionTitle}>🔮 Mana Cost:</Text>
                {Object.entries(detectionResult.extracted_mana).map(([regionName, data]) => (
                  <View key={regionName} style={styles.extractedItem}>
                    <Text style={styles.extractedLabel}>Mana String:</Text>
                    <Text style={styles.manaString}>
                      {data.mana_string || 'No mana detected'}
                    </Text>
                    <Text style={styles.extractedConfidence}>
                      Confidence: {(data.confidence * 100).toFixed(1)}%
                    </Text>
                    <Text style={styles.detectionMethod}>
                      Method: {data.detection_method}
                    </Text>
                    {data.symbols && data.symbols.length > 0 && (
                      <View style={styles.symbolsContainer}>
                        <Text style={styles.symbolsLabel}>Symbols:</Text>
                        {data.symbols.map((symbol, index) => (
                          <Text key={index} style={styles.symbolItem}>
                            {symbol.symbol} ({(symbol.confidence * 100).toFixed(1)}%)
                          </Text>
                        ))}
                      </View>
                    )}
                  </View>
                ))}
              </>
            )}

            <Text style={styles.regionsTitle}>🎯 Detected Regions:</Text>
            {detectionResult.detected_regions.map((region, index) => (
              <View key={index} style={styles.regionItem}>
                <View
                  style={[
                    styles.regionColor,
                    { backgroundColor: `rgb(${region.color.join(',')})` },
                  ]}
                />
                <View style={styles.regionInfo}>
                  <Text style={styles.regionName}>
                    {region.name.charAt(0).toUpperCase() + region.name.slice(1)}
                  </Text>
                  <Text style={styles.regionConfidence}>
                    Confidence: {(region.confidence * 100).toFixed(1)}%
                  </Text>
                </View>
              </View>
            ))}
          </View>
        )}

        {/* Action Buttons */}
        <View style={styles.actionsContainer}>
          <TouchableOpacity
            style={[styles.actionButton, styles.shareButton]}
            onPress={shareResult}
          >
            <Ionicons name="share" size={20} color="#fff" />
            <Text style={styles.actionButtonText}>Share Result</Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={[styles.actionButton, styles.saveButton]}
            onPress={saveToGallery}
          >
            <Ionicons name="download" size={20} color="#fff" />
            <Text style={styles.actionButtonText}>Save to Gallery</Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={[styles.actionButton, styles.retakeButton]}
            onPress={retakePhoto}
          >
            <Ionicons name="camera" size={20} color="#fff" />
            <Text style={styles.actionButtonText}>Detect Another Card</Text>
          </TouchableOpacity>
        </View>

        {/* Legend */}
        <View style={styles.legendContainer}>
          <Text style={styles.legendTitle}>Color Legend:</Text>
          <View style={styles.legendItem}>
            <View style={[styles.legendColor, { backgroundColor: '#00ff00' }]} />
            <Text style={styles.legendText}>Title Area</Text>
          </View>
          <View style={styles.legendItem}>
            <View style={[styles.legendColor, { backgroundColor: '#ff0000' }]} />
            <Text style={styles.legendText}>Type Line</Text>
          </View>
          <View style={styles.legendItem}>
            <View style={[styles.legendColor, { backgroundColor: '#0000ff' }]} />
            <Text style={styles.legendText}>Text Box</Text>
          </View>
          <View style={styles.legendItem}>
            <View style={[styles.legendColor, { backgroundColor: '#ffff00' }]} />
            <Text style={styles.legendText}>Mana Cost</Text>
          </View>
          <View style={styles.legendItem}>
            <View style={[styles.legendColor, { backgroundColor: '#ff00ff' }]} />
            <Text style={styles.legendText}>Set Symbol</Text>
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
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 40,
  },
  loadingText: {
    fontSize: 20,
    fontWeight: '600',
    color: '#2c3e50',
    marginTop: 20,
    marginBottom: 8,
  },
  loadingSubtext: {
    fontSize: 16,
    color: '#7f8c8d',
    textAlign: 'center',
  },
  errorContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 40,
  },
  errorTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#e74c3c',
    marginTop: 20,
    marginBottom: 12,
  },
  errorMessage: {
    fontSize: 16,
    color: '#7f8c8d',
    textAlign: 'center',
    lineHeight: 22,
    marginBottom: 30,
  },
  errorActions: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    width: '100%',
  },
  header: {
    alignItems: 'center',
    marginBottom: 30,
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#27ae60',
    textAlign: 'center',
    marginBottom: 8,
  },
  subtitle: {
    fontSize: 16,
    color: '#7f8c8d',
    textAlign: 'center',
    lineHeight: 22,
  },
  resultContainer: {
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 16,
    marginBottom: 30,
    elevation: 4,
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.25,
    shadowRadius: 3.84,
  },
  resultImage: {
    width: '100%',
    height: 400,
    borderRadius: 12,
  },
  detailsContainer: {
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 20,
    marginBottom: 30,
    elevation: 2,
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 1,
    },
    shadowOpacity: 0.22,
    shadowRadius: 2.22,
  },
  detailsTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#2c3e50',
    marginBottom: 16,
  },
  detailRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  detailLabel: {
    fontSize: 16,
    color: '#7f8c8d',
  },
  detailValue: {
    fontSize: 16,
    fontWeight: '600',
    color: '#2c3e50',
  },
  regionsTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#2c3e50',
    marginTop: 16,
    marginBottom: 12,
  },
  regionItem: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 12,
  },
  regionColor: {
    width: 20,
    height: 20,
    borderRadius: 4,
    marginRight: 12,
  },
  regionInfo: {
    flex: 1,
  },
  regionName: {
    fontSize: 14,
    fontWeight: '600',
    color: '#2c3e50',
  },
  regionConfidence: {
    fontSize: 12,
    color: '#7f8c8d',
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
  shareButton: {
    backgroundColor: '#3498db',
  },
  saveButton: {
    backgroundColor: '#27ae60',
  },
  retakeButton: {
    backgroundColor: '#9b59b6',
  },
  secondaryButton: {
    backgroundColor: '#95a5a6',
  },
  actionButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
    marginLeft: 8,
  },
  legendContainer: {
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
  legendTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#2c3e50',
    marginBottom: 16,
  },
  legendItem: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 12,
  },
  legendColor: {
    width: 20,
    height: 20,
    borderRadius: 4,
    marginRight: 12,
  },
  legendText: {
    fontSize: 14,
    color: '#34495e',
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#2c3e50',
    marginTop: 20,
    marginBottom: 12,
  },
  extractedItem: {
    backgroundColor: '#f8f9fa',
    borderRadius: 8,
    padding: 12,
    marginBottom: 12,
    borderLeftWidth: 4,
    borderLeftColor: '#3498db',
  },
  extractedLabel: {
    fontSize: 14,
    fontWeight: '600',
    color: '#34495e',
    marginBottom: 4,
  },
  extractedText: {
    fontSize: 16,
    color: '#2c3e50',
    marginBottom: 4,
    fontFamily: 'monospace',
  },
  manaString: {
    fontSize: 18,
    color: '#8e44ad',
    marginBottom: 4,
    fontWeight: '600',
    fontFamily: 'monospace',
  },
  extractedConfidence: {
    fontSize: 12,
    color: '#7f8c8d',
    marginBottom: 4,
  },
  detectionMethod: {
    fontSize: 12,
    color: '#7f8c8d',
    fontStyle: 'italic',
  },
  symbolsContainer: {
    marginTop: 8,
    paddingTop: 8,
    borderTopWidth: 1,
    borderTopColor: '#ecf0f1',
  },
  symbolsLabel: {
    fontSize: 12,
    fontWeight: '600',
    color: '#34495e',
    marginBottom: 4,
  },
  symbolItem: {
    fontSize: 12,
    color: '#7f8c8d',
    marginBottom: 2,
    fontFamily: 'monospace',
  },
  extractedItemEmpty: {
    opacity: 0.6,
    borderLeftColor: '#bdc3c7',
  },
  segmentHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  regionSize: {
    fontSize: 10,
    color: '#95a5a6',
    fontFamily: 'monospace',
  },
  metadataRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginTop: 4,
  },
  errorText: {
    fontSize: 10,
    color: '#e74c3c',
    fontStyle: 'italic',
  },
});
