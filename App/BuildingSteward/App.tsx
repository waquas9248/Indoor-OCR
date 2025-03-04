import React, { useEffect, useState } from 'react';
import {
  SafeAreaView,
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Alert,
  Platform,
  NativeModules,
} from 'react-native';
import { requestMultiple, PERMISSIONS, RESULTS, openSettings } from 'react-native-permissions';
import { useCameraDevice, Camera } from 'react-native-vision-camera';
import TextRecognition from '@react-native-ml-kit/text-recognition'; //placeholder for the model
import Geolocation from '@react-native-community/geolocation';
import RNFS from 'react-native-fs';
import DeviceInfo from 'react-native-device-info';

function BuildingStewardUI() {
  const device = useCameraDevice('back');
  const [cameraRef, setCameraRef] = useState<Camera | null>(null);
  const [cameraPermission, setCameraPermission] = useState(false);
  const [locationPermission, setLocationPermission] = useState(false);
  const [capturedText, setCapturedText] = useState<string | null>(null);
  const [recording, setRecording] = useState(false);
  const [location, setLocation] = useState<{ latitude: number; longitude: number } | null>(null);
  const [locationError, setLocationError] = useState<string | null>(null);
  const [processingTime, setProcessingTime] = useState<number | null>(null);

  useEffect(() => {
    const initializeLocationTracking = async () => {
      await requestPermissions();

      const watchId = Geolocation.watchPosition(
        (position) => {
          const { latitude, longitude } = position.coords;
          setLocation({ latitude, longitude });
          setLocationError(null);
        },
        (error) => {
          console.error("Error watching location:", error);
          setLocationError(error.message);
          Alert.alert(
            "Location Error",
            "Unable to retrieve location. Ensure location services are enabled and set to high accuracy."
          );
        },
        {
          enableHighAccuracy: true,
          distanceFilter: 10,
          interval: 5000,
          fastestInterval: 2000,
        }
      );

      return () => Geolocation.clearWatch(watchId);
    };

    initializeLocationTracking();
  }, []);

  const requestPermissions = async () => {
    const statuses = await requestMultiple([
      PERMISSIONS.ANDROID.CAMERA,
      PERMISSIONS.ANDROID.ACCESS_FINE_LOCATION,
    ]);

    setCameraPermission(statuses[PERMISSIONS.ANDROID.CAMERA] === RESULTS.GRANTED);
    setLocationPermission(statuses[PERMISSIONS.ANDROID.ACCESS_FINE_LOCATION] === RESULTS.GRANTED);

    if (statuses[PERMISSIONS.ANDROID.CAMERA] !== RESULTS.GRANTED) {
      Alert.alert("Permissions Needed", "Camera permission is required.");
    }

    if (statuses[PERMISSIONS.ANDROID.ACCESS_FINE_LOCATION] !== RESULTS.GRANTED) {
      Alert.alert(
        "Location Permission Needed",
        "Precise location permission is required to use this feature.",
        [
          { text: "Go to Settings", onPress: () => openSettings() },
          { text: "Cancel", style: "cancel" },
        ]
      );
    }
  };

  const getBatteryLevel = async () => {
    const batteryLevel = await DeviceInfo.getBatteryLevel();
    return `${Math.round(batteryLevel * 100)}%`;
  };

  const getDeviceTemperature = () => {
    if (Platform.OS === 'android') {
      const BatteryManager = NativeModules.BatteryManager || NativeModules.BatteryModule;
      if (BatteryManager) {
        return BatteryManager.getBatteryTemperature();
      }
    }
    return null;
  };

  const getCPULoad = async () => {
    const cpuLoad = await DeviceInfo.getUsedMemory();
    return cpuLoad ? `${(cpuLoad * 100).toFixed(2)}%` : 'N/A';
  };

  interface MetricData {
    timestamp: string;
    text: string;
    processingTime: string;
    userDelay: string;
    location: string;
    batteryLevelStart: string;
    batteryLevelEnd: string;
    baselineTemperature: string;
    postTaskTemperature: string;
    temperatureChange: string;
    baselineCPULoad: string;
    postTaskCPULoad: string;
    cpuLoadChange: string;
  }

  const saveMetricsToFile = async (data: MetricData) => {
    const filePath = `${RNFS.DocumentDirectoryPath}/metrics_mlkit.json`;
    try {
      const fileExists = await RNFS.exists(filePath);
      const currentData = fileExists ? JSON.parse(await RNFS.readFile(filePath)) : [];

      currentData.push(data);
      await RNFS.writeFile(filePath, JSON.stringify(currentData, null, 2), 'utf8');
      console.log("Metrics saved successfully:", data);
    } catch (error) {
      console.error("Error saving metrics:", error);
    }
  };

  const handleCapture = async () => {
    try {
      if (cameraRef) {
        const startTime = Date.now();

        const batteryLevelStart = await getBatteryLevel();
        const baselineTemperature = getDeviceTemperature();
        const baselineCPULoad = await getCPULoad();

        const photo = await cameraRef.takePhoto();
        const photoUri = `file://${photo.path}`;

        const captureTime = Date.now();

        const result = await TextRecognition.recognize(photoUri); //placeholder for the model

        const endTime = Date.now();
        const processingTime = endTime - captureTime;
        const userDelay = endTime - startTime;

        const recognizedText = result?.text || "No text detected";
        const locationData = location ? `${location.latitude}, ${location.longitude}` : "N/A";

        const postTaskTemperature = getDeviceTemperature();
        const postTaskCPULoad = await getCPULoad();
        const batteryLevelEnd = await getBatteryLevel();

        const temperatureChange = postTaskTemperature && baselineTemperature
          ? `${(postTaskTemperature - baselineTemperature).toFixed(2)} °C`
          : 'N/A';
        const cpuLoadChange = postTaskCPULoad && baselineCPULoad
          ? `${(parseFloat(postTaskCPULoad) - parseFloat(baselineCPULoad)).toFixed(2)}%`
          : 'N/A';

        const metricData = {
          timestamp: new Date().toISOString(),
          text: recognizedText,
          processingTime: `${processingTime} ms`,
          userDelay: `${userDelay} ms`,
          location: locationData,
          batteryLevelStart,
          batteryLevelEnd,
          baselineTemperature: baselineTemperature ? `${baselineTemperature} °C` : 'N/A',
          postTaskTemperature: postTaskTemperature ? `${postTaskTemperature} °C` : 'N/A',
          temperatureChange,
          baselineCPULoad: baselineCPULoad,
          postTaskCPULoad: postTaskCPULoad,
          cpuLoadChange
        };

        saveMetricsToFile(metricData);

        Alert.alert(
          "Capture Info",
          `Text detected: ${recognizedText}\nProcessing Time: ${processingTime} ms\nUser Delay: ${userDelay} ms\nLocation: ${locationData}\nBattery Level Start: ${batteryLevelStart}\nBattery Level End: ${batteryLevelEnd}\nTemperature Change: ${temperatureChange}\nCPU Load Change: ${cpuLoadChange}`
        );

        setCapturedText(recognizedText);
        setProcessingTime(processingTime);
      }
    } catch (error) {
      console.error("Error recognizing text:", error);
      Alert.alert("Error", "Unable to capture text.");
    }
  };

  const handleDone = () => {
    setRecording(false);
    setCapturedText(null);
    setProcessingTime(null);
    Alert.alert("Recording Stopped", "Recording for this sign is complete.");
  };

  return (
    <SafeAreaView style={styles.container}>
      {device && (
        <Camera
          ref={(ref) => setCameraRef(ref)}
          style={styles.camera}
          device={device}
          isActive={true}
          photo={true}
        />
      )}
      <View style={styles.overlay}>
        <Text style={styles.statusText}>
          {recording ? "Recording..." : "Press 'Capture' when sign is in view."}
        </Text>
        <View style={styles.buttonContainer}>
          <TouchableOpacity style={styles.captureButton} onPress={handleCapture}>
            <Text style={styles.buttonText}>Capture</Text>
          </TouchableOpacity>
          <TouchableOpacity style={styles.doneButton} onPress={handleDone}>
            <Text style={styles.buttonText}>Done</Text>
          </TouchableOpacity>
        </View>
        {capturedText && (
          <View style={styles.capturedTextContainer}>
            <Text style={styles.capturedText}>Captured Text: {capturedText}</Text>
            <Text style={styles.capturedText}>Processing Time: {processingTime} ms</Text>
            {location && (
              <Text style={styles.capturedText}>
                Location: {location.latitude}, {location.longitude}
              </Text>
            )}
          </View>
        )}
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: 'black',
  },
  permissionText: {
    color: 'white',
    textAlign: 'center',
    fontSize: 18,
    margin: 20,
  },
  camera: {
    ...StyleSheet.absoluteFillObject,
  },
  overlay: {
    position: 'absolute',
    bottom: 50,
    width: '100%',
    alignItems: 'center',
  },
  statusText: {
    color: 'white',
    fontSize: 16,
    marginBottom: 20,
    textAlign: 'center',
  },
  buttonContainer: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    width: '80%',
  },
  captureButton: {
    backgroundColor: '#28A745',
    paddingVertical: 15,
    paddingHorizontal: 25,
    borderRadius: 8,
  },
  doneButton: {
    backgroundColor: '#DC3545',
    paddingVertical: 15,
    paddingHorizontal: 25,
    borderRadius: 8,
  },
  buttonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: 'bold',
  },
  capturedTextContainer: {
    marginTop: 20,
    paddingHorizontal: 20,
    paddingVertical: 10,
    backgroundColor: '#333',
    borderRadius: 8,
  },
  capturedText: {
    color: 'white',
    fontSize: 16,
    textAlign: 'center',
  },
});

export default BuildingStewardUI;
