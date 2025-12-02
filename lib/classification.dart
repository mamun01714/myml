import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/foundation.dart';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:flutter/services.dart' show rootBundle;

class ClassificationModel {
  late Interpreter _interpreter;
  List<String> labels = []; // initialize empty to avoid LateInitializationError
  String classification = '';
  int _height = 160;
  int _width = 160;

  // Placeholder result for your model
  var result = List.filled(1 * 4, 0).reshape([1, 4]);

  /// Load the TFLite model
  Future<void> loadModel() async {
    try {
      _interpreter = await Interpreter.fromAsset('assets/model.tflite');
      if (kDebugMode) print("Classifier Loaded!");
    } catch (e) {
      if (kDebugMode) print("Classifier load error: $e");
    }
  }

  /// Load labels from assets
  Future<void> loadLabels() async {
    try {
      final String labelsData = await rootBundle.loadString('assets/labels.txt');
      labels = labelsData.split('\n').map((e) => e.trim()).toList();
      if (kDebugMode) print("Labels loaded: ${labels.length}");
    } catch (e) {
      labels = [];
      if (kDebugMode) print("Labels load error: $e");
    }
  }

  /// Preprocess image to model input size
  img.Image preprocessImage(Uint8List imageBytes) {
    img.Image image = img.decodeImage(imageBytes)!;
    return img.copyResize(image, width: _width, height: _height);
  }

  /// Convert preprocessed image to Float32 tensor
  Float32List imageToTensor(img.Image image) {
    var convertedBytes = Float32List(1 * _height * _width * 3);
    var buffer = Float32List.view(convertedBytes.buffer);
    int pixelIndex = 0;
    for (int y = 0; y < _height; y++) {
      for (int x = 0; x < _width; x++) {
        var pixel = image.getPixel(x, y);
        buffer[pixelIndex++] = img.getRed(pixel) / 1.0;
        buffer[pixelIndex++] = img.getGreen(pixel) / 1.0;
        buffer[pixelIndex++] = img.getBlue(pixel) / 1.0;
      }
    }
    return convertedBytes;
  }

  /// Classify image given its file path
  Future<void> classifyImage(String path) async {
    if (labels.isEmpty) {
      if (kDebugMode) print("Labels not loaded yet!");
      return;
    }

    final file = File(path);
    if (!await file.exists()) {
      if (kDebugMode) print("Image file does not exist: $path");
      return;
    }

    Uint8List imageBytes = await file.readAsBytes();
    img.Image processedImage = preprocessImage(imageBytes);
    Float32List inputTensor = imageToTensor(processedImage);
    var input = inputTensor.buffer.asUint8List();

    try {
      _interpreter.run(input, result);
    } catch (e) {
      if (kDebugMode) print("TFLite run error: $e");
    }

    var output = result[0];
    int maxIndex = 0;
    double maxValue = output[0].toDouble();

    for (int i = 1; i < output.length; i++) {
      if (output[i] > maxValue) {
        maxValue = output[i];
        maxIndex = i;
      }
    }

    classification = (maxIndex < labels.length) ? labels[maxIndex] : '';
    if (kDebugMode) print("Predicted: $classification");
  }

  /// Clear previous classification
  void clearClassification() {
    classification = '';
  }

  /// Dispose interpreter
  void dispose() {
    _interpreter.close();
  }
}
