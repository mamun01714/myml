// ==================== IMPORTS ====================
import 'dart:io';
import 'dart:typed_data';
import 'dart:math';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:image/image.dart' as img;
import 'package:flutter_vision/flutter_vision.dart';

// ==================== EXTRACTION MODEL ====================
class ExtractionModel {
  late FlutterVision _vision;
  String? _imagePath;
  List<Map<String, dynamic>> _detectedBoxes = [];
  int _imageActualWidth = 480;
  int _imageActualHeight = 320;
  String _reading = '';

  bool _isInitialized = false;

  // ====== GETTERS ======
  String get reading => _reading;
  List<Map<String, dynamic>> get detectedBoxes => _detectedBoxes;
  int get imageActualWidth => _imageActualWidth;
  int get imageActualHeight => _imageActualHeight;
  String? get imagePath => _imagePath;
  bool get isInitialized => _isInitialized;

  // ====== INITIALIZATION ======
  Future<void> loadModel() async {
    _vision = FlutterVision();

    try {
      await _vision.loadYoloModel(
        labels: 'assets/ylabels.txt',
        modelPath: 'assets/best_float32.tflite',
        modelVersion: "yolov8",
        quantization: false,
        numThreads: 4,
        useGpu: true,
      );
      if (kDebugMode) print('Yolo Model Loaded!');
      _isInitialized = true;
    } catch (e) {
      if (kDebugMode) print('YOLO load error: $e');
    }
  }

  void close() {
    try {
      _vision.closeYoloModel();
    } catch (e) {
      if (kDebugMode) print('Error closing YOLO: $e');
    }
  }

  // ====== IMAGE PATH ======
  void setImagePath(String path) {
    _imagePath = path;
    _detectedBoxes.clear();
    _reading = '';
  }

  void clearDetection() {
    _detectedBoxes.clear();
    _reading = '';
  }

  // ====== DETECTION ======
  Future<void> detect(String imagePath) async {
    if (!_isInitialized) return;

    try {
      File file = File(imagePath);
      Uint8List bytes = await file.readAsBytes();
      final decoded = img.decodeImage(bytes)!;
      final int imgW = decoded.width;
      final int imgH = decoded.height;

      final result = await _vision.yoloOnImage(
        bytesList: bytes,
        imageHeight: imgH,
        imageWidth: imgW,
        confThreshold: 0.2,
        classThreshold: 0.2,
        iouThreshold: 0.5,
      );

      if (result.isEmpty) return;

      // Center Y of each box
      List<double> centerY = result
          .map((r) => (((r['box'][1] as num).toDouble() +
          (r['box'][3] as num).toDouble()) /
          2.0))
          .cast<double>()
          .toList();

      double rowY = centerY.reduce((a, b) => a + b) / centerY.length;
      final double tol = max(12.0, 0.03 * imgH);

      List<Map<String, dynamic>> rowDetections = result.where((r) {
        final cy = (((r['box'][1] as num).toDouble() +
            (r['box'][3] as num).toDouble()) /
            2.0);
        return (cy - rowY).abs() <= tol;
      }).cast<Map<String, dynamic>>().toList();

      if (rowDetections.isEmpty) return;

      Map<int, Map<String, dynamic>> bestPerColumn = {};
      for (var r in rowDetections) {
        final double cx = (((r['box'][0] as num).toDouble() +
            (r['box'][2] as num).toDouble()) /
            2.0);
        final int key = cx.round();
        final double conf = (r['box'].length > 4)
            ? (r['box'][4] as num).toDouble()
            : 0.0;

        if (!bestPerColumn.containsKey(key) ||
            conf > ((bestPerColumn[key]!['box'][4] as num).toDouble())) {
          bestPerColumn[key] = r;
        }
      }

      List<Map<String, dynamic>> sorted = bestPerColumn.values.toList()
        ..sort((a, b) {
          double ax = (((a['box'][0] as num).toDouble() +
              (a['box'][2] as num).toDouble()) /
              2.0);
          double bx = (((b['box'][0] as num).toDouble() +
              (b['box'][2] as num).toDouble()) /
              2.0);
          return ax.compareTo(bx);
        });

      List<Map<String, dynamic>> finalBoxes = [];
      for (var r in sorted) {
        final b = r['box'];
        final double x0 = (b[0] as num).toDouble();
        final double y0 = (b[1] as num).toDouble();
        final double x1 = (b[2] as num).toDouble();
        final double y1 = (b[3] as num).toDouble();
        final String tag = r['tag']?.toString() ?? '';
        final double conf = (b.length > 4) ? (b[4] as num).toDouble() : 0.0;

        finalBoxes.add({
          'box': [x0, y0, x1, y1],
          'tag': tag,
          'conf': conf,
        });
      }

      _detectedBoxes = finalBoxes;
      _reading = finalBoxes.map((b) => b['tag']).join();
      _imageActualWidth = imgW;
      _imageActualHeight = imgH;

      if (kDebugMode) {
        print(
            'Detected: $_reading (img ${imgW}x${imgH}, boxes ${_detectedBoxes.length})');
      }
    } catch (e, st) {
      if (kDebugMode) {
        print('Detection error: $e');
        print(st);
      }
    }
  }

  // ====== BOX PAINTER ======
  CustomPainter boxPainter(double displayW, double displayH) {
    return _BoxPainter(
        _detectedBoxes, _imageActualWidth.toDouble(), _imageActualHeight.toDouble(), displayW, displayH);
  }
}

// ==================== PRIVATE BOX PAINTER ====================
class _BoxPainter extends CustomPainter {
  final List<Map<String, dynamic>> boxes;
  final double imageWidth;
  final double imageHeight;
  final double displayWidth;
  final double displayHeight;

  _BoxPainter(
      this.boxes, this.imageWidth, this.imageHeight, this.displayWidth, this.displayHeight);

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = Colors.red
      ..strokeWidth = 2.0
      ..style = PaintingStyle.stroke;

    final textPainter = TextPainter(
      textAlign: TextAlign.left,
      textDirection: TextDirection.ltr,
    );

    if (imageWidth <= 0 || imageHeight <= 0) return;

    final double scaleX = displayWidth / imageWidth;
    final double scaleY = displayHeight / imageHeight;

    for (var box in boxes) {
      final List<dynamic> b = box['box'];
      final double x0 = (b[0] as num).toDouble() * scaleX;
      final double y0 = (b[1] as num).toDouble() * scaleY;
      final double x1 = (b[2] as num).toDouble() * scaleX;
      final double y1 = (b[3] as num).toDouble() * scaleY;

      final rect = Rect.fromLTRB(x0, y0, x1, y1);
      canvas.drawRect(rect, paint);

      final tag = box['tag']?.toString() ?? '';
      final textSpan = TextSpan(
        text: tag,
        style: TextStyle(color: Colors.red, fontSize: 12),
      );
      textPainter.text = textSpan;
      textPainter.layout();
      final dx = x0;
      final dy = max(0.0, y0 - textPainter.height - 2);
      textPainter.paint(canvas, Offset(dx, dy));
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => true;
}
