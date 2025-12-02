import 'dart:io';
import 'dart:typed_data';
import 'dart:math';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:image/image.dart' as img;
import 'package:image_picker/image_picker.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:flutter_vision/flutter_vision.dart';
import 'package:camera/camera.dart';
import 'classification.dart'; // Your separate ClassificationModel

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final cameras = await availableCameras();
  runApp(MyApp(cameras: cameras));
}

class MyApp extends StatelessWidget {
  final List<CameraDescription> cameras;
  const MyApp({super.key, required this.cameras});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      home: ImageClassificationScreen(cameras: cameras),
    );
  }
}

class ImageClassificationScreen extends StatefulWidget {
  final List<CameraDescription> cameras;
  const ImageClassificationScreen({super.key, required this.cameras});

  @override
  _ImageClassificationScreenState createState() =>
      _ImageClassificationScreenState();
}

class _ImageClassificationScreenState extends State<ImageClassificationScreen> {
  String _classification = '';
  String _reading = '';
  List<Map<String, dynamic>> _detectedBoxes = [];

  late FlutterVision _vision;
  late CameraController _cameraController;
  late Future<void> _initializeControllerFuture;

  ClassificationModel classificationModel = ClassificationModel();

  int _height = 160, _width = 160;
  String? _imagePath;
  bool _isCameraInitialized = false;

  int _imageActualWidth = 480;
  int _imageActualHeight = 320;

  final ImagePicker _picker = ImagePicker();

  @override
  void initState() {
    super.initState();
    _initializeClassifier();
    _initializeCamera();
  }

  Future<void> _initializeClassifier() async {
    await classificationModel.loadModel();
    await classificationModel.loadLabels();
    _vision = FlutterVision();
    try {
      await _vision.loadYoloModel(
        labels: 'assets/ylabels.txt',
        modelPath: 'assets/best_float32.tflite',
        modelVersion: "yolov8",
        quantization: false,
        numThreads: 2,
        useGpu: true,
      );
      if (kDebugMode) print('Yolo Model Loaded!');
    } catch (e) {
      if (kDebugMode) print('Yolo load failed: $e');
    }
  }

  @override
  void dispose() {
    super.dispose();
    try {
      classificationModel.dispose();
    } catch (e) {}
    try {
      _vision.closeYoloModel();
    } catch (e) {}
    try {
      _cameraController.dispose();
    } catch (e) {}
  }

  _initializeCamera() {
    _cameraController = CameraController(
      widget.cameras.first,
      ResolutionPreset.high,
    );
    _initializeControllerFuture = _cameraController.initialize().then((_) {
      setState(() {
        _isCameraInitialized = true;
        _imagePath = null;
        _classification = '';
        _reading = '';
      });
    }).catchError((e) {
      if (kDebugMode) print("Camera init error: $e");
    });
  }

  Future<void> _classifyImage(String? imgPath) async {
    if (imgPath == null) return;

    if (classificationModel.labels.isEmpty) {
      if (kDebugMode) print("Labels not loaded yet!");
      return;
    }

    await classificationModel.classifyImage(imgPath);

    if (classificationModel.classification.trim() == 'Meter') {
      File imageFile = File(imgPath);
      Uint8List imageBytes = await imageFile.readAsBytes();
      await _detect(imageBytes);
    }

    setState(() {
      _classification = classificationModel.classification;
    });
  }

  _detect(Uint8List imageBytes) async {
    final double confThreshold = 0.2;
    if (_vision == null) return;

    try {
      final decoded = img.decodeImage(imageBytes)!;
      final int imgW = decoded.width;
      final int imgH = decoded.height;

      final result = await _vision.yoloOnImage(
        bytesList: imageBytes,
        imageHeight: imgH,
        imageWidth: imgW,
        confThreshold: confThreshold,
        classThreshold: 0.2,
        iouThreshold: 0.5,
      );

      if (result.isEmpty) return;

      List<double> centerY = result
          .map((r) => (((r['box'][1] as num).toDouble() + (r['box'][3] as num).toDouble()) / 2.0))
          .cast<double>()
          .toList();

      double rowY = centerY.reduce((a, b) => a + b) / centerY.length;
      final double tol = max(12.0, 0.03 * imgH);

      List<Map<String, dynamic>> rowDetections = result.where((r) {
        final cy = (((r['box'][1] as num).toDouble() + (r['box'][3] as num).toDouble()) / 2.0);
        return (cy - rowY).abs() <= tol;
      }).cast<Map<String, dynamic>>().toList();

      if (rowDetections.isEmpty) return;

      Map<int, Map<String, dynamic>> bestPerColumn = {};
      for (var r in rowDetections) {
        final double cx = (((r['box'][0] as num).toDouble() + (r['box'][2] as num).toDouble()) / 2.0);
        final int key = cx.round();
        final double conf = (r['box'].length > 4) ? (r['box'][4] as num).toDouble() : 0.0;

        if (!bestPerColumn.containsKey(key) ||
            conf > ((bestPerColumn[key]!['box'][4] as num).toDouble())) {
          bestPerColumn[key] = r;
        }
      }

      List<Map<String, dynamic>> sorted = bestPerColumn.values.toList()
        ..sort((a, b) {
          double ax = (((a['box'][0] as num).toDouble() + (a['box'][2] as num).toDouble()) / 2.0);
          double bx = (((b['box'][0] as num).toDouble() + (b['box'][2] as num).toDouble()) / 2.0);
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

      setState(() {
        _reading = finalBoxes.map((b) => b['tag']).join();
        _detectedBoxes = finalBoxes;
        _imageActualWidth = imgW;
        _imageActualHeight = imgH;
      });

      if (kDebugMode) {
        print('Detected: $_reading (img ${imgW}x${imgH}, boxes ${_detectedBoxes.length})');
      }
    } catch (e, st) {
      if (kDebugMode) {
        print('Error in _detect: $e');
        print(st);
      }
    }
  }

  Future<void> _pickImage() async {
    final XFile? image = await _picker.pickImage(source: ImageSource.gallery);
    if (image != null) {
      setState(() {
        _imagePath = image.path;
        _classification = "";
        _reading = "";
        _detectedBoxes.clear();
      });
    }
  }

  Future<void> _captureImage() async {
    try {
      await _initializeControllerFuture;
      final file = await _cameraController.takePicture();
      setState(() {
        _imagePath = file.path;
        _classification = "";
        _reading = "";
        _detectedBoxes.clear();
      });
    } catch (e) {
      if (kDebugMode) print('Capture error: $e');
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('DPDC Meter Reading')),
      body: SingleChildScrollView(
        child: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              SizedBox(height: 50),
              Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  ElevatedButton(onPressed: _pickImage, child: Text('Gallery')),
                  SizedBox(width: 10),
                  ElevatedButton(onPressed: _initializeCamera, child: Text('Camera')),
                  SizedBox(width: 10),
                  ElevatedButton(onPressed: _captureImage, child: Text('Capture')),
                ],
              ),
              SizedBox(height: 20),
              if (_imagePath != null)
                Container(
                  width: 480,
                  child: LayoutBuilder(builder: (context, constraints) {
                    final displayW = constraints.maxWidth;
                    final aspect = (_imageActualWidth > 0 && _imageActualHeight > 0)
                        ? _imageActualWidth / _imageActualHeight
                        : 480 / 320;
                    final displayH = displayW / aspect;
                    return SizedBox(
                      width: displayW,
                      height: displayH,
                      child: Stack(children: [
                        Positioned.fill(child: Image.file(File(_imagePath!), fit: BoxFit.contain)),
                        CustomPaint(
                          size: Size(displayW, displayH),
                          painter: BoxPainter(
                              _detectedBoxes,
                              _imageActualWidth.toDouble(),
                              _imageActualHeight.toDouble(),
                              displayW,
                              displayH),
                        ),
                      ]),
                    );
                  }),
                )
              else if (_isCameraInitialized)
                Container(
                  width: 480,
                  height: 255,
                  decoration: BoxDecoration(border: Border.all(color: Colors.grey)),
                  child: ClipRect(
                    child: OverflowBox(
                      alignment: Alignment.center,
                      child: FittedBox(
                        fit: BoxFit.cover,
                        child: Container(
                          width: _cameraController.value.previewSize!.height,
                          height: _cameraController.value.previewSize!.width,
                          child: CameraPreview(_cameraController),
                        ),
                      ),
                    ),
                  ),
                )
              else
                Container(
                  height: 320,
                  width: 480,
                  decoration: BoxDecoration(border: Border.all(color: Colors.grey)),
                  child: Center(child: Text("No image selected")),
                ),
              SizedBox(height: 20),
              ElevatedButton(
                onPressed: () => _classifyImage(_imagePath),
                child: Text('Classify Image'),
              ),
              SizedBox(height: 30),
              Text('Classified as: $_classification',
                  style: TextStyle(fontSize: 18, color: Colors.red, fontWeight: FontWeight.bold)),
              Text('Reading: $_reading',
                  style: TextStyle(fontSize: 18, color: Colors.red, fontWeight: FontWeight.bold)),
            ],
          ),
        ),
      ),
    );
  }
}

// ---------- BoxPainter ----------
class BoxPainter extends CustomPainter {
  final List<Map<String, dynamic>> boxes;
  final double imageWidth, imageHeight, displayWidth, displayHeight;

  BoxPainter(this.boxes, this.imageWidth, this.imageHeight, this.displayWidth, this.displayHeight);

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()..color = Colors.red..strokeWidth = 2.0..style = PaintingStyle.stroke;
    final textPainter = TextPainter(textAlign: TextAlign.left, textDirection: TextDirection.ltr);
    if (imageWidth <= 0 || imageHeight <= 0) return;
    final scaleX = displayWidth / imageWidth;
    final scaleY = displayHeight / imageHeight;

    for (var box in boxes) {
      final b = box['box'];
      final x0 = (b[0] as num).toDouble() * scaleX;
      final y0 = (b[1] as num).toDouble() * scaleY;
      final x1 = (b[2] as num).toDouble() * scaleX;
      final y1 = (b[3] as num).toDouble() * scaleY;

      canvas.drawRect(Rect.fromLTRB(x0, y0, x1, y1), paint);

      final tag = box['tag']?.toString() ?? '';
      final textSpan = TextSpan(text: tag, style: TextStyle(color: Colors.red, fontSize: 12));
      textPainter.text = textSpan;
      textPainter.layout();
      textPainter.paint(canvas, Offset(x0, max(0.0, y0 - textPainter.height - 2)));
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => true;
}
