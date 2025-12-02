import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:image/image.dart' as img;
import 'package:image_picker/image_picker.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:flutter_vision/flutter_vision.dart';
import 'package:camera/camera.dart';

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

  late Interpreter _interpreter;
  late List<String> _labels;
  late FlutterVision _vision;
  late CameraController _cameraController;
  late Future<void> _initializeControllerFuture;

  // ignore: prefer_final_fields
  int _height = 160, _width = 160;
  String? _imagePath;
  bool _isCameraInitialized = false;

  var result = List.filled(1 * 4, 0).reshape([1, 4]);

  final ImagePicker _picker = ImagePicker();

  @override
  void initState() {
    super.initState();
    _loadModel();
    _loadLabels();
    _initializeCamera();
  }

  @override
  void dispose() {
    super.dispose();
    _interpreter.close();
    _vision.closeYoloModel();
    _cameraController.dispose();
  }

  _initializeCamera() {
    _cameraController = CameraController(
      widget.cameras.first,
      ResolutionPreset.high, // Use a higher resolution for better quality
    );
    _initializeControllerFuture = _cameraController.initialize().then((_) {
      setState(() {
        _isCameraInitialized = true;
        _imagePath = null;
        _classification = '';
        _reading = '';
      });
    });
  }

  _loadModel() async {
    _interpreter = await Interpreter.fromAsset('assets/model.tflite');
    if (kDebugMode) {
      print('Model Loaded!');
    }

    // Load and preprocess the image
    _vision = FlutterVision();
    await _vision.loadYoloModel(
        labels: 'assets/ylabels.txt',
        modelPath: 'assets/best_float32.tflite',
        modelVersion: "yolov8",
        quantization: false,
        numThreads: 1,
        useGpu: false);

    if (kDebugMode) print('Yolo Model Loaded!');
  }

  _loadLabels() async {
    final String labelsData = await rootBundle.loadString('assets/labels.txt');
    _labels = labelsData.split('\n');
  }

  Future<List<String>> loadLabels(String path) async {
    final String labelsData = await rootBundle.loadString(path);
    return labelsData.split('\n');
  }

  img.Image preprocessImage(Uint8List imageBytes, int height, int width) {
    img.Image image = img.decodeImage(imageBytes)!;
    img.Image resizedImage =
    img.copyResize(image, width: width, height: height);
    return resizedImage;
  }

  Float32List imageToTensor(img.Image image, int height, int width) {
    var convertedBytes =
    Float32List(1 * height * width * 3); // Example input size
    var buffer = Float32List.view(convertedBytes.buffer);
    int pixelIndex = 0;

    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        var pixel = image.getPixel(x, y);
        buffer[pixelIndex++] = img.getRed(pixel) / 1.0;
        buffer[pixelIndex++] = img.getGreen(pixel) / 1.0;
        buffer[pixelIndex++] = img.getBlue(pixel) / 1.0;
      }
    }
    return convertedBytes;
  }

  _detect(Uint8List imageBytes) async {
    var confThreshold = 0.3;
    var y_adjustment = 20;

    final result = await _vision.yoloOnImage(
        bytesList: imageBytes,
        imageHeight: 320,
        imageWidth: 480,
        iouThreshold: 0.2,
        confThreshold: confThreshold,
        classThreshold: 0.3);

    if (result.isEmpty) return;

    Map<int, String> digit = {};

    var y0 = result.map((e) => e['box'][1]).reduce((a, b) => a + b) / result.length;
    var y1 = result.map((e) => e['box'][3]).reduce((a, b) => a + b) / result.length;
    var y_avg = (y0 + y1) / 2.0;

    // Group by box[0]
    var grouped = <int, List<Map<String, dynamic>>>{};
    for (var result in result) {
      int key = (result['box'][0]).toInt();
      if (!grouped.containsKey(key)) {
        grouped[key] = [];
      }
      grouped[key]!.add(result);
    }

    // Find the entry with the highest box[4] in each group
    var highestConfidenceEntries = grouped.map((key, value) {
      var highest = value.reduce((a, b) => a['box'][4] > b['box'][4] ? a : b);
      return MapEntry(key, highest);
    });

    // Process the results
    _detectedBoxes.clear();
    List<Map<String, dynamic>> boxes = [];

    for (var r in highestConfidenceEntries.values.toList()) {
      if ((((r['box'][1] + r['box'][3]) / 2) - y_avg).abs() <= y_adjustment) {
        digit[(r['box'][0]).toInt()] = r['tag'];
        boxes.add(r);
      }
    }

    print(digit);

    var sortedEntries = digit.entries.toList();
    // Sort the list by keys
    sortedEntries.sort((a, b) => a.key.compareTo(b.key));

    // Convert the digits to a string
    final digitsStr = sortedEntries.map((e) => e.value).join('');

    print(digitsStr);

    setState(() {
      _reading = digitsStr;
      _detectedBoxes = boxes;
    });
  }

  _classifyImage(String? imgPath) async {
    if (imgPath == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text("Please select an image first")),
      );
      return;
    }

    // Load the image as bytes
    File imageFile = File(imgPath);
    Uint8List imageBytes = await imageFile.readAsBytes();

    // Load and preprocess the image
    img.Image processedImage =
    preprocessImage(imageBytes, _height, _width); // Example input size
    Float32List inputTensor = imageToTensor(processedImage, _height, _width);
    var input = inputTensor.buffer.asUint8List();

    _interpreter.run(input, result);

    if (kDebugMode) {
      print(result);
    }
    if (kDebugMode) {
      print(_labels);
    }

    var output = result[0];
    int maxIndex = 0;
    double maxValue = output[0];

    for (int i = 1; i < output.length; i++) {
      if (output[i] > maxValue) {
        maxValue = output[i];
        maxIndex = i;
      }
    }

    String predictedLabel = _labels[maxIndex];

    if (predictedLabel.trimLeft().trimRight() == 'Meter') {
      _detect(imageBytes);
    }

    setState(() {
      _classification = predictedLabel;
    });
  }

  Future<void> _pickImage() async {
    final XFile? image = await _picker.pickImage(source: ImageSource.gallery);
    if (image != null) {
      setState(() {
        _imagePath = image.path;
        _classification = ""; // Reset classification
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
        _classification = ""; // Reset classification
        _reading = "";
        _detectedBoxes.clear();
      });
    } catch (e) {
      print(e);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('DPDC Meter Reading'),
      ),
      body: SingleChildScrollView(
        child: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              SizedBox(height: 50,),
              Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: <Widget>[
                  ElevatedButton(
                    onPressed: _pickImage,
                    child: Text('Gallery'),
                  ),
                  SizedBox(width: 10),
                  ElevatedButton(
                    onPressed: _initializeCamera,
                    child: Text('Camera'),
                  ),
                  SizedBox(width: 10),
                  ElevatedButton(
                    onPressed: () async {
                      await _captureImage();
                    },
                    child: Text('Capture'),
                  ),
                ],
              ),
              SizedBox(height: 20),
              if (_imagePath != null)
                Container(
                  width: 480,
                  height: 255,
                  decoration: BoxDecoration(
                    border: Border.all(color: Colors.grey),
                  ),
                  child: Stack(
                    children: [
                      Image.file(
                        File(_imagePath!),
                        fit: BoxFit.cover, // Adjust this to BoxFit.contain if needed
                        width: double.infinity,
                        height: double.infinity,
                      ),
                      CustomPaint(
                        size: Size(480, 320),
                        painter: BoxPainter(_detectedBoxes, 480, 320),
                      ),
                    ],
                  ),
                )
              else if (_isCameraInitialized)
                Container(
                  width: 480,
                  height: 255,
                  decoration: BoxDecoration(
                    border: Border.all(color: Colors.grey),
                  ),
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
                  decoration: BoxDecoration(
                    border: Border.all(color: Colors.grey),
                  ),
                  child: Center(
                    child: Text("No image selected"),
                  ),
                ),
              SizedBox(height: 20),
              ElevatedButton(
                onPressed: () async {
                  await _classifyImage(_imagePath);
                },
                child: Text('Classify Image'),
              ),
              SizedBox(height: 30),
              Text('Classified as: $_classification', style: TextStyle(fontSize: 18, color: Colors.red, fontWeight: FontWeight.bold)),
              Text('Reading: $_reading', style: TextStyle(fontSize: 18, color: Colors.red, fontWeight: FontWeight.bold)),
            ],
          ),
        ),
      ),
    );
  }



}

class BoxPainter extends CustomPainter {
  final List<Map<String, dynamic>> boxes;
  final double imageWidth;
  final double imageHeight;

  BoxPainter(this.boxes, this.imageWidth, this.imageHeight);

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

    for (var box in boxes) {
      final left = box['box'][0].toDouble() * size.width / imageWidth;
      final top = box['box'][1].toDouble() * size.height / imageHeight;
      final right = box['box'][2].toDouble() * size.width / imageWidth;
      final bottom = box['box'][3].toDouble() * size.height / imageHeight;

      final rect = Rect.fromLTRB(left, top, right, bottom);
      canvas.drawRect(rect, paint);

      // Draw the tag above the box
      final tag = box['tag'];
      final textSpan = TextSpan(
        text: tag,
        style: TextStyle(color: Colors.red, fontSize: 12),
      );
      textPainter.text = textSpan;
      textPainter.layout();
      textPainter.paint(canvas, Offset(left, top - textPainter.height));
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) {
    return true;
  }
}