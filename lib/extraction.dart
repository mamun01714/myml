import 'dart:io';
import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter/cupertino.dart';
import 'package:flutter/services.dart';

import 'package:image/image.dart' as img;
import 'package:tflite_v2/tflite_v2.dart';


class Yolo {

  Future loadModel() async {
    Tflite.close();
    try {
      String res;

      res = (await Tflite.loadModel(
        model: "assets/best_float32.tflite",
        labels: "assets/ylabels.txt",
        // useGpuDelegate: true,
      ))!;

      print(res);
    } on PlatformException {
      print('Failed to load model.');
    }
  }

  Future yolov2Tiny(File image) async {
    int startTime = new DateTime.now().millisecondsSinceEpoch;
    var recognitions = await Tflite.detectObjectOnImage(
      path: image.path,
      model: "YOLO",
      threshold: 0.3,
      imageMean: 0.0,
      imageStd: 255.0,
      numResultsPerClass: 1,
    );
    print(recognitions);
    int endTime = new DateTime.now().millisecondsSinceEpoch;
    print("Inference took ${endTime - startTime}ms");
  }

}