/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package tv.danmaku.ijk.media.example.widget.media;

import android.app.Activity;
import android.util.Log;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.io.IOException;

/**
 * This classifier works with the Inception-v3 slim model.
 * It applies floating point inference rather than using a quantized model.
 */
public class ImageClassifierFloatBodypose extends ImageClassifier {
  private Mat mMat;
  /**
   * The inception net requires additional normalization of the used input.
   */
  //private static final int IMAGE_MEAN = 128;
  //private static final float IMAGE_STD = 128.0f;

  /**
   * An array to hold inference results, to be feed into Tensorflow Lite as outputs.
   * This isn't part of the super class, because we need a primitive array here.
   */
  //input shape float32[1,353,257,3]
  //outpu shape float32[1,23,17,17]
  private float[][][][] jointProbArray = null;

  /**
   * Initializes an {@code ImageClassifier}.
   *
   * @param activity
   */
  ImageClassifierFloatBodypose(Activity activity) throws IOException {
    super(activity);
    jointProbArray = new float[1][getHeatmapWidth()][getHeatmapHeight()][getNumJoint()];
  }

  @Override
  protected String getModelPath() {
    // you can download this file from
    // https://storage.googleapis.com/download.tensorflow.org/models/tflite/inception_v3_slim_2016_android_2017_11_10.zip
    //return "multi_person_mobilenet_v1_075_float.tflite";
    //return "model_cpm.tflite";
    return "model_h.tflite";
  }

  /*
  @Override
  protected String getLabelPath() {
    return "labels_imagenet_slim.txt";
  }
  */

  @Override
  protected int getImageSizeX() {
    return 192;
    //return 353;

  }


  @Override
  protected int getImageSizeY() {
    return 192;
    //return 257;
  }

  @Override
  protected int getNumBytesPerChannel() {
    // a 32bit float value requires 4 bytes
    return 4;
  }

  @Override
  protected void addPixelValue(int pixelValue) {

    /*
    imgData.putFloat((pixelValue & 0xFF) / 255.f);
    imgData.putFloat(((pixelValue >> 8) & 0xFF) / 255.f);
    imgData.putFloat(((pixelValue >> 16) & 0xFF) / 255.f);
    */

    imgData.putFloat(Float.valueOf(pixelValue & 0xFF));
    imgData.putFloat(Float.valueOf(pixelValue >> 8 & 0xFF));
    imgData.putFloat(Float.valueOf(pixelValue >> 16 & 0xFF));

  }

  @Override
  protected float getProbability(int index, int width, int height, int joint) {
    return jointProbArray[index][width][height][joint];
  }


  /*
  @Override
  protected void setProbability(int labelIndex, Number value) {
    labelProbArray[0][labelIndex] = value.floatValue();
  }
  */

  /*
  @Override
  protected float getNormalizedProbability(int labelIndex) {
    // TODO the following value isn't in [0,1] yet, but may be greater. Why?
    return getProbability(labelIndex);
  }
  */
  @Override
  protected void runInference() {
    tflite.run(imgData, jointProbArray);

    /*for (int i=0; i<96; i++){
      for (int j=0; j<96; j++) {
        for(int k=0; k<14; k++) {
            if(jointProbArray[0][i][j][1]>=0.01){
                Log.d("POSE Estimation", "value[" + i + "][" + j + "][" + k + "] = " + jointProbArray[0][i][j][1]);
            }
        }
      }
    }*/

      if (mPrintPointArray == null){
          mPrintPointArray = new float[2][14];
      }
      if (mMat == null){
          mMat = new Mat(96, 96, CvType.CV_32F);
      }
      float[] tempArray = new float[getHeatmapHeight() * getHeatmapWidth()];
      float[] outTempArray = new float[getHeatmapHeight() * getHeatmapWidth()];

      long st = System.currentTimeMillis();
      for (int i = 0; i < 14; i++) {
          int index = 0;
          for (int x = 0; x < 96; x++) {
              for (int y = 0; y < 96; y++) {
                  tempArray[index] = result[x * getHeatmapHeight() * 14 + y * 14 + i];
                  index++;
              }
          }

          mMat.put(0, 0, tempArray);
          Imgproc.GaussianBlur(mMat, mMat, new Size(5, 5), 0, 0);
          mMat.get(0, 0, outTempArray);

          float maxX = 0, maxY = 0;
          float max = 0;

          for (int x = 0; x < getOutputSizeX(); x++) {
              for (int y = 0; y < getOutputSizeY(); y++) {
                  float center = get(x, y, outTempArray);

                  if (center >= 0.01) {

                      if (center > max) {
                          max = center;
                          maxX = x;
                          maxY = y;
                      }
                  }
              }
          }

          if (max == 0) {
              mPrintPointArray = new float[2][14];
              return;
          }

          mPrintPointArray[0][i] = maxY;
          mPrintPointArray[1][i] = maxX;
      }
  }
}
