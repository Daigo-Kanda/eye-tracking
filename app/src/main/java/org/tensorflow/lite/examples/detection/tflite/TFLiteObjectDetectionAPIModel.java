/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

package org.tensorflow.lite.examples.detection.tflite;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.os.Trace;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.examples.detection.env.Logger;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Vector;

/**
 * Wrapper for frozen detection models trained using the Tensorflow Object Detection API:
 * github.com/tensorflow/models/tree/master/research/object_detection
 */
public class TFLiteObjectDetectionAPIModel implements Classifier {
    private static final Logger LOGGER = new Logger();

    // Only return this many results.
    // ディテクションの数
    private static final int NUM_DETECTIONS = 10;

    // Float model
    // 画像の平均と標準偏差
    private static final float IMAGE_MAX = 255.0f;

    // Float model
    // 画像の平均と標準偏差
    private static final float IMAGE_MEAN = 128.0f;
    private static final float IMAGE_STD = 128.0f;

    // Number of threads in the java app
    // Javaのスレッドの数
    private static final int NUM_THREADS = 4;

    // 量子化モードかどうか決定
    private boolean isModelQuantized;

    // Config values.
    // 画像のサイズ
    private int inputSize;

    // Pre-allocated buffers.
    private Vector<String> labels = new Vector<String>();
    private int[] intValues;
    // outputLocations: array of shape [Batchsize, NUM_DETECTIONS,4]
    // contains the location of detected boxes
    private float[][][] outputLocations;
    // outputClasses: array of shape [Batchsize, NUM_DETECTIONS]
    // contains the classes of detected boxes
    private float[][] outputClasses;
    // outputScores: array of shape [Batchsize, NUM_DETECTIONS]
    // contains the scores of detected boxes
    private float[][] outputScores;
    // numDetections: array of shape [Batchsize]
    // contains the number of detected boxes
    private float[] numDetections;

    //バイトバッファー
    private ByteBuffer imgData;

    // 画像のベクトル化
    private int[] intValues_face;
    private int[] intValues_right;
    private int[] intValues_left;
    private int[] intValues_grid;

    //バイトバッファー
    private ByteBuffer imgData_face;
    private ByteBuffer imgData_right;
    private ByteBuffer imgData_left;
    private ByteBuffer imgData_grid;

    // 推論を行う関数
    private Interpreter tfLite;

    private TFLiteObjectDetectionAPIModel() {
    }

    /**
     * Memory-map the model file in Assets.
     */
    private static MappedByteBuffer loadModelFile(AssetManager assets, String modelFilename)
            throws IOException {
        AssetFileDescriptor fileDescriptor = assets.openFd(modelFilename);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    /**
     * Initializes a native TensorFlow session for classifying images.
     *
     * @param assetManager  The asset manager to be used to load assets.
     * @param modelFilename The filepath of the model GraphDef protocol buffer.
     * @param labelFilename The filepath of label file for classes.
     * @param inputSize     The size of image input
     * @param isQuantized   Boolean representing model is quantized or not
     */
    public static Classifier create(
            final AssetManager assetManager,
            final String modelFilename,
            final String labelFilename,
            final int inputSize,
            final boolean isQuantized)
            throws IOException {
        final TFLiteObjectDetectionAPIModel d = new TFLiteObjectDetectionAPIModel();

        //ラベルのバイトを読み込む
        InputStream labelsInput = null;
        String actualFilename = labelFilename.split("file:///android_asset/")[1];
        labelsInput = assetManager.open(actualFilename);
        BufferedReader br = null;
        br = new BufferedReader(new InputStreamReader(labelsInput));
        String line;
        while ((line = br.readLine()) != null) {
            LOGGER.w(line);
            d.labels.add(line);
        }
        br.close();

        //画像のサイズを入力
        d.inputSize = inputSize;

        try {
            d.tfLite = new Interpreter(loadModelFile(assetManager, modelFilename));
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        //量子化の条件
        d.isModelQuantized = isQuantized;

        // Pre-allocate buffers.
        int numBytesPerChannel;
        if (isQuantized) {
            numBytesPerChannel = 1; // Quantized
        } else {
            numBytesPerChannel = 4; // Floating point
        }
        // imgDataに新しいダイレクトbyteバッファーを割り当てる
        d.imgData = ByteBuffer.allocateDirect(1 * d.inputSize * d.inputSize * 3 * numBytesPerChannel);
        // 効率を良くするためにbyte順序を変更
        d.imgData.order(ByteOrder.nativeOrder());
        // 画像の数
        d.intValues = new int[d.inputSize * d.inputSize];

        // imgDataに新しいダイレクトbyteバッファーを割り当てる
        d.imgData_face = ByteBuffer.allocateDirect(1 * d.inputSize * d.inputSize * 3 * numBytesPerChannel);
        // 効率を良くするためにbyte順序を変更
        d.imgData_face.order(ByteOrder.nativeOrder());
        // 画像の数
        d.intValues_face = new int[d.inputSize * d.inputSize];

        // imgDataに新しいダイレクトbyteバッファーを割り当てる
        d.imgData_right = ByteBuffer.allocateDirect(1 * d.inputSize * d.inputSize * 3 * numBytesPerChannel);
        // 効率を良くするためにbyte順序を変更
        d.imgData_right.order(ByteOrder.nativeOrder());
        // 画像の数
        d.intValues_right = new int[d.inputSize * d.inputSize];

        // imgDataに新しいダイレクトbyteバッファーを割り当てる
        d.imgData_left = ByteBuffer.allocateDirect(1 * d.inputSize * d.inputSize * 3 * numBytesPerChannel);
        // 効率を良くするためにbyte順序を変更
        d.imgData_left.order(ByteOrder.nativeOrder());
        // 画像の数
        d.intValues_left = new int[d.inputSize * d.inputSize];

        // imgDataに新しいダイレクトbyteバッファーを割り当てる
        d.imgData_grid = ByteBuffer.allocateDirect(1 * 25 * 25 * 1 * numBytesPerChannel);
        // 効率を良くするためにbyte順序を変更
        d.imgData_grid.order(ByteOrder.nativeOrder());
        // 画像の数
        d.intValues_grid = new int[25 * 25];

        // スレッドの数を指定
        d.tfLite.setNumThreads(NUM_THREADS);
        // 少し不明
        d.outputLocations = new float[1][NUM_DETECTIONS][4];
        d.outputClasses = new float[1][NUM_DETECTIONS];
        d.outputScores = new float[1][NUM_DETECTIONS];
        d.numDetections = new float[1];
        return d;
    }

    // 画像の認識
    @Override
    public List<Recognition> recognizeImage(final Bitmap bitmap) {
        // Log this method so that it can be analyzed with systrace.
        Trace.beginSection("recognizeImage");

        Trace.beginSection("preprocessBitmap");
        // Preprocess the image data from 0-255 int to normalized float based
        // on the provided parameters.
        // 画像を配列に変換する
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

        // マーク位置を0に戻す
        imgData.rewind();

        for (int i = 0; i < inputSize; ++i) {
            for (int j = 0; j < inputSize; ++j) {
                // 一つずつピクセルを取り出す
                int pixelValue = intValues[i * inputSize + j];

                //量子化か量子化じゃないかで条件分岐
                if (isModelQuantized) {
                    // Quantized model
                    imgData.put((byte) ((pixelValue >> 16) & 0xFF));
                    imgData.put((byte) ((pixelValue >> 8) & 0xFF));
                    imgData.put((byte) (pixelValue & 0xFF));
                } else { // Float model

                    // 画像をfloat型に変換
                    imgData.putFloat((((pixelValue >> 16) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                    imgData.putFloat((((pixelValue >> 8) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                    imgData.putFloat(((pixelValue & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                }
            }
        }
        Trace.endSection(); // preprocessBitmap

        // Copy the input data into TensorFlow.
        Trace.beginSection("feed");

        // 1 * 10 * 4
        outputLocations = new float[1][NUM_DETECTIONS][4];
        outputClasses = new float[1][NUM_DETECTIONS];
        outputScores = new float[1][NUM_DETECTIONS];
        numDetections = new float[1];

        // inputArrayにimgDataのBufferを代入（object型）
        Object[] inputArray = {imgData};
        // Map型　キーint型 数値Object型
        Map<Integer, Object> outputMap = new HashMap<>();
        // 各キーにそれぞれfloat型を代入
        outputMap.put(0, outputLocations);
        outputMap.put(1, outputClasses);
        outputMap.put(2, outputScores);
        outputMap.put(3, numDetections);
        Trace.endSection();

        // Run the inference call.
        Trace.beginSection("run");
        tfLite.runForMultipleInputsOutputs(inputArray, outputMap);
        Trace.endSection();

        // Show the best detections.
        // after scaling them back to the input size.
        final ArrayList<Recognition> recognitions = new ArrayList<>(NUM_DETECTIONS);

        // Detectionを返す．
        // 結果をここに格納している
        for (int i = 0; i < NUM_DETECTIONS; ++i) {
            final RectF detection =
                    new RectF(
                            outputLocations[0][i][1] * inputSize,
                            outputLocations[0][i][0] * inputSize,
                            outputLocations[0][i][3] * inputSize,
                            outputLocations[0][i][2] * inputSize);
            // SSD Mobilenet V1 Model assumes class 0 is background class
            // in label file and class labels start from 1 to number_of_classes+1,
            // while outputClasses correspond to class index from 0 to number_of_classes
            int labelOffset = 1;
            recognitions.add(
                    new Recognition(
                            "" + i,
                            labels.get((int) outputClasses[0][i] + labelOffset),
                            outputScores[0][i],
                            detection));
        }
        Trace.endSection(); // "recognizeImage"
        return recognitions;
    }

    @Override
    public float[][] recognizeImageEye(Bitmap face, Bitmap right_eye, Bitmap left_eye, Bitmap face_grid) {

        //Bitmap2Mat(face);

        // 推定値
        float[][] recognizedValues = new float[1][2];

        // 画像を配列に変換する
        face.getPixels(intValues_face, 0, face.getWidth(), 0, 0, face.getWidth(), face.getHeight());
        right_eye.getPixels(intValues_right, 0, right_eye.getWidth(), 0, 0, right_eye.getWidth(), right_eye.getHeight());
        left_eye.getPixels(intValues_left, 0, left_eye.getWidth(), 0, 0, left_eye.getWidth(), left_eye.getHeight());
        face_grid.getPixels(intValues_grid, 0, face_grid.getWidth(), 0, 0, face_grid.getWidth(), face_grid.getHeight());

        // マーク位置を0に戻す
        imgData_face.rewind();
        imgData_right.rewind();
        imgData_left.rewind();
        imgData_grid.rewind();


        int count = 0;
        float face_mean = 0, right_mean = 0, left_mean = 0;
        // 画像の平均．画像を全て255で割ったあと，平均を求める
        // すごい重い作業なのであとで改善する必要がある
        for (int i = 0; i < inputSize; ++i) {
            for (int j = 0; j < inputSize; ++j) {
                // 一つずつピクセルを取り出す
                int pixelValue_face = intValues_face[i * inputSize + j];
                int pixelValue_right = intValues_right[i * inputSize + j];
                int pixelValue_left = intValues_left[i * inputSize + j];

                // 画像をfloat型に変換
                face_mean += (((pixelValue_face >> 16) & 0xFF) / IMAGE_MAX + ((pixelValue_face >> 8) & 0xFF) / IMAGE_MAX + (pixelValue_face & 0xFF) / IMAGE_MAX);
                right_mean += (((pixelValue_right >> 16) & 0xFF) / IMAGE_MAX + ((pixelValue_right >> 8) & 0xFF) / IMAGE_MAX + (pixelValue_right & 0xFF) / IMAGE_MAX);
                left_mean += (((pixelValue_left >> 16) & 0xFF) / IMAGE_MAX + ((pixelValue_left >> 8) & 0xFF) / IMAGE_MAX + (pixelValue_left & 0xFF) / IMAGE_MAX);

                count++;
            }
        }

        face_mean = face_mean / (count * 3);
        right_mean = right_mean / (count * 3);
        left_mean = left_mean / (count * 3);

        for (int i = 0; i < inputSize; ++i) {
            for (int j = 0; j < inputSize; ++j) {
                // 一つずつピクセルを取り出す
                int pixelValue = intValues_face[i * inputSize + j];

                // 画像をfloat型に変換
                imgData_face.putFloat(((pixelValue & 0xFF) / IMAGE_MAX) - face_mean);
                imgData_face.putFloat((((pixelValue >> 8) & 0xFF) / IMAGE_MAX) - face_mean);
                imgData_face.putFloat((((pixelValue >> 16) & 0xFF) / IMAGE_MAX) - face_mean);

            }
        }

        for (int i = 0; i < inputSize; ++i) {
            for (int j = 0; j < inputSize; ++j) {
                // 一つずつピクセルを取り出す
                int pixelValue = intValues_right[i * inputSize + j];

                // 画像をfloat型に変換
                imgData_right.putFloat(((pixelValue & 0xFF) / IMAGE_MAX) - right_mean);
                imgData_right.putFloat((((pixelValue >> 8) & 0xFF) / IMAGE_MAX) - right_mean);
                imgData_right.putFloat((((pixelValue >> 16) & 0xFF) / IMAGE_MAX) - right_mean);

            }
        }

        for (int i = 0; i < inputSize; ++i) {
            for (int j = 0; j < inputSize; ++j) {
                // 一つずつピクセルを取り出す
                int pixelValue = intValues_left[i * inputSize + j];


                float test = (((pixelValue >> 16) & 0xFF) / IMAGE_MAX) - left_mean;
                // 画像をfloat型に変換
                imgData_left.putFloat(((pixelValue & 0xFF) / IMAGE_MAX) - left_mean);
                imgData_left.putFloat((((pixelValue >> 8) & 0xFF) / IMAGE_MAX) - left_mean);
                imgData_left.putFloat((((pixelValue >> 16) & 0xFF) / IMAGE_MAX) - left_mean);

            }
        }

        for (int i = 0; i < 25; ++i) {
            for (int j = 0; j < 25; ++j) {
                // 一つずつピクセルを取り出す
                int pixelValue = intValues_grid[i * 25 + j];
                if (pixelValue == -16777216) {
                    imgData_grid.putFloat(0.0f);
                }
                if (pixelValue == -1) {
                    imgData_grid.putFloat(1.0f);
                }

            }
        }

        // inputArrayにimgDataのBufferを代入（object型）
        Object[] inputArray = {imgData_right, imgData_left, imgData_face, imgData_grid};

        // Map型　キーint型 数値Object型
        Map<Integer, Object> outputMap = new HashMap<>();

        // 各キーにそれぞれfloat型を代入
        outputMap.put(0, recognizedValues);

        tfLite.runForMultipleInputsOutputs(inputArray, outputMap);

        return recognizedValues;
    }

    @Override
    public void enableStatLogging(final boolean logStats) {
    }

    @Override
    public String getStatString() {
        return "";
    }

    @Override
    public void close() {
    }

    public void setNumThreads(int num_threads) {
        if (tfLite != null) tfLite.setNumThreads(num_threads);
    }

    @Override
    public void setUseNNAPI(boolean isChecked) {
        if (tfLite != null) tfLite.setUseNNAPI(isChecked);
    }
}
