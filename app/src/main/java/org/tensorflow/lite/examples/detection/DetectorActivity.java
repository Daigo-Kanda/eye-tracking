/*
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.detection;

import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Rect;
import android.graphics.Typeface;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.Environment;
import android.os.SystemClock;
import android.util.Log;
import android.util.Size;
import android.util.TypedValue;
import android.widget.Toast;

import androidx.annotation.NonNull;

import com.google.android.gms.tasks.OnFailureListener;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.android.gms.tasks.Task;
import com.google.firebase.ml.vision.FirebaseVision;
import com.google.firebase.ml.vision.common.FirebaseVisionImage;
import com.google.firebase.ml.vision.common.FirebaseVisionPoint;
import com.google.firebase.ml.vision.face.FirebaseVisionFace;
import com.google.firebase.ml.vision.face.FirebaseVisionFaceContour;
import com.google.firebase.ml.vision.face.FirebaseVisionFaceDetector;
import com.google.firebase.ml.vision.face.FirebaseVisionFaceDetectorOptions;

import org.tensorflow.lite.examples.detection.customview.OverlayView;
import org.tensorflow.lite.examples.detection.customview.OverlayView.DrawCallback;
import org.tensorflow.lite.examples.detection.env.BorderedText;
import org.tensorflow.lite.examples.detection.env.ImageUtils;
import org.tensorflow.lite.examples.detection.env.Logger;
import org.tensorflow.lite.examples.detection.tflite.Classifier;
import org.tensorflow.lite.examples.detection.tflite.TFLiteObjectDetectionAPIModel;
import org.tensorflow.lite.examples.detection.tracking.MultiBoxTracker;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.List;

/**
 * An activity that uses a TensorFlowMultiBoxDetector and ObjectTracker to detect and then track
 * objects.
 */
public class DetectorActivity extends CameraActivity implements OnImageAvailableListener {
    private static final Logger LOGGER = new Logger();

    // Configuration values for the prepackaged SSD model.
    private static final int TF_OD_API_INPUT_SIZE = 64;
    private static final int cropSizex = 2560;
    private static final int cropSizey = 1600;

    private static final boolean TF_OD_API_IS_QUANTIZED = false;
    private static final String TF_OD_API_MODEL_FILE = "converted_model.tflite";
    private static final String TF_OD_API_LABELS_FILE = "file:///android_asset/labelmap.txt";
    private static final DetectorMode MODE = DetectorMode.TF_OD_API;
    // Minimum detection confidence to track a detection.
    private static final float MINIMUM_CONFIDENCE_TF_OD_API = 0.5f;
    private static final boolean MAINTAIN_ASPECT = false;
    private static final Size DESIRED_PREVIEW_SIZE = new Size(2560, 1600);
    private static final boolean SAVE_PREVIEW_BITMAP = false;
    private static final float TEXT_SIZE_DIP = 10;
    OverlayView trackingOverlay;
    private Integer sensorOrientation;

    private Classifier detector;

    private long lastProcessingTimeMs;
    private Bitmap rgbFrameBitmap = null;
    private Bitmap croppedBitmap = null;
    private Bitmap cropCopyBitmap = null;

    private boolean computingDetection = false;

    private long timestamp = 0;

    private Matrix frameToCropTransform;
    private Matrix cropToFrameTransform;

    private MultiBoxTracker tracker;

    private BorderedText borderedText;


    /**
     * 自分で定義したフィールド
     */

    private int scaledSize = 64;

    /**
     * 顔検出の下準備
     */
    // Real-time contour detection of multiple faces
    private FirebaseVisionFaceDetectorOptions realTimeOpts =
            new FirebaseVisionFaceDetectorOptions.Builder()
                    .setContourMode(FirebaseVisionFaceDetectorOptions.ALL_CONTOURS)
                    .build();

    @Override
    public void onPreviewSizeChosen(final Size size, final int rotation) {
        final float textSizePx =
                TypedValue.applyDimension(
                        TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
        borderedText = new BorderedText(textSizePx);
        borderedText.setTypeface(Typeface.MONOSPACE);

        tracker = new MultiBoxTracker(this);

        int cropSize = TF_OD_API_INPUT_SIZE;

        try {
            detector =
                    TFLiteObjectDetectionAPIModel.create(
                            getAssets(),
                            TF_OD_API_MODEL_FILE,
                            TF_OD_API_LABELS_FILE,
                            TF_OD_API_INPUT_SIZE,
                            TF_OD_API_IS_QUANTIZED);
            cropSize = TF_OD_API_INPUT_SIZE;
        } catch (final IOException e) {
            e.printStackTrace();
            LOGGER.e(e, "Exception initializing classifier!");
            Toast toast =
                    Toast.makeText(
                            getApplicationContext(), "Classifier could not be initialized", Toast.LENGTH_SHORT);
            toast.show();
            finish();
        }

        previewWidth = size.getWidth();
        previewHeight = size.getHeight();

        sensorOrientation = rotation - getScreenOrientation();
        LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);

        LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
        rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);
        croppedBitmap = Bitmap.createBitmap(cropSizex, cropSizey, Config.ARGB_8888);

        frameToCropTransform =
                ImageUtils.getTransformationMatrix(
                        previewWidth, previewHeight,
                        cropSizex, cropSizey,
                        sensorOrientation, MAINTAIN_ASPECT);

        cropToFrameTransform = new Matrix();
        frameToCropTransform.invert(cropToFrameTransform);

        trackingOverlay = (OverlayView) findViewById(R.id.tracking_overlay);
        trackingOverlay.addCallback(
                new DrawCallback() {
                    @Override
                    public void drawCallback(final Canvas canvas) {
                        tracker.draw(canvas);
                        if (isDebug()) {
                            tracker.drawDebug(canvas);
                        }
                    }
                });

        tracker.setFrameConfiguration(previewWidth, previewHeight, sensorOrientation);
    }

    @Override
    protected void processImage() {

        ++timestamp;
        final long currTimestamp = timestamp;
        trackingOverlay.postInvalidate();

        // No mutex needed as this method is not reentrant.
        if (computingDetection) {
            readyForNextImage();
            return;
        }
        computingDetection = true;
        LOGGER.i("Preparing image " + currTimestamp + " for detection in bg thread.");

        rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);

        readyForNextImage();

        final Canvas canvas = new Canvas(croppedBitmap);
        // ここで情報を移動している
        canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);
        // For examining the actual TF input.
        if (SAVE_PREVIEW_BITMAP) {
            ImageUtils.saveBitmap(croppedBitmap);
        }

        runInBackground(
                new Runnable() {
                    @Override
                    public void run() {

                        Log.v("MLKit", "aaaaaaa");

                        LOGGER.i("Running detection on image " + currTimestamp);
                        final long startTime = SystemClock.uptimeMillis();

                        //顔検出のための処理
                        FirebaseVisionImage image = FirebaseVisionImage.fromBitmap(croppedBitmap);

                        Bitmap bitmap = croppedBitmap.copy(Config.ARGB_8888, true);

                        FirebaseVisionFaceDetector detectorFace = FirebaseVision.getInstance()
                                .getVisionFaceDetector(realTimeOpts);

                        Task<List<FirebaseVisionFace>> resultFace =
                                detectorFace.detectInImage(image)
                                        .addOnSuccessListener(
                                                new OnSuccessListener<List<FirebaseVisionFace>>() {

                                                    // 認識結果がゼロの場合でも呼ばれる
                                                    @Override
                                                    public void onSuccess(List<FirebaseVisionFace> faces) {

                                                        // 最初の一人のみ顔を検出する．
                                                        if (faces.size() != 0) {
                                                            // 顔の境界
                                                            Rect bounds = faces.get(0).getBoundingBox();
                                                            // If landmark detection was enabled (mouth, ears, eyes, cheeks, and
                                                            // nose available):
                                                            // ランドマーク 右目
                                                            List<FirebaseVisionPoint> rightEyeContour =
                                                                    faces.get(0).getContour(FirebaseVisionFaceContour.RIGHT_EYE).getPoints();

                                                            // ランドマーク 左目
                                                            List<FirebaseVisionPoint> leftEyeContour =
                                                                    faces.get(0).getContour(FirebaseVisionFaceContour.LEFT_EYE).getPoints();


                                                            // 顔の領域が画面外でない場合
                                                            if (bounds.left >= 0 && bounds.right <= bitmap.getWidth()
                                                                    && bounds.top >= 0 && bounds.bottom <= bitmap.getHeight()) {

                                                                // 顔を切り取った画像
                                                                Bitmap face = Bitmap.createScaledBitmap(cropBitmap(bitmap, bounds), scaledSize, scaledSize, false);

                                                                Log.v("Contour", bounds.toString());
                                                                // 右目を切り取った画像
                                                                Rect rightRec = calEyeRect(rightEyeContour);
                                                                Log.v("EyeRect", rightRec.toString());
                                                                Bitmap test  =cropBitmap(bitmap, rightRec);
                                                                Bitmap right = Bitmap.createScaledBitmap(cropBitmap(bitmap, rightRec), scaledSize, scaledSize, false);
                                                                Log.v("right_eye", "crop :" + test.getWidth() + ":" + test.getHeight()
                                                                        + "\nscaled :" + right.getWidth() +":"+ right.getHeight());

                                                                // 左目を切り取った画像
                                                                Rect leftRec = calEyeRect(leftEyeContour);
                                                                Bitmap left = Bitmap.createScaledBitmap(cropBitmap(bitmap, leftRec), scaledSize, scaledSize, false);

                                                                // 画面の中のどこに顔があるかを示している画像
                                                                Bitmap grid = Bitmap.createBitmap(bitmap.getWidth(), bitmap.getHeight(), bitmap.getConfig());
                                                                Canvas cv = new Canvas(grid);
                                                                cv.drawColor(Color.BLACK);
                                                                Paint p = new Paint();
                                                                p.setColor(Color.WHITE);
                                                                cv.drawRect(bounds, p);
                                                                cv.drawBitmap(grid, 0, 0, null);
                                                                grid = Bitmap.createScaledBitmap(grid, 25, 25, false);

                                                                saveImage(grid, "grid");

                                                                saveImage(face, "face");

                                                                saveImage(right, "right");

                                                                saveImage(left, "left");

                                                                //computingDetection = false;
                                                                recognize(face, right, left, grid);
                                                            }
                                                            // 顔の領域が画面の外に及ぶ場合
                                                            else {
                                                                computingDetection = false;
                                                            }

                                                            // ここで色々用いて顔画像を切り出す

                                                        } else {
                                                            computingDetection = false;
                                                        }
                                                    }
                                                })
                                        .addOnFailureListener(
                                                new OnFailureListener() {
                                                    @Override
                                                    public void onFailure(@NonNull Exception e) {
                                                        // Task failed with an exception
                                                        // ...
                                                        computingDetection = false;
                                                    }
                                                });

                    }
                });
    }

    private Rect calEyeRect(List<FirebaseVisionPoint> lists) {

        int left, right, top, bottom = 0;
        left = (int) lists.get(0).getX().floatValue();
        right = (int) lists.get(0).getX().floatValue();
        top = (int) lists.get(0).getY().floatValue();
        bottom = (int) lists.get(0).getY().floatValue();

        for (int i = 1; i < lists.size(); i++) {
            if ((int) lists.get(i).getX().floatValue() < left) {
                left = (int) lists.get(i).getX().floatValue();
            }
            if ((int) lists.get(i).getX().floatValue() > right) {
                right = (int) lists.get(i).getX().floatValue();
            }
            if ((int) lists.get(i).getY().floatValue() < top) {
                top = (int) lists.get(i).getY().floatValue();
            }
            if ((int) lists.get(i).getY().floatValue() > bottom) {
                bottom = (int) lists.get(i).getY().floatValue();
            }
        }
        int addValue = (right - left) / 2;
        int ymiddlePoint = top + (bottom - top) / 2;

        Rect rec = new Rect(left - addValue, ymiddlePoint - addValue * 2, right + addValue, ymiddlePoint + addValue * 2);

        return rec;
    }

    private void saveImage(Bitmap finalBitmap, String word) {

        String root = Environment.getExternalStorageDirectory().getPath();
        File myDir = new File(root + "/saved_images");
        myDir.mkdirs();
        try {
            if (!myDir.exists()) {
                boolean test = myDir.mkdirs();
                Log.v("mkdir", String.valueOf(test));
                Log.e("MKDIR", "directory is not exists");
            } else {
                Log.e("MKDIR", "directory is exists");
            }
        } catch (SecurityException e) {
            e.printStackTrace();
        }

        String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
        String fname = word + "_" + timeStamp + ".jpg";

        File file = new File(myDir, fname);
        if (file.exists()) file.delete();
        try {
            FileOutputStream out = new FileOutputStream(file);
            finalBitmap.compress(Bitmap.CompressFormat.JPEG, 100, out);
            out.flush();
            out.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /* Checks if external storage is available for read and write */
    public boolean isExternalStorageWritable() {
        String state = Environment.getExternalStorageState();
        if (Environment.MEDIA_MOUNTED.equals(state)) {
            return true;
        }
        return false;
    }

    // 画像をクロップする
    public static Bitmap cropBitmap(Bitmap bitmap, Rect rect) {

        Log.v("cropBitmap", String.valueOf(bitmap.getWidth()) + ":" + bitmap.getHeight());
        Log.v("Bitmaprect", rect.left + ":" + rect.right + ":" + rect.top + ":" + rect.bottom);
        // トリミングしたい画像の横幅と縦幅（マイナスになることもある．）
        int w = rect.width();
        int h = rect.height();
        Bitmap ret = Bitmap.createBitmap(bitmap, rect.left, rect.top, w, h);
        Log.v("cropBitmap", ret.getWidth() + ":" + ret.getHeight());
        return ret;
    }

    private void recognize(Bitmap face, Bitmap right, Bitmap left, Bitmap grid) {

        runInBackground(
                new Runnable() {
                    @Override
                    public void run() {
                        Log.v("MLKit", "recognize");

                        //LOGGER.i("Running detection on image " + currTimestamp);
                        final long startTime = SystemClock.uptimeMillis();

                        //final List<Classifier.Recognition> results = detector.recognizeImage(croppedBitmap);
                        final float[][] results = detector.recognizeImageEye(face, right, left, grid);

                        Log.v("neko", "x:" + results[0][0] + "\n" + results[0][1]);
                        computingDetection = false;


            /*            lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;

                        cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);
                        // 紐付け？
                        final Canvas canvas = new Canvas(cropCopyBitmap);
                        final Paint paint = new Paint();
                        paint.setColor(Color.RED);
                        paint.setStyle(Style.STROKE);
                        paint.setStrokeWidth(2.0f);

                        float minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
                        switch (MODE) {
                            case TF_OD_API:
                                minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
                                break;
                        }

                        final List<Classifier.Recognition> mappedRecognitions =
                                new LinkedList<Classifier.Recognition>();

                        for (final Classifier.Recognition result : results) {
                            final RectF location = result.getLocation();
                            if (location != null && result.getConfidence() >= minimumConfidence) {
                                canvas.drawRect(location, paint);

                                cropToFrameTransform.mapRect(location);

                                result.setLocation(location);
                                mappedRecognitions.add(result);
                            }
                        }

                        // 一時的にtimestampに100を代入
                        tracker.trackResults(mappedRecognitions, 100);
                        trackingOverlay.postInvalidate();

                        computingDetection = false;

                        runOnUiThread(
                                new Runnable() {
                                    @Override
                                    public void run() {
                                        showFrameInfo(previewWidth + "x" + previewHeight);
                                        showCropInfo(cropCopyBitmap.getWidth() + "x" + cropCopyBitmap.getHeight());
                                        showInference(lastProcessingTimeMs + "ms");
                                    }
                                });*/
                    }
                });
    }

    @Override
    protected int getLayoutId() {
        return R.layout.camera_connection_fragment_tracking;
    }

    @Override
    protected Size getDesiredPreviewFrameSize() {
        return DESIRED_PREVIEW_SIZE;
    }

    // Which detection model to use: by default uses Tensorflow Object Detection API frozen
    // checkpoints.
    private enum DetectorMode {
        TF_OD_API;
    }

    @Override
    protected void setUseNNAPI(final boolean isChecked) {
        runInBackground(() -> detector.setUseNNAPI(isChecked));
    }

    @Override
    protected void setNumThreads(final int numThreads) {
        runInBackground(() -> detector.setNumThreads(numThreads));
    }
}
