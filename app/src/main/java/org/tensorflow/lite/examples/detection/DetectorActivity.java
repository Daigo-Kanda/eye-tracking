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

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Rect;
import android.graphics.Typeface;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.Environment;
import android.os.SystemClock;
import android.util.DisplayMetrics;
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

import org.jetbrains.bio.npy.NpyArray;
import org.jetbrains.bio.npy.NpyFile;
import org.tensorflow.lite.examples.detection.customview.OverlayView;
import org.tensorflow.lite.examples.detection.customview.OverlayView.DrawCallback;
import org.tensorflow.lite.examples.detection.env.BorderedText;
import org.tensorflow.lite.examples.detection.env.ImageUtils;
import org.tensorflow.lite.examples.detection.env.Logger;
import org.tensorflow.lite.examples.detection.tflite.Classifier;
import org.tensorflow.lite.examples.detection.tflite.TFLiteObjectDetectionAPIModel;
import org.tensorflow.lite.examples.detection.tracking.MultiBoxTracker;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.file.Paths;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;

/**
 * An activity that uses a TensorFlowMultiBoxDetector and ObjectTracker to detect and then track
 * objects.
 */
public class DetectorActivity extends CameraActivity implements OnImageAvailableListener {
    private static final Logger LOGGER = new Logger();

    // Configuration values for the prepackaged SSD model.
    private static final int TF_OD_API_INPUT_SIZE = 224;
    private static final int cropSizex = 960;
    private static final int cropSizey = 1280;

    private static final boolean TF_OD_API_IS_QUANTIZED = false;
    private static final String TF_OD_API_MODEL_FILE = "converted_model.tflite";
    private static final String TF_OD_API_LABELS_FILE = "file:///android_asset/labelmap.txt";
    private static final DetectorMode MODE = DetectorMode.TF_OD_API;
    // Minimum detection confidence to track a detection.
    private static final float MINIMUM_CONFIDENCE_TF_OD_API = 0.5f;
    private static final boolean MAINTAIN_ASPECT = false;
    private static final Size DESIRED_PREVIEW_SIZE = new Size(960, 1280);
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

    private int scaledSize = 224;

    // 画像上での1pixelあたりの長さ[cm]を格納
    private float[] dis = new float[2];

    private WriteCSV writeCSV = new WriteCSV("gazeEsti_time", false);

    // 確認用
    private float realWidthPerPixel;
    private float realHeightPercPixel;

    /**
     * 顔検出の下準備
     */
    // Real-time contour detection of multiple faces
    private FirebaseVisionFaceDetectorOptions realTimeOpts =
            new FirebaseVisionFaceDetectorOptions.Builder()
                    .setContourMode(FirebaseVisionFaceDetectorOptions.ALL_CONTOURS)
                    .build();

    // Preview画像のサイズが決定されたときに呼ばれるメソッド
    @Override
    public void onPreviewSizeChosen(final Size size, final int rotation) {


        int width = findViewById(R.id.container).getWidth();
        int height = findViewById(R.id.container).getHeight();
        dis = calDistanceOfPerPixel(width, height, cropSizex, cropSizey);

        Log.v("Container", dis[0] + ":" + dis[1]);
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

        // previewのサイズを代入 1280*960
        previewWidth = size.getWidth();
        previewHeight = size.getHeight();

        Log.v("previewsize", previewWidth + ":" + previewHeight);

        sensorOrientation = rotation - getScreenOrientation();
        LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);

        LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);

        // プレビュー（画面上に表示された画像）の情報を持つBitmap
        rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);

        // 深層学習用にサイズが変更された画像を持つBitmap
        croppedBitmap = Bitmap.createBitmap(cropSizex, cropSizey, Config.ARGB_8888);

        // PreviewFrameからCropFrameへの変換用のMatrix
        frameToCropTransform =
                ImageUtils.getTransformationMatrix(
                        previewWidth, previewHeight,
                        cropSizex, cropSizey,
                        sensorOrientation, MAINTAIN_ASPECT);

        // CropFrameからPreviewFrameへの変換用Matrix
        cropToFrameTransform = new Matrix();
        frameToCropTransform.invert(cropToFrameTransform);

        // 画面と関連付け
        trackingOverlay = (OverlayView) findViewById(R.id.tracking_overlay);

        // OverlayViewが更新されるときに呼ばれる
  /*      trackingOverlay.addCallback(
                new DrawCallback() {
                    @Override
                    public void drawCallback(final Canvas canvas) {
                        tracker.draw(canvas);
                        if (isDebug()) {
                            tracker.drawDebug(canvas);
                        }
                    }
                });*/

        trackingOverlay.addCallback(
                new DrawCallback() {
                    @Override
                    public void drawCallback(final Canvas canvas) {
                        tracker.draw_circle(canvas);
                    }
                });

        // 画面描画の設定 実際
        tracker.setFrameConfiguration(previewWidth, previewHeight, sensorOrientation);
    }

    @Override
    protected void processImage() {

        // テスト用に画像を保存
        // saveImage(rgbFrameBitmap, "real");
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

        // Previewの画像を取得．しかし，実際にPreviewを表示しているときにはOrientationをいじっている．
        // 画像を回転させている
        rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);

        readyForNextImage();

        // カメラから取得した映像とcanvasを関連付け
        // Canvasを用いてcroppedBitmapを自由に変更できる
        final Canvas canvas = new Canvas(croppedBitmap);

        // croppedBitmapの描かれたキャンバスにframeToCropTransformを用いてrgbFrameBitmapを描画
        canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);

        Log.v("croppedBitmap", croppedBitmap.getWidth() + ":" + croppedBitmap.getHeight());

        //saveImage(croppedBitmap, "cropped");

        // croppedBitmapを保存
        if (SAVE_PREVIEW_BITMAP) {
            ImageUtils.saveBitmap(croppedBitmap);
        }

        runInBackground(
                new Runnable() {
                    @Override
                    public void run() {

                        LOGGER.i("Running detection on image " + currTimestamp);

                        long startFace = SystemClock.uptimeMillis();

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
                                                                Bitmap test = cropBitmap(bitmap, rightRec);
                                                                Bitmap right = Bitmap.createScaledBitmap(cropBitmap(bitmap, rightRec), scaledSize, scaledSize, false);
                                                                Log.v("right_eye", "crop :" + test.getWidth() + ":" + test.getHeight()
                                                                        + "\nscaled :" + right.getWidth() + ":" + right.getHeight());

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

                                                                int[] intValues_grid;
                                                                intValues_grid = new int[25 * 25];
                                                                grid.getPixels(intValues_grid, 0, grid.getWidth(), 0, 0, grid.getWidth(), grid.getHeight());

                                                                for (int i = 0; i < 25; ++i) {
                                                                    for (int j = 0; j < 25; ++j) {
                                                                        // 一つずつピクセルを取り出す
                                                                        int pixelValue = intValues_grid[i * 25 + j];
                                                                        if (pixelValue == -16777216) {
                                                                            Log.v("grid", "黒");
                                                                        } else if (pixelValue == -1) {
                                                                            Log.v("grid", "白");
                                                                        } else {
                                                                            Log.v("grid", "白黒以外が存在しています");
                                                                        }

                                                                    }
                                                                }

                                                                long faceTime = SystemClock.uptimeMillis() - startFace;

                                                                // saveImage(grid, "grid");

                                                                // saveImage(face, "face");

                                                                // saveImage(right, "right");

                                                                // saveImage(left, "left");

                                                                //computingDetection = false;
                                                                recognize(getBaseContext(), face, left, right, grid, faceTime);
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

    // 画像上での1pixelの実際の距離を返す(cm)．

    /**
     * @param previewWidth  ディスプレイ上に描画している画像の横幅（物理ピクセル）
     * @param previewHeight ディスプレイ上に描画している画像の高さ（物理ピクセル）
     * @param cropWidth     　深層学習に用いる画像の横幅
     * @param cropHeight    　深層学習に用いる画像の高さ
     * @return
     */
    private float[] calDistanceOfPerPixel(int previewWidth, int previewHeight, int cropWidth, int cropHeight) {

        float[] dis = new float[2];

        DisplayMetrics metrics = new DisplayMetrics();
        getWindowManager().getDefaultDisplay().getMetrics(metrics);

        // 実際の解像度(pixel)
        int realWidthPixels = metrics.widthPixels;
        int realHeightPixels = metrics.heightPixels;

        // 画面の大きさ(cm)
        float realWidthCentiMeter = ((float) realWidthPixels / metrics.xdpi) * 2.54f;
        float realHeightCentimeter = ((float) realHeightPixels / metrics.ydpi) * 2.54f;

        // 1物理ピクセルごとの距離
        float realWidthPerPixel = (1 / metrics.xdpi) * 2.54f;
        float realHeightPerPixel = (1 / metrics.ydpi) * 2.54f;

        this.realWidthPerPixel = realWidthPerPixel;
        this.realHeightPercPixel = realHeightPerPixel;

        float imageWidthPerPixel = ((float) previewWidth / (float) cropWidth) * realWidthPerPixel;
        float imageHeightPerPixel = ((float) previewHeight / (float) cropHeight) * realHeightPerPixel;

        dis[0] = imageWidthPerPixel;
        dis[1] = imageHeightPerPixel;

        Log.v("distance", previewHeight + ":" + realHeightCentimeter);


        return dis;
    }

    // bitmap上の視線の位置

    /**
     * @param gaze 深層学習によって手に入ったカメラからの相対的な位置（座標系x右y上）
     * @param dis  画像上での1pixelの実世界上での大きさ
     * @return
     */
    private float[] gazePointOnBitmap(float[] gaze, float[] dis) {

        // MediaPad M5 Proを縦型カメラ右側に持ったときの画面左上から見たカメラの位置[cm]（画面座標系）
        float dx = 15.2f;
        float dy = 11.7f;

        // 画面左上からgazeの位置（座標系の向き考慮）
        float realx = dx + gaze[0];
        float realy = dy - gaze[1];

        int x = (int) (realx / dis[0]);
        int y = (int) (realy / dis[1]);

        return new float[]{x, y};

    }

    private float[] gazePointOnReal(float[] gaze) {

        // MediaPad M5 Proを縦型カメラ右側に持ったときの画面左上から見たカメラの位置[cm]（画面座標系）
        float dx = 15.2f;
        float dy = 11.7f;

        // 画面左上からgazeの位置（座標系の向き考慮）
        float realx = dx + gaze[0];
        float realy = dy - gaze[1];

        int x = (int) (realx / realWidthPerPixel);
        int y = (int) (realy / realHeightPercPixel);

        return new float[]{x, y};

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

    private void recognize(Context context, Bitmap face, Bitmap right, Bitmap left, Bitmap grid, long faceTime) {

        runInBackground(
                new Runnable() {
                    @Override
                    public void run() {
                        Log.v("MLKit", "recognize");

                        long startCNN = SystemClock.uptimeMillis();

                        //LOGGER.i("Running detection on image " + currTimestamp);
                        final long startTime = SystemClock.uptimeMillis();

                        BitmapFactory.Options options = new BitmapFactory.Options();
                        options.inSampleSize = 2;

                        Bitmap face_b = null;
                        Bitmap right_b = null;
                        Bitmap left_b = null;

                        MyApplication ma = (MyApplication)context.getApplicationContext();
                        ma.loadNPY();
                        NpyArray face_mean = ma.face_mean;
                        NpyArray right_mean = ma.right_mean;
                        NpyArray left_mean = ma.left_mean;

                        float test = face_mean.asFloatArray()[224];

//                        NpyArray test = NpyFile.read(Paths.get(URI.parse("file:///android_asset/face_mean.npy")), 1000);


                        try {
                            face_b = BitmapFactory.decodeStream(getResources().getAssets().open("00757_face.jpg"));
                            right_b = BitmapFactory.decodeStream(getResources().getAssets().open("00757_right.jpg"));
                            left_b = BitmapFactory.decodeStream(getResources().getAssets().open("00757_left.jpg"));
                        } catch (IOException e) {


                        }

                        face_b = Bitmap.createScaledBitmap(face_b, scaledSize, scaledSize, true);
                        right_b = Bitmap.createScaledBitmap(right_b, scaledSize, scaledSize, true);
                        left_b = Bitmap.createScaledBitmap(left_b, scaledSize, scaledSize, true);


                        InputStream stream = null;
                        try {
                            stream = context.getAssets().open("grid.txt");
                        } catch (IOException e) {
                            e.printStackTrace();
                        }
                        BufferedReader reader = new BufferedReader(new InputStreamReader(stream));

                        float[] check = new float[625];
                        int count = 0;

                        try {
                            String csvLine;
                            while ((csvLine = reader.readLine()) != null) {
                                check[count] = Float.parseFloat(csvLine.split(",")[0]);
                                count++;
                            }
                        } catch (IOException ex) {
                            throw new RuntimeException("Error in reading CSV file: " + ex);
                        } finally {
                            try {
                                stream.close();
                            } catch (IOException e) {
                                throw new RuntimeException("Error while closing input stream: " + e);
                            }
                        }

                        //final List<Classifier.Recognition> results = detector.recognizeImage(croppedBitmap);
                        final float[][] results = detector.recognizeImageEye(face_b, right_b, left_b, check, face_mean, right_mean, left_mean);

                        // 視線推定の結果からcropのビットマップ上の位置を計算
                        float[] result = gazePointOnReal(new float[]{results[0][0], results[0][1]});
                        // float[] result = gazePointOnReal(new float[]{-1f, 0f});
                        //float[] result = gazePointOnBitmap(new float[]{-5f, 0f}, dis);

                        Log.v("neko", "x:" + results[0][0] + "\n" + results[0][1]);


                        // 時間の測定
                        long endCNN = SystemClock.uptimeMillis();

                        if (result != null) {
                            writeCSV.MakeFile(faceTime + "," + String.valueOf(endCNN - startCNN));
                        }
                        // cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);
                        // 紐付け？
                        // final Canvas canvas = new Canvas(cropCopyBitmap);
//                        final Paint paint = new Paint();
//                        paint.setColor(Color.RED);
//                        paint.setStyle(Paint.Style.STROKE);
//                        paint.setStrokeWidth(2.0f);

                        // cropToFrameTransform.mapPoints(result);

                        tracker.setEyePosition(result);

                        // 更新要請
                        trackingOverlay.postInvalidate();

/*                        runOnUiThread(
                                new Runnable() {
                                    @Override
                                    public void run() {
                                        showFrameInfo(previewWidth + "x" + previewHeight);
                                        showCropInfo(cropCopyBitmap.getWidth() + "x" + cropCopyBitmap.getHeight());
                                        showInference(lastProcessingTimeMs + "ms");
                                    }
                                });*/
/*

                        // 時間の測定
                        lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;

                        cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);
                        // 紐付け？
                        final Canvas canvas = new Canvas(cropCopyBitmap);
                        final Paint paint = new Paint();
                        paint.setColor(Color.RED);
                        paint.setStyle(Paint.Style.STROKE);
                        paint.setStrokeWidth(2.0f);

                        // 検知するサイズが小さい場合検知しない
                        float minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
                        switch (MODE) {
                            case TF_OD_API:
                                minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
                                break;
                        }

                        // 結果を保存するリストの作成
                        final List<Classifier.Recognition> mappedRecognitions =
                                new LinkedList<Classifier.Recognition>();

                        // 深層学習で得た結果をbitmapに貼り付けている
                        for (final Classifier.Recognition result : results) {
                            final RectF location = result.getLocation();
                            if (location != null && result.getConfidence() >= minimumConfidence) {
                                canvas.drawRect(location, paint);

                                // locatin(rect)をcropToFrameTransformのマトリックスで変形
                                cropToFrameTransform.mapRect(location);

                                // 変形させた結果を元に戻している
                                result.setLocation(location);
                                // listに格納
                                mappedRecognitions.add(result);
                            }
                        }

                        // trackingのデータについて
                        tracker.trackResults(mappedRecognitions, 100);

                        // 更新要請
                        trackingOverlay.postInvalidate();

                        runOnUiThread(
                                new Runnable() {
                                    @Override
                                    public void run() {
                                        showFrameInfo(previewWidth + "x" + previewHeight);
                                        showCropInfo(cropCopyBitmap.getWidth() + "x" + cropCopyBitmap.getHeight());
                                        showInference(lastProcessingTimeMs + "ms");
                                    }
                                });
*/

                        computingDetection = false;
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
