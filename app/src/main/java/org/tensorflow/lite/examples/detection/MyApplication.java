package org.tensorflow.lite.examples.detection;

import android.app.Application;
import android.util.Log;

import org.jetbrains.bio.npy.NpyArray;
import org.jetbrains.bio.npy.NpyFile;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.nio.file.Paths;

public class MyApplication extends Application {

    private final String TAG = "DEBUG-APPLICATION";
    public NpyArray face_mean, right_mean, left_mean;

    @Override
    public void onCreate() {
        /** Called when the Application-class is first created. */


        Log.v(TAG, "--- onCreate() in ---");
    }

    public void loadNPY() {
        getNpyArray("face_mean");
        getNpyArray("right_mean");
        getNpyArray("left_mean");
    }

    @Override
    public void onTerminate() {
        /** This Method Called when this Application finished. */
        Log.v(TAG, "--- onTerminate() in ---");
    }

    private void getNpyArray(String s) {
        File f = new File(getCacheDir() + "/" + s + ".npy");
        if (!f.exists()) try {

            InputStream is = getAssets().open(s + ".npy");
            int size = is.available();
            byte[] buffer = new byte[size];
            is.read(buffer);
            is.close();


            FileOutputStream fos = new FileOutputStream(f);
            fos.write(buffer);
            fos.close();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        if (s.equals("face_mean")) {
            face_mean = NpyFile.read(Paths.get(f.getAbsolutePath()), 1000);
        }

        if (s.equals("right_mean")) {
            right_mean = NpyFile.read(Paths.get(f.getAbsolutePath()), 1000);
        }

        if (s.equals("left_mean")) {
            left_mean = NpyFile.read(Paths.get(f.getAbsolutePath()), 1000);
        }
    }
}