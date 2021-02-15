package org.tensorflow.lite.examples.detection;

import android.os.Environment;
import android.util.Log;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStreamWriter;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;

/**
 * Created by jylab on 2017/12/13.
 */

//データを保存するためのクラス

/**
 * 使い方
 * csvを保存するとき
 */
public class WriteCSV {

    private static final String check = "writecsv";
    //パスを取得する
    private String storagePath = Environment.getExternalStorageDirectory().getPath();
    private String fileName;

    /**
     * @param directoryName ディレクトリの名前を入れる
     * @param flag          画像かcsvかを決定する
     */
    WriteCSV(String directoryName, boolean flag) {

        Log.v(check, "DataStorage");

        //保存先のディレクトリがなければ作成する．画像保存用のディレクトリ
        File file = new File(storagePath + "/" + directoryName + "/");
        try {
            if (!file.exists()) {
                boolean test = file.mkdirs();
                Log.v("mkdir", String.valueOf(test));
                Log.e(check, "directory is not exists");
            } else {
                Log.e(check, "directory is exists");
            }
        } catch (SecurityException e) {
            e.printStackTrace();
        }

        fileName = "/" + directoryName + "/" + generateFileName(flag);


        //外部ストレージにアクセスできるかどうかの確認
        if (!isExternalStorageWritable()) {

            Log.i(check, "External Storage Not Writable.");

            return;

        }

    }

    //一行分のデータを入れる
    //改行入れる必要はない
    public void MakeFile(String str) {

        //Log.v("DataStorage", "MakeFile");

        FileOutputStream fileOutputStream;
        File file = new File(storagePath + fileName);

        file.getParentFile().mkdir();
        try {
            //Log.e(check, "in the try");
            fileOutputStream = new FileOutputStream(file, true);
            OutputStreamWriter outputStreamWriter
                    = new OutputStreamWriter(fileOutputStream, "UTF-8");
            BufferedWriter bw = new BufferedWriter(outputStreamWriter);

            bw.write(str);
            bw.newLine();
            bw.flush();
            bw.close();
            //Log.e(check, "can write");

        } catch (Exception e) {
            e.printStackTrace();
            //Log.e(check, "can not write");
        }
    }

    //配列を格納するための関数
    public void MakeFile(double[] data) {

        FileOutputStream fileOutputStream;
        File file = new File(storagePath + fileName);

        file.getParentFile().mkdir();
        try {
            Log.e(check, "in the try");
            fileOutputStream = new FileOutputStream(file, true);
            OutputStreamWriter outputStreamWriter
                    = new OutputStreamWriter(fileOutputStream, "UTF-8");
            BufferedWriter bw = new BufferedWriter(outputStreamWriter);

            for (int j = 0; j < data.length; j++) {
                bw.append(String.valueOf(data[j]) + ",");
            }

            bw.newLine();
            bw.flush();
            bw.close();
            Log.e(check, "can write");

        } catch (Exception e) {
            e.printStackTrace();
            Log.e(check, "can not write");
        }

    }

    //外部ストレージが書き込み可能かチェックする
    private boolean isExternalStorageWritable() {

        Log.v(check, "isExternalStorageWritable");

        String state = Environment.getExternalStorageState();

        return Environment.MEDIA_MOUNTED.equals(state);

    }

    //画像の名前の決定
    //trueの時ＰＮＧでfalseの時はＣＳＶ
    synchronized private String generateFileName(Boolean b) {

        Log.v(check, "generateFileName");

        Date date = new Date();

        SimpleDateFormat fileNameFormat = new SimpleDateFormat("yyyyMMdd_HHmmss_SSS", Locale.ENGLISH);

        if (b == true) {
            return fileNameFormat.format(date) + ".png";
        } else {
            return fileNameFormat.format(date) + ".csv";
        }
    }
}
