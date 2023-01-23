package com.example.eyegaze;

import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.app.Activity;
import android.content.Context;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.util.Log;
import android.view.SurfaceView;
import android.view.Window;
import android.view.WindowManager;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

public class CameraActivity extends Activity implements CameraBridgeViewBase.CvCameraViewListener2 {

    private static final String TAG = "CameraActivity";

    private Mat mRgba;
    private Mat mGray;

    private CameraBridgeViewBase mOpenCvCameraView;

    private CascadeClassifier cascadeClassifierFace;
    private CascadeClassifier cascadeClassifierEye;

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    Log.i(TAG, "OpenCv is loaded.");
                    // Missing in the tutorial!!! https://www.youtube.com/watch?v=mZBIPOl983I&list=PL0aoTDj9Nwgh0hTC3QBHwKtJuxl1veGyG&index=3
                    mOpenCvCameraView.enableView();
                } break;
                default: {
                    super.onManagerConnected(status);
                } break;
            }

        }
    };

    public CameraActivity() {
        Log.i(TAG, "Instantiated new "+this.getClass());
    }

//    @Override
//    protected List<? extends CameraBridgeViewBase> getCameraViewList() {
//        return Collections.singletonList(mOpenCvCameraView);
//    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        requestWindowFeature(Window.FEATURE_NO_TITLE);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        int MY_PERMISSION_REQUEST_CAMERA = 1;
        // If camera permission is not granted, this code will ask for it
        if (ContextCompat.checkSelfPermission(CameraActivity.this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_DENIED) {
            ActivityCompat.requestPermissions(CameraActivity.this, new String[] {Manifest.permission.CAMERA}, MY_PERMISSION_REQUEST_CAMERA);
        }

        setContentView(R.layout.activity_camera);

        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.frame_camera);
        mOpenCvCameraView.setCameraIndex(1);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);

        // load model for face detection
        try {
            InputStream is = getResources().openRawResource(R.raw.haarcascade_frontalface_alt);
            File cascadeDir = getDir("cascade", Context.MODE_PRIVATE); // creating a folder
            File mCascadeFileFace = new File(cascadeDir, "haarcascade_frontalface_alt.xml"); // creating file in that folder
            FileOutputStream os = new FileOutputStream(mCascadeFileFace);

            byte[] buffer = new byte[4096];
            int byteRead;

            // writing that file from raw folder
            while ((byteRead = is.read(buffer)) != -1) {
                os.write(buffer, 0, byteRead);
            }

            is.close();
            os.close();

            // loading file from cascade folder above
            cascadeClassifierFace = new CascadeClassifier(mCascadeFileFace.getAbsolutePath());

            InputStream is2 = getResources().openRawResource(R.raw.haarcascade_eye);
            File mCascadeFileEye = new File(cascadeDir, "haarcascade_eye.xml"); // creating file in that folder
            FileOutputStream os2 = new FileOutputStream(mCascadeFileEye);

            byte[] buffer2 = new byte[4096];
            int byteRead2;

            while ((byteRead2 = is2.read(buffer2)) != -1) {
                os2.write(buffer2, 0, byteRead2);
            }

            is2.close();
            os2.close();

            cascadeClassifierEye = new CascadeClassifier(mCascadeFileEye.getAbsolutePath());

        } catch (IOException e) {
            Log.i(TAG, "Cascade file not found!");
        }
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (OpenCVLoader.initDebug()) {
            Log.d(TAG, "OpenCV initialization is done!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        } else {
            // If no loaded
            Log.d(TAG, "OpenCV is not loaded. Try again!");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, this, mLoaderCallback);
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        if (mOpenCvCameraView != null) {
            mOpenCvCameraView.disableView();
        }
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null) {
            mOpenCvCameraView.disableView();
        }
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        Log.i(TAG, "onCameraViewStarted");
        mRgba = new Mat(height, width, CvType.CV_8UC4);
        mGray = new Mat(height, width, CvType.CV_8UC1);
    }

    @Override
    public void onCameraViewStopped() {
        Log.i(TAG, "onCameraViewStopped");
        mRgba.release();
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        mRgba = inputFrame.rgba();
        mGray = inputFrame.gray();

        // ------------ Face/eye detection using OpenCV HaarCascade --------------
         mRgba = HaarFaceDetection.detectFace(mRgba, cascadeClassifierFace, cascadeClassifierEye);

         return mRgba;
    }

}


//         Convert rgba to gray
//                        (input, output, action)
//         Imgproc.cvtColor(mRgba, mRgba, Imgproc.COLOR_RGBA2GRAY);

//        // ------------- Line detection ------------------
//        // Edges first
//        Mat edges = new Mat();
//        Imgproc.Canny(mRgba, edges, 80, 200);
//        // Then lines
//        Mat lines = new Mat();
//        // Starting and ending point of lines
//        Point p1 = new Point();
//        Point p2 = new Point();
//        double a, b;
//        double x0, y0;
//
//        Imgproc.HoughLines(edges, lines, 1.0, Math.PI / 180.0, 140);
//
//        // Loop through each line
//        for (int i = 0; i < lines.rows(); i++) {
//            double[] vec = lines.get(i, 0);
//            double rho = vec[0];
//            double theta = vec[1];
//
//            //
//            a = Math.cos(theta);
//            b = Math.sin(theta);
//
//            x0 = a * rho;
//            y0 = b * rho;
//
//            // Starting point and ending point.
//            p1.x = Math.round(x0 + 1000 * (-b));
//            p1.y = Math.round(y0 + 1000 * a);
//            p2.x = Math.round(x0 - 1000 * (-b));
//            p2.y = Math.round(y0 - 1000 * a);
//
//            // Draw line on original frame
//            //            draw on, start, end, color, thickness
//            Imgproc.line(mRgba, p1, p2, new Scalar(255.0, 255.0, 255.0), 1, Imgproc.LINE_AA, 0);
//        }
//        // ------------- Line detection -----------------
