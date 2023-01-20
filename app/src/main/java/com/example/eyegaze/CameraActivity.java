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
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.objdetect.Objdetect;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Collections;
import java.util.List;

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

        // ------------ Face/eye detection --------------
        mRgba = CascadeRec(mRgba);

         return mRgba;
        // return mGray;
        // return edges;
    }

    private Mat CascadeRec(Mat mRgba) {
        // original frame was rotated -90 deg, need to rotate again for model
        Core.flip(mRgba.t(), mRgba, 1);
        // convert into RGB
        Mat mRgb = new Mat();
        Imgproc.cvtColor(mRgba, mRgb, Imgproc.COLOR_RGBA2RGB);

        int height = mRgb.height();
        // minimum size of face in frame
        int absoluteFaceSize = (int) (height * 0.1);

        MatOfRect faces = new MatOfRect();
        if (cascadeClassifierFace != null) {
                                            // input, output,                                       min size of output
            cascadeClassifierFace.detectMultiScale(mRgb, faces, 1.1, 2, 2, new Size(absoluteFaceSize, absoluteFaceSize), new Size());
        }

        // loop through all faces
        Rect[] facesArray = faces.toArray();

        for (int i = 0; i < facesArray.length; i++) {
            // draw face on original frame mRgba
            Imgproc.rectangle(mRgba, facesArray[i].tl(), facesArray[i].br(), new Scalar(0, 255, 0, 255), 2);

            // For eyes, loop through all detected faces
            // crop face image and pass it through eye classifier

            // https://answers.opencv.org/question/203268/mat-out-of-bounds/
            Rect r = facesArray[i];
            r.x = Math.max(r.x,0);
            r.y = Math.max(r.y,0);
            r.width = Math.min(mRgba.cols()-1-r.x, r.width);
            r.height = Math.min(mRgba.rows()-1-r.y, r.height);
            Mat cropped_face = new Mat(mRgba,r); // now you can crop it safely !

            // starting point, width, height of crop
            Rect roi = new Rect((int)facesArray[i].tl().x, (int)facesArray[i].br().y,
                    (int)facesArray[i].br().x - (int)facesArray[i].tl().x,
                    (int)facesArray[i].br().y - (int)facesArray[i].tl().y);

            // cropped image matrix
            Mat cropped = new Mat(mRgba, roi);


            // array to store eye coords, have to pass MatOfRect to classifier
            MatOfRect eyes = new MatOfRect();

            if (cascadeClassifierEye != null) {
                // find biggest - find biggest size object (eye)
                cascadeClassifierEye.detectMultiScale(cropped, eyes, 1.15, 2,
                        Objdetect.CASCADE_FIND_BIGGEST_OBJECT | Objdetect.CASCADE_SCALE_IMAGE,
                        // minimum size of eye
                        new Size(35, 35), new Size());

                // now create an array
                Rect[] eyesCoords = eyes.toArray();

                // loop through each eye and draw it out
                for (int j = 0; j < eyesCoords.length; j++) {
                    // find coords on original frame mRgba
                    // starting coords
                    int x1 = (int)(eyesCoords[j].tl().x + facesArray[i].tl().x);
                    int y1 = (int)(eyesCoords[j].tl().y + facesArray[i].tl().y);
                    // width and height
                    int w1 = (int)(eyesCoords[j].tl().x - eyesCoords[j].tl().x);
                    int h1 = (int)(eyesCoords[j].tl().y - eyesCoords[j].tl().y);
                    // ending coords
                    int x2 = (int)(x1 + x1);
                    int y2 = (int)(y1 + h1);

                    Imgproc.rectangle(mRgba, new Point(x1, y1), new Point(x2, y2), new Scalar(0, 255, 0), 2);

                }

            }

        }

        // rotate back to -90deg
        Core.flip(mRgba.t(), mRgba, 0);



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
