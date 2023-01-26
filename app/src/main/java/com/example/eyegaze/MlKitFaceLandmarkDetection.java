package com.example.eyegaze;

import android.graphics.Bitmap;
import android.graphics.PointF;
import android.util.Log;

import androidx.annotation.NonNull;

import com.google.android.gms.tasks.OnFailureListener;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.face.Face;
import com.google.mlkit.vision.face.FaceContour;
import com.google.mlkit.vision.face.FaceDetection;
import com.google.mlkit.vision.face.FaceDetector;
import com.google.mlkit.vision.face.FaceDetectorOptions;

import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

public class MlKitFaceLandmarkDetection {

    private static String TAG = "MlKitFaceLandmarkDetection";

    static Mat runFaceContourDetection(Mat mRgba) {
        int width = mRgba.width();
        int height = mRgba.height();
        Bitmap bmp = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(mRgba, bmp);

        InputImage image = InputImage.fromBitmap(bmp, 0);
        // Real-time contour detection
        FaceDetectorOptions realTimeOpts =
                new FaceDetectorOptions.Builder()
                        .setContourMode(FaceDetectorOptions.CONTOUR_MODE_ALL)
//                        .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)
                        .build();

        FaceDetector detector = FaceDetection.getClient(realTimeOpts);
        detector.process(image)
                .addOnSuccessListener(
                        new OnSuccessListener<List<Face>>() {
                            @Override
                            public void onSuccess(List<Face> faces) {
                                Log.i(TAG, "Found face");
                                List<PointF> mLeyftEyePoints = processFaceContourDetectionResult(faces);
                                Point min = new Point(1000, 1000);
                                Point max = new Point(-1, -1);
                                for (PointF point : mLeyftEyePoints) {
                                    if (point.x < min.x) {
                                        min.x = point.x;
                                        min.y = point.y;
                                    }
                                    if (point.x > max.x) {
                                        max.x = point.x;
                                        max.y = point.y;
                                    }
                                }

                                Imgproc.drawMarker(mRgba, min, new Scalar(0, 255, 0 ,255));
                                Imgproc.drawMarker(mRgba, max, new Scalar(0, 255, 0 ,255));
                            }
                        })
                .addOnFailureListener(
                        new OnFailureListener() {
                            @Override
                            public void onFailure(@NonNull Exception e) {
                                e.printStackTrace();
                            }
                        });

        return mRgba;

    }

    private static List<PointF> processFaceContourDetectionResult(List<Face> faces) {
        // Task completed successfully
        if (faces.size() == 0) {
            Log.i(TAG, "No face");
        }

        List<PointF> leftEyeContour = new ArrayList<PointF>();

        for (Face face : faces) {
//            Rect bounds = face.getBoundingBox();
//            float rotY = face.getHeadEulerAngleY();  // Head is rotated to the right rotY degrees
//            float rotZ = face.getHeadEulerAngleZ();  // Head is tilted sideways rotZ degrees

            // If contour detection was enabled:
            leftEyeContour =
                    face.getContour(FaceContour.LEFT_EYE).getPoints();
//            List<PointF> rightEyeContour =
//                    face.getContour(FaceContour.RIGHT_EYE).getPoints();

//            Log.d(TAG, "Left eye contours: " + leftEyeContour);
//            Log.d(TAG, "Right eye contours: " + rightEyeContour);

//            for (PointF point : rightEyeContour) {
//                if (!leftEyeContour.contains(point)) {
//                    leftEyeContour.add(point);
//                }
//            }
        }

        return leftEyeContour;
    }

}
