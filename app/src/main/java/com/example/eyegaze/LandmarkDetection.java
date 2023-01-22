package com.example.eyegaze;

import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

public class LandmarkDetection {

    public static Mat LandmarkDetection(Mat mRgba) {
        // convert into RGB (three channel)
        Mat mRgb = new Mat();
        Imgproc.cvtColor(mRgba, mRgb, Imgproc.COLOR_RGBA2GRAY);



        return mRgb;
    };
}
