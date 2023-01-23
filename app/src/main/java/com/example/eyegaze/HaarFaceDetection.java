package com.example.eyegaze;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

public class HaarFaceDetection {

    static Mat detectFace(Mat mRgba, CascadeClassifier cascadeClassifierFace, CascadeClassifier cascadeClassifierEye) {
            // convert into RGB
            Mat mRgb = new Mat();
            Imgproc.cvtColor(mRgba, mRgb, Imgproc.COLOR_RGBA2RGB);

            int height = mRgb.height();
            // minimum size of face in frame
            int absoluteFaceSize = (int) (height * 0.1);

            MatOfRect faces = new MatOfRect();
            if (cascadeClassifierFace != null) {
                // input, output,                                       min size of output
                cascadeClassifierFace.detectMultiScale(
                        mRgb,
                        faces,
                        1.1,
                        2,
                        2,
                        new Size(absoluteFaceSize, absoluteFaceSize),
                        new Size()
                );
            }

            // loop through all faces
            Rect[] facesArray = faces.toArray();

            for (int i = 0; i < facesArray.length; i++) {
                // draw face on original frame mRgba
                Imgproc.rectangle(
                        mRgba,
                        facesArray[i].tl(),
                        facesArray[i].br(),
                        new Scalar(0, 255, 0, 255),
                        2
                );

                // For eyes, loop through all detected faces
                // crop face image and pass it through eye classifier
                // starting point, width, height of crop
                Rect roi=new Rect(
                        (int)facesArray[i].tl().x,
                        (int)facesArray[i].tl().y,
                        (int)facesArray[i].br().x-(int)facesArray[i].tl().x,
                        (int)facesArray[i].br().y-(int)facesArray[i].tl().y
                );

                Mat cropped = new Mat(mRgba, roi);

                // array to store eye coords, have to pass MatOfRect to classifier
                MatOfRect eyes = new MatOfRect();

                if (cascadeClassifierEye != null) {
                    // find biggest size object (eye)
                    cascadeClassifierEye.detectMultiScale(
                            cropped,
                            eyes,
                            1.15,
                            2,
                            2,
                            new Size(35,35),
                            new Size()
                    );

                    // now create an array
                    Rect[] eyesCoords = eyes.toArray();

                    // loop through each eye and draw it out
                    for (int j = 0; j < eyesCoords.length; j++) {
                        // find coordinate on original frame mRgba
                        // starting point
                        int x1 = (int)(eyesCoords[j].tl().x + facesArray[i].tl().x);
                        int y1 = (int)(eyesCoords[j].tl().y + facesArray[i].tl().y);
                        // width and height
                        int w1 = (int)(eyesCoords[j].br().x - eyesCoords[j].tl().x);
                        int h1 = (int)(eyesCoords[j].br().y - eyesCoords[j].tl().y);
                        // end point
                        int x2 = (int)(w1 + x1);
                        int y2 = (int)(h1 + y1);
                        // draw eye on original frame mRgba
                        //input    starting point   ending point   color                 thickness
                        Imgproc.rectangle(
                                mRgba,
                                new Point(x1,y1),
                                new Point(x2,y2),
                                new Scalar(0,255,0,255),
                                2
                        );

                        // -------------- Pupil detection ---------------------
                        // crop eye from face
                        // to reduce cropped eye image (with some changes - experimental)
                        Rect eye_roi = new Rect(
                                x1 + 5,
                                y1 + 22,
                                w1 -5,
                                h1 - 10
                        );

                        Mat eye_cropped = new Mat(mRgba, eye_roi);


                        // Convert to grayscale
                        Mat grayscaled_eye_cropped = new Mat();
                        Imgproc.cvtColor(
                                eye_cropped,
                                grayscaled_eye_cropped,
                                Imgproc.COLOR_RGB2GRAY
                        );
                        // blur image to get better result
                        Imgproc.blur(
                                grayscaled_eye_cropped,
                                grayscaled_eye_cropped,
                                new Size(5, 5)
                        );

                        // apply threshold layer to convert it into inverse binary image thresh was 110
                        Imgproc.threshold(
                                grayscaled_eye_cropped,
                                grayscaled_eye_cropped,
                                60,
                                255,
                                Imgproc.THRESH_BINARY_INV
                        );

                        // add it back to original cropped eye image
                        Imgproc.cvtColor(
                                grayscaled_eye_cropped,
                                grayscaled_eye_cropped,
                                Imgproc.COLOR_GRAY2RGBA
                        );

                        // input, input, output
                        Core.add(grayscaled_eye_cropped, eye_cropped, eye_cropped);

                        eye_cropped.copyTo(new Mat(mRgba, eye_roi));
                    }
                }
            }
            return mRgba;

    }
}
