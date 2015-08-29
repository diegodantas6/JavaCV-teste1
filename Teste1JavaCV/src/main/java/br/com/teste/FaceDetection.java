package br.com.teste;

import static org.bytedeco.javacpp.helper.opencv_imgproc.cvFindContours;
import static org.bytedeco.javacpp.helper.opencv_objdetect.cvHaarDetectObjects;
import static org.bytedeco.javacpp.opencv_calib3d.cvRodrigues2;
import static org.bytedeco.javacpp.opencv_core.IPL_DEPTH_8U;
import static org.bytedeco.javacpp.opencv_core.cvClearMemStorage;
import static org.bytedeco.javacpp.opencv_core.cvGetSeqElem;
import static org.bytedeco.javacpp.opencv_core.cvLoad;
import static org.bytedeco.javacpp.opencv_core.cvPoint;
import static org.bytedeco.javacpp.opencv_imgproc.CV_AA;
import static org.bytedeco.javacpp.opencv_imgproc.CV_BGR2GRAY;
import static org.bytedeco.javacpp.opencv_imgproc.CV_CHAIN_APPROX_SIMPLE;
import static org.bytedeco.javacpp.opencv_imgproc.CV_RETR_LIST;
import static org.bytedeco.javacpp.opencv_imgproc.CV_THRESH_BINARY;
import static org.bytedeco.javacpp.opencv_imgproc.cvCvtColor;
import static org.bytedeco.javacpp.opencv_imgproc.cvRectangle;
import static org.bytedeco.javacpp.opencv_imgproc.cvThreshold;
import static org.bytedeco.javacpp.opencv_imgproc.cvWarpPerspective;
import static org.bytedeco.javacpp.opencv_objdetect.CV_HAAR_DO_CANNY_PRUNING;

import java.io.File;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.opencv_core.CvContour;
import org.bytedeco.javacpp.opencv_core.CvMat;
import org.bytedeco.javacpp.opencv_core.CvMemStorage;
import org.bytedeco.javacpp.opencv_core.CvPoint;
import org.bytedeco.javacpp.opencv_core.CvRect;
import org.bytedeco.javacpp.opencv_core.CvScalar;
import org.bytedeco.javacpp.opencv_core.CvSeq;
import org.bytedeco.javacpp.opencv_core.IplImage;
import org.bytedeco.javacpp.opencv_objdetect;
import org.bytedeco.javacpp.opencv_objdetect.CvHaarClassifierCascade;
import org.bytedeco.javacpp.indexer.DoubleIndexer;
import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.FrameGrabber;
import org.bytedeco.javacv.OpenCVFrameConverter;

public class FaceDetection {

	public static void main(String[] args) throws Exception {

		ClassLoader classLoader = new FaceDetection().getClass().getClassLoader();
		File file = new File(classLoader.getResource("haarcascade_frontalface_alt.xml").getFile());
		
		String classifierName = file.getAbsolutePath();
		
		// Preload the opencv_objdetect module to work around a known bug.
		Loader.load(opencv_objdetect.class);

		// We can "cast" Pointer objects by instantiating a new object of the
		// desired class.
		CvHaarClassifierCascade classifier = new CvHaarClassifierCascade(cvLoad(classifierName));
		if (classifier.isNull()) {
			System.err.println("Error loading classifier file \"" + classifierName + "\".");
			System.exit(1);
		}

		// The available FrameGrabber classes include OpenCVFrameGrabber
		// (opencv_videoio),
		// DC1394FrameGrabber, FlyCaptureFrameGrabber, OpenKinectFrameGrabber,
		// PS3EyeFrameGrabber, VideoInputFrameGrabber, and FFmpegFrameGrabber.
		Thread.sleep(1000);
		FrameGrabber grabber = FrameGrabber.createDefault(0);
		Thread.sleep(1000);
		grabber.start();
		Thread.sleep(1000);

		// CanvasFrame, FrameGrabber, and FrameRecorder use Frame objects to
		// communicate image data.
		// We need a FrameConverter to interface with other APIs (Android, Java
		// 2D, or OpenCV).
		OpenCVFrameConverter.ToIplImage converter = new OpenCVFrameConverter.ToIplImage();

		// FAQ about IplImage and Mat objects from OpenCV:
		// - For custom raw processing of data, createBuffer() returns an NIO
		// direct
		// buffer wrapped around the memory pointed by imageData, and under
		// Android we can
		// also use that Buffer with Bitmap.copyPixelsFromBuffer() and
		// copyPixelsToBuffer().
		// - To get a BufferedImage from an IplImage, or vice versa, we can
		// chain calls to
		// Java2DFrameConverter and OpenCVFrameConverter, one after the other.
		// - Java2DFrameConverter also has static copy() methods that we can use
		// to transfer
		// data more directly between BufferedImage and IplImage or Mat via
		// Frame objects.
		IplImage grabbedImage = converter.convert(grabber.grab());
		int width = grabbedImage.width();
		int height = grabbedImage.height();
		IplImage grayImage = IplImage.create(width, height, IPL_DEPTH_8U, 1);
		IplImage rotatedImage = grabbedImage.clone();

		// Objects allocated with a create*() or clone() factory method are
		// automatically released
		// by the garbage collector, but may still be explicitly released by
		// calling release().
		// You shall NOT call cvReleaseImage(), cvReleaseMemStorage(), etc. on
		// objects allocated this way.
		CvMemStorage storage = CvMemStorage.create();

		// CanvasFrame is a JFrame containing a Canvas component, which is
		// hardware accelerated.
		// It can also switch into full-screen mode when called with a
		// screenNumber.
		// We should also specify the relative monitor/camera response for
		// proper gamma correction.
		CanvasFrame frame = new CanvasFrame("Image Capture", CanvasFrame.getDefaultGamma() / grabber.getGamma());

		// Let's create some random 3D rotation...
		CvMat randomR = CvMat.create(3, 3), randomAxis = CvMat.create(3, 1);
		
		// We can easily and efficiently access the elements of matrices and
		// images
		// through an Indexer object with the set of get() and put() methods.
		DoubleIndexer Ridx = randomR.createIndexer(), axisIdx = randomAxis.createIndexer();
		
		axisIdx.put(0, 0, 0, 0);
		
		cvRodrigues2(randomAxis, randomR, null);
		double f = (width + height) / 2.0;
		Ridx.put(0, 2, Ridx.get(0, 2) * f);
		Ridx.put(1, 2, Ridx.get(1, 2) * f);
		Ridx.put(2, 0, Ridx.get(2, 0) / f);
		Ridx.put(2, 1, Ridx.get(2, 1) / f);
		System.out.println(Ridx);

		// We can allocate native arrays using constructors taking an integer as
		// argument.
		CvPoint hatPoints = new CvPoint(3);

		while (frame.isVisible() && (grabbedImage = converter.convert(grabber.grab())) != null) {
			cvClearMemStorage(storage);

			// Let's try to detect some faces! but we need a grayscale image...
			cvCvtColor(grabbedImage, grayImage, CV_BGR2GRAY);
			CvSeq faces = cvHaarDetectObjects(grayImage, classifier, storage, 1.1, 3, CV_HAAR_DO_CANNY_PRUNING);
			
			int total = faces.total();
			for (int i = 0; i < total; i++) {
				CvRect r = new CvRect(cvGetSeqElem(faces, i));
				int x = r.x(), y = r.y(), w = r.width(), h = r.height();
				cvRectangle(grabbedImage, cvPoint(x, y), cvPoint(x + w, y + h), CvScalar.RED, 1, CV_AA, 0);
			}

			// Let's find some contours! but first some thresholding...
			cvThreshold(grayImage, grayImage, 64, 255, CV_THRESH_BINARY);

			// To check if an output argument is null we may call either
			// isNull() or equals(null).
			CvSeq contour = new CvSeq(null);
			cvFindContours(grayImage, storage, contour, Loader.sizeof(CvContour.class), CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
			
			cvWarpPerspective(grabbedImage, rotatedImage, randomR);

			Frame rotatedFrame = converter.convert(rotatedImage);
			frame.showImage(rotatedFrame);
		}
		frame.dispose();
		grabber.stop();
		
	}
	
}
