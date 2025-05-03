import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgproc.Moments;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.Videoio;
import org.opencv.highgui.HighGui;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class HandGestureRecognition {

    static {
        try {
            System.load("C:\\Users\\Moacir\\Downloads\\opencv\\build\\java\\x64\\opencv_java4110.dll");
            System.out.println("OpenCV " + Core.VERSION + " carregado com sucesso");
        } catch (UnsatisfiedLinkError e) {
            System.err.println("Erro ao carregar OpenCV: " + e.getMessage());
            System.exit(1);
        }
    }

    // Parâmetros ajustáveis para detecção de pele
    private static final Scalar LOWER_SKIN = new Scalar(0, 20, 70);
    private static final Scalar UPPER_SKIN = new Scalar(20, 255, 255);
    private static final int MIN_AREA = 5000;

    public static void main(String[] args) {
        VideoCapture camera = initializeCamera();
        if (camera == null) return;

        Mat frame = new Mat();
        while (true) {
            if (!camera.read(frame) || frame.empty()) {
                System.out.println("Problema ao capturar frame");
                break;
            }

            Mat processedFrame = detectHand(frame);
            HighGui.imshow("Detecção de Mão", processedFrame);

            if (HighGui.waitKey(30) == 27) break;
        }

        camera.release();
        HighGui.destroyAllWindows();
    }

    private static VideoCapture initializeCamera() {
        for (int i = 0; i < 3; i++) {
            VideoCapture camera = new VideoCapture(i);
            if (camera.isOpened()) {
                System.out.println("Câmera " + i + " inicializada");
                camera.set(Videoio.CAP_PROP_FRAME_WIDTH, 640);
                camera.set(Videoio.CAP_PROP_FRAME_HEIGHT, 480);
                camera.set(Videoio.CAP_PROP_FPS, 30);
                return camera;
            }
            camera.release();
        }
        System.out.println("Nenhuma câmera encontrada!");
        return null;
    }

    private static Mat detectHand(Mat frame) {
        Mat processedFrame = frame.clone();

        // 1. Pré-processamento
        Mat hsv = new Mat();
        Imgproc.cvtColor(frame, hsv, Imgproc.COLOR_BGR2HSV);

        // 2. Detecção de pele
        Mat skinMask = new Mat();
        Core.inRange(hsv, LOWER_SKIN, UPPER_SKIN, skinMask);

        // 3. Melhoria da máscara
        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(11, 11));
        Imgproc.morphologyEx(skinMask, skinMask, Imgproc.MORPH_CLOSE, kernel);
        Imgproc.GaussianBlur(skinMask, skinMask, new Size(3, 3), 0);

        // 4. Converter máscara para BGR para exibição
        Mat maskBGR = new Mat();
        Imgproc.cvtColor(skinMask, maskBGR, Imgproc.COLOR_GRAY2BGR);

        // 5. Encontrar contornos
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(skinMask, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        if (!contours.isEmpty()) {
            int maxIdx = findLargestContour(contours);
            if (maxIdx != -1 && Imgproc.contourArea(contours.get(maxIdx)) > MIN_AREA) {
                drawHandContour(processedFrame, contours.get(maxIdx));
            }
        }

        // 6. Combinar imagens para exibição
        Mat combined = new Mat();
        List<Mat> imagesToCombine = new ArrayList<>();
        imagesToCombine.add(processedFrame);
        imagesToCombine.add(maskBGR);

        Core.hconcat(imagesToCombine, combined);

        return combined;
    }

    private static int findLargestContour(List<MatOfPoint> contours) {
        double maxArea = 0;
        int maxIdx = -1;
        for (int i = 0; i < contours.size(); i++) {
            double area = Imgproc.contourArea(contours.get(i));
            if (area > maxArea) {
                maxArea = area;
                maxIdx = i;
            }
        }
        return maxIdx;
    }

    private static void drawHandContour(Mat frame, MatOfPoint contour) {
        // Desenhar contorno
        Imgproc.drawContours(frame, Arrays.asList(contour), -1, new Scalar(0, 255, 0), 3);

        // Calcular e desenhar centro
        Moments moments = Imgproc.moments(contour);
        Point center = new Point(moments.get_m10() / moments.get_m00(), moments.get_m01() / moments.get_m00());
        Imgproc.circle(frame, center, 8, new Scalar(255, 0, 0), -1);

        // Calcular e desenhar casco convexo
        MatOfInt hull = new MatOfInt();
        Imgproc.convexHull(contour, hull);

        List<Point> hullPoints = new ArrayList<>();
        for (int i = 0; i < hull.rows(); i++) {
            int idx = (int)hull.get(i, 0)[0];
            hullPoints.add(contour.toList().get(idx));
        }

        MatOfPoint hullContour = new MatOfPoint();
        hullContour.fromList(hullPoints);
        Imgproc.drawContours(frame, Arrays.asList(hullContour), -1, new Scalar(0, 0, 255), 2);

        // Adicionar texto informativo
        String text = String.format("Area: %.0f", Imgproc.contourArea(contour));
        Imgproc.putText(frame, text, new Point(center.x + 20, center.y),
                Imgproc.FONT_HERSHEY_SIMPLEX, 0.7, new Scalar(255, 255, 255), 2);
    }
}