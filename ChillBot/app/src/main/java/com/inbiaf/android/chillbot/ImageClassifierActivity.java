package com.inbiaf.android.chillbot;

import android.app.Activity;
import android.graphics.Bitmap;
import android.media.ImageReader;
import android.os.Bundle;
import android.util.Log;
import android.view.KeyEvent;
import android.view.View;
import android.view.WindowManager;
import android.widget.ImageView;
import android.widget.TextView;

import com.google.firebase.database.DatabaseReference;
import com.google.firebase.database.FirebaseDatabase;
import com.inbiaf.android.chillbot.classifier.Recognition;
import com.inbiaf.android.chillbot.classifier.TensorFlowHelper;

import org.tensorflow.lite.Interpreter;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;

public class ImageClassifierActivity extends Activity {
    private static final String TAG = "ImageClassifierActivity";

    /**
     * Camera image capture size
     */
    private static final int PREVIEW_IMAGE_WIDTH = 640;
    private static final int PREVIEW_IMAGE_HEIGHT = 480;
    /**
     * Image dimensions required by TF model
     */
    private static final int TF_INPUT_IMAGE_WIDTH = 224;
    private static final int TF_INPUT_IMAGE_HEIGHT = 224;
    /**
     * Dimensions of model inputs.
     */
    private static final int DIM_BATCH_SIZE = 1;
    private static final int DIM_PIXEL_SIZE = 3;
    /**
     * TF model asset files
     */
    private static final String LABELS_FILE = "labels.txt";
    private static final String MODEL_FILE = "chill-bot.tflite";

    private boolean mProcessing;

    private ImageView mImage;
    private TextView mResultText;
    private TextView cameraButton;

    private Interpreter mTensorFlowLite;
    private List<String> mLables;

    private CameraHandler mCameraHandler;
    private ImagePreprocessor mImagePreprocessor;

    /**
     * Initialize the classifier that will be used to process images.
     */
    private void initClassifier() {
        try {
            mTensorFlowLite = new Interpreter(TensorFlowHelper.loadModelFile(this, MODEL_FILE));
            mLables = TensorFlowHelper.readLabels(this, LABELS_FILE);
        } catch (IOException e) {
            Log.w(TAG, "Unable to initialize TensorFlow Lite.", e);
        }

    }

    /**
     * Clean up the resources used by the classifier.
     */
    private void destroyClassifier() {
        mTensorFlowLite.close();
    }

    /**
     * Process an image and identify what is in it. When done, the method
     * {@link #onPhotoRecognitionReady(Collection)} must be called with the results of
     * the image recognition process.
     *
     * @param image Bitmap containing the image to be classified. The image can be
     *              of any size, but preprocessing might occur to resize it to the
     *              format expected by the classification process, which can be time
     *              and power consuming.
     */
    private void doRecognize(Bitmap image) {
        byte[][] confidencePerLabel = new byte[1][mLables.size()];

        int[] intValues = new int[TF_INPUT_IMAGE_WIDTH * TF_INPUT_IMAGE_WIDTH];
        ByteBuffer imgData = ByteBuffer.allocateDirect(
                DIM_BATCH_SIZE * TF_INPUT_IMAGE_WIDTH * TF_INPUT_IMAGE_HEIGHT * DIM_PIXEL_SIZE);
        imgData.order(ByteOrder.nativeOrder());

        TensorFlowHelper.convertBitmapToByteBuffer(image, intValues, imgData);

        mTensorFlowLite.run(imgData, confidencePerLabel);
        for (int i = 0; i < confidencePerLabel[0].length; i++) {
            Log.d("DevLogger", "Confidence = " + confidencePerLabel[0][i]);
        }

        Collection<Recognition> results = TensorFlowHelper.getBestResults(confidencePerLabel, mLables);

        boolean hasCoke = false;
        boolean hasPerrier = false;
        boolean hasOther = false;

        for (Recognition result : results) {
            switch (result.getTitle()) {
                case "cocacola":
                    if (result.getConfidence() > 0.5) {
                        hasCoke = true;
                    }
                    break;
                case "perrier":
                    if (result.getConfidence() > 0.5) {
                        hasPerrier = true;
                    }
                    break;

                case "other":
                    if (result.getConfidence() > 0.5) {
                        hasOther = true;
                    }
                    break;
            }
        }

        final Drinks drinksData = new Drinks(hasCoke, hasPerrier, hasOther);

        new Thread(new Runnable() {
            public void run() {
                DatabaseReference database = FirebaseDatabase.getInstance().getReference();
                database.child("drinks").setValue(drinksData);
            }
        }).start();

        onPhotoRecognitionReady(results);
    }

    /**
     * Initialize the camera that will be used to capture images.
     */
    private void initCamera() {
        mImagePreprocessor = new ImagePreprocessor(PREVIEW_IMAGE_WIDTH, PREVIEW_IMAGE_HEIGHT,
                TF_INPUT_IMAGE_WIDTH, TF_INPUT_IMAGE_HEIGHT);
        mCameraHandler = CameraHandler.getInstance();
        mCameraHandler.initializeCamera(this,
                PREVIEW_IMAGE_WIDTH, PREVIEW_IMAGE_HEIGHT, null,
                new ImageReader.OnImageAvailableListener() {
                    @Override
                    public void onImageAvailable(ImageReader imageReader) {
                        Bitmap bitmap = mImagePreprocessor.preprocessImage(imageReader.acquireNextImage());
                        onPhotoReady(bitmap);
                    }
                });
    }

    /**
     * Clean up resources used by the camera.
     */
    private void closeCamera() {
        mCameraHandler.shutDown();
    }

    /**
     * Load the image that will be used in the classification process.
     * When done, the method {@link #onPhotoReady(Bitmap)} must be called with the image.
     */
    private void loadPhoto() {
        mCameraHandler.takePicture();
    }


    // --------------------------------------------------------------------------------------
    // NOTE: The normal codelab flow won't require you to change anything below this line,
    // although you are encouraged to read and understand it.

    @Override
    protected void onCreate(final Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.activity_camera);
        mImage = findViewById(R.id.imageView);
        mResultText = findViewById(R.id.resultText);
        cameraButton = findViewById(R.id.camera_button);

        updateStatus(getString(R.string.initializing));
        initCamera();
        initClassifier();
        initButton();
        updateStatus(getString(R.string.button_message));
    }

    /**
     * Register a GPIO button that, when clicked, will generate the {@link KeyEvent#KEYCODE_ENTER}
     * key, to be handled by {@link #onKeyUp(int, KeyEvent)} just like any regular keyboard
     * event.
     * <p>
     * If there's no button connected to the board, the doRecognize can still be triggered by
     * sending key events using a USB keyboard or `adb shell input keyevent 66`.
     */
    private void initButton() {
        cameraButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Log.d("DevLogger", "cameraButton button press");
                if (mProcessing) {
                    updateStatus("Still processing, please wait");
                }

                updateStatus("Running photo recognition");
                mProcessing = true;
                loadPhoto();
            }
        });
    }

    /**
     * Image capture process complete
     */
    private void onPhotoReady(Bitmap bitmap) {
        mImage.setImageBitmap(bitmap);
        doRecognize(bitmap);
    }

    /**
     * Image classification process complete
     */
    private void onPhotoRecognitionReady(Collection<Recognition> results) {
        updateStatus(formatResults(results));
        mProcessing = false;
    }

    /**
     * Format results list for display
     */
    private String formatResults(Collection<Recognition> results) {
        if (results == null || results.isEmpty()) {
            return getString(R.string.empty_result);
        } else {
            StringBuilder sb = new StringBuilder();
            Iterator<Recognition> it = results.iterator();
            int counter = 0;
            while (it.hasNext()) {
                Recognition r = it.next();
                sb.append(r.getTitle());
                counter++;
                if (counter < results.size() - 1) {
                    sb.append(", ");
                } else if (counter == results.size() - 1) {
                    sb.append(" or ");
                }
            }

            return sb.toString();
        }
    }

    /**
     * Report updates to the display and log output
     */
    private void updateStatus(String status) {
        Log.d(TAG, status);
        mResultText.setText(status);
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        try {
            destroyClassifier();
        } catch (Throwable t) {
            // close quietly
        }
        try {
            closeCamera();
        } catch (Throwable t) {
            // close quietly
        }
    }
}
