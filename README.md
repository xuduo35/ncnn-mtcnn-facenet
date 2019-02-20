# ncnn-mtcnn-facenet

Combine mtcnn and mobilefacenet to do face recognization for MacOS/Linux/iOS

For MacOS/Linux, check README.md under MacOS directory.

For iOS:

Need to add ncnn.framework/opencv2.framework/openmp.framework to project first. Then run in XCode directly.

Reference: https://medium.com/@yiweini/opencv-with-swift-step-by-step-c3cc1d1ee5f1

Default sample image:

![main](https://github.com/xuduo35/ncnn-mtcnn-facenet/blob/master/images/main.jpg)

Red rectangle if recognization:

![detected](https://github.com/xuduo35/ncnn-mtcnn-facenet/blob/master/images/detected.jpg)

Green dots for landmarks if not recognization:

![undetected](https://github.com/xuduo35/ncnn-mtcnn-facenet/blob/master/images/undetected.jpg)
