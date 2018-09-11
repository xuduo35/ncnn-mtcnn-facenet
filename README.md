# ncnn-mtcnn-facenet

combine mtcnn and facenet to do face recognization

How to run it:

1. install OpenCV

2. git clone https://github.com/Tencent/ncnn.git

3. git clone this repo in the same directory with ncnn.

4. cd ncnn-mtcnn-facenet

5. mkdir build

6. cmake ..

7. make

8. ./facenet -sample ../yangmi.jpg

9. ./facenet -facenet your_mp4_file

Integrate code and models from:

https://github.com/moli232777144/mtcnn_ncnn

https://github.com/GRAYKEY/mobilefacenet_ncnn
