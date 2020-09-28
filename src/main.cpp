#include<slic.h>
#include<opencv2/opencv.hpp>
#include<iostream>

using namespace image_split;

int main(int argc,char** argv)
{
   
    std::string image_path = argv[1];
   
    cv::Mat image = cv::imread(image_path,1);

    if(image.empty())
    {
        std::cerr<<"image is empty,please check..."<<std::endl;
    }

     //convert rgb-image to lab
    cv::Mat image_filter;
    cv::GaussianBlur(image,image_filter,cv::Size(7,7),3,3);
    cv::GaussianBlur(image,image_filter,cv::Size(7,7),3,3);

    cv::Mat labImg;
    cv::cvtColor(image,labImg,CV_BGR2Lab);

     //parameter
    int width = image.cols;
    int height = image.rows;

    int nr_superpixels_num = 1000;
    //nc表示空间和像素颜色的相对重要性的度量
    //当nc大时，空间邻近性更重要，并且所得到的超像素更紧凑（即它们具有更低的面积与周长比）。
    //当nc小时，所得到的超像素更紧密地粘附到图像边界，但是具有较小的规则尺寸和形状。
    //当使用CIELAB色彩空间时，m可以在[1,40]的范围内。
    int nc = 10; //0~40

    float step = sqrtf((width * height) / (float)nr_superpixels_num);

    /* Perform the SLIC superpixel algorithm. */
    Slic slic;
    IplImage IplLabImg = labImg;
    slic.generate_superpixels(&IplLabImg, step, nc);

    /* Display the contours and show the result. */
    std::vector<cv::Point> contours;
    IplImage IplImg = image;
    slic.display_contours(&IplImg, CV_RGB(255,0,0), contours);
    cv::imshow("slic image", image);
    cv::imwrite("slic_image.png",image);
    cv::waitKey(0);

    std::vector<std::vector<cv::Point>> spPoints;

    slic.getClusterPoints(spPoints);

    cv::Mat slic_img(height, width, CV_8UC3, cv::Scalar(0,0,0));
    
    for (int i = 0, size = spPoints.size(); i < size; ++i) {

         uint8_t r = rand() % 256;
         uint8_t g = rand() % 256;
         uint8_t b = rand() % 256;
         cv::Vec3b color(b,g,r);
         for (auto &pt : spPoints[i]) {
             slic_img.at<cv::Vec3b>(pt) = color;
         }

   }
   cv::namedWindow("slic_img",1);
   cv::imshow("slic_img",slic_img);
   cv::waitKey();

   return 1;
}
