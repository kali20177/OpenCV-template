#pragma once
#include "pch.h"

using namespace std;
using namespace cv;

typedef void (*FunType)(Mat &image, Mat &result);

//灰度化函数列表
void rgbtogray_1(Mat& image, Mat& result);
void rgbtogray_2(Mat& image, Mat& result);
void rgbtogray_3(Mat& image, Mat& result);
void rgbtogray_4(Mat& image, Mat& result);
void rgbtogray_5(Mat& image, Mat& result);
void rgbtogray_6(Mat& image, Mat& result);
void rgb2gray_cv(Mat& image, Mat& image_gray);

vector<double> get_time(FunType func, Mat& image, Mat& result);

