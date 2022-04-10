#include "pch.h"
#include "preprocess.h"

using namespace std;
using namespace cv;

//// 循环计时 总时间 / count
//double get_time(FunType func, Mat& image, Mat& result)
//{
//	int count = 50;
//	vector<int> vec;
//
//	auto start = std::chrono::steady_clock::now();
//
//	for (int i = 0; i < count; i++)
//		func(image, result);
//
//	auto end = std::chrono::steady_clock::now();
//	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
//
//	double agv_time = duration.count() / count;
//	return agv_time;
//}


// 计时 直接返回向量
vector<double> get_time(FunType func, Mat& image, Mat& result)
{
	int count = 50;
	vector<double> vec;

	for (int i = 0; i < count; i++)
	{
		auto start = std::chrono::steady_clock::now();
		
		func(image, result);

		auto end = std::chrono::steady_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

		double agv_time = duration.count();

		vec.push_back(agv_time);
	}
	return vec;
}



// opencv 官方 灰度化
void rgb2gray_cv(Mat& image, Mat& image_gray)
{
	cvtColor(image, image_gray, CV_BGR2GRAY);
}

void rgbtogray_1(Mat& image, Mat& image_gray)
{
	int height = image.rows;
	int width = image.cols;
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			int B = image.at<cv::Vec3b>(i, j)[0];
			int G = image.at<cv::Vec3b>(i, j)[1];
			int R = image.at<cv::Vec3b>(i, j)[2];
			image_gray.at<uchar>(i, j) = (uchar)(0.114 * B + 0.587 * G + 0.2989 * R);
		}
	}
}

void rgbtogray_2(Mat& image, Mat& result)
{
	cv::Mat_<Vec3b>::iterator it = image.begin<Vec3b>();
	cv::Mat_<Vec3b>::iterator itend = image.end<Vec3b>();
	cv::Mat_<uchar>::iterator itt = result.begin<uchar>();
	for (; it != itend; ++it, ++itt)
	{
		int B = (*it)[0];
		int G = (*it)[1];
		int R = (*it)[2];
		(*itt) = (uchar)(0.114 * B + 0.587 * G + 0.2989 * R);
	}
}

void rgbtogray_3(Mat& image, Mat& result)
{
	int n = image.rows * image.cols;
	uchar* src = image.ptr<uchar>(0);
	uchar* dest = result.ptr<uchar>(0);
	for (int i = 0; i < n; i++)
	{
		int B = *(src);
		int G = *(src + 1);
		int R = *(src + 2);
		*dest = (uchar)(0.114 * B + 0.587 * G + 0.2989 * R);
		dest++;
		src += 3;
	}
}

void rgbtogray_4(Mat& image, Mat& result)
{
	int n = image.rows * image.cols;
	uchar* src = image.ptr<uchar>(0);
	uchar* dest = result.ptr<uchar>(0);
	for (int i = 0; i < n; i++)
	{
		int B = *(src);
		int G = *(src + 1);
		int R = *(src + 2);
		*dest = cv::saturate_cast<uchar>((1140 * B + 5870 * G + 2989 * R) / 10000);
		dest++;
		src += 3;
	}
}

void rgbtogray_5(Mat& image, Mat& result)
{
	int n = image.rows * image.cols;
	uchar* src = image.ptr<uchar>(0);
	uchar* dest = result.ptr<uchar>(0);
	for (int i = 0; i < n; i++)
	{
		int B = *(src);
		int G = *(src + 1);
		int R = *(src + 2);
		*dest = cv::saturate_cast<uchar>((4898 * B + 9618 * G + 1868 * R) >> 14);
		dest++;
		src += 3;
	}
}

void rgbtogray_6(Mat& image, Mat& result)
{
	int LUT_B[256] = { 0 };
	int LUT_G[256] = { 0 };
	int LUT_R[256] = { 0 };
	for (int i = 0; i < 256; i++)
	{
		LUT_B[i] = (i * 4898);
		LUT_G[i] = (i * 9618);
		LUT_R[i] = (i * 1868);
	}
	int n = image.rows * image.cols;
	uchar* src = image.ptr<uchar>(0);
	uchar* dest = result.ptr<uchar>(0);
	for (int i = 0; i < n; i++)
	{
		int B = *(src);
		int G = *(src + 1);
		int R = *(src + 2);
		*dest = cv::saturate_cast<uchar>((LUT_B[B] + LUT_G[G] + LUT_R[R]) >> 14);
		dest++;
		src += 3;
	}
}

