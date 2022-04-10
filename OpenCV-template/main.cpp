#include "pch.h"
#include "preprocess.h"
#include "sharpness.h"

using namespace std;
using namespace std::chrono;
using namespace cv;
namespace plt = matplotlibcpp;

void test_rgb2graywithfilter()
{
    Mat image_gray, imageres;

    vector<double> res;
    vector<double> time_vec;

    for (int i = 0; i < 150; i++)
    {
        
        string path = "E:/MyData/2022-01-21/ÈéÏÙÏËÎ¬Áö/" + to_string(i) + ".bmp";

        Mat image = imread(path, 1);
        if (image.empty())   std::cout << "read error" << endl;
        
        auto start = std::chrono::steady_clock::now();

        cvtColor(image, image_gray, CV_BGR2GRAY);
        cv::medianBlur(image_gray, imageres, 3);

        auto end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        time_vec.push_back(duration.count());
    }

    float avgtime = accumulate(begin(time_vec), end(time_vec), 0.0) / time_vec.size();

    double variance = 0.0;
    for (int i = 0; i < time_vec.size(); i++)  {  variance += pow(time_vec[i] - avgtime, 2);  }
    variance = variance / time_vec.size();
    
    double standard_deviation = sqrt(variance);
    double maxtime = *max_element(time_vec.begin(), time_vec.end());

    cout << "avgtime: " << avgtime << "ms" << '\n';
    cout << "size: " << time_vec.size() << '\n';
    cout << "max_time: " << maxtime << "ms" << '\n';
    cout << "standard_deviation: " << standard_deviation << '\n';

    /*plt::plot(time_vec, "bo-");
    plt::title("opencv_time");
    plt::show();*/
}

int main()
{
    vector<double> res;
    Mat image_gray, imageres;

    for (int i = 0; i < 150; ++ i)
    {
        string path = "E:/MyData/2022-01-21/ÈéÏÙÏËÎ¬Áö/" + to_string(i) + ".bmp";

        Mat image = imread(path, 1);
        if (image.empty())   std::cout << "read error" << endl;

        //cvtColor(image, image_gray, CV_BGR2GRAY);
        //cv::medianBlur(image_gray, imageres, 3);
        //double f = brenner(image);
        //double f = tenengard(image);
        //double f = laplacian(image);
        //double f = smd(image);
        //double f = smd2(image);
        //double f = energy_gradient(image);
        //double f = eav(image);
        double f = FC(image);
        res.push_back(f);
    }

    plt::plot(res);
    plt::title("fvalue");
    plt::show();

    //res.push_back(f);

    
    
    //cout << image.size().width << ' ' << image.size().height << endl;

    //double time_consume = get_time(rgb2gray_cv, image, image_gray);
    //cout << time_consume << "ms" << endl;

    /*vector<double> ans;
    ans = get_time(rgb2gray_cv, image, image_gray);
    cout << "cvt:" << endl;
    for (double i : ans)
    {
        cout << i << ' ';
    }
    cout << endl;

    vector<double> ans1;
    ans = get_time(rgbtogray_6, image, image_gray);
    cout << "cv6g:" << endl;
    for (double i : ans1)
    {
        cout << i << ' ';
    }
    cout << endl;*/
    //string path = "E:/MyData/2022-01-21/ÈéÏÙÏËÎ¬Áö/75.bmp";
    //Mat image = imread(path, 1);
    //cvtColor(image, image_gray, CV_BGR2GRAY);
    //cv::medianBlur(image_gray, imageres, 3);

    ////´´½¨´°¿Ú ÏÔÊ¾Í¼Ïñ
    //namedWindow("Display Image", WINDOW_AUTOSIZE);
    //imshow("Display Image", image);
    //waitKey(0);
    return 0;
}
