#include "trackClass.h"
#include<fstream>
#include <time.h>

int main()
{
	string hog_xml = "hog_64x64_16x32_8x16_8x16_1.xml";
	string hog_like_xml = "4bit_40_16.xml";
	string lbp_xml = "c1_lbp_958_18.xml";

	Tracking hog_tracking;
	Tracking hog_like_tracking;
	Tracking lbp_tracking;

	//double aff[]={15, 15, 0.32, 0.32, 0.02, 2};
	double aff[]={15, 15, 1.0, 1.0, 0.02, 2};		//x, y, width, height, rotation, scale

	if (hog_like_tracking.initialize(HOG_LIKE, hog_like_xml, 100, aff))
	{
		cout << "hog-like initialize fail!" << endl;
		return -1;
	}

	if (hog_tracking.initialize(HOG, hog_xml, 25, aff))
	{
		cout << "hog initialize fail!" << endl;
		return -1;
	}

	if (lbp_tracking.initialize(LBP, lbp_xml, 50, aff))
	{
		cout << "lbp initialize fail!" << endl;
		return -1;
	}

	char type = 0;
	char type_name[56] = {0};
	bool detect = false;
	VideoCapture cap(1);

	cap.set(CV_CAP_PROP_FRAME_HEIGHT, 480);
	cap.set(CV_CAP_PROP_FRAME_WIDTH, 640);
	Mat test_image;
	cap >> test_image;
	const int image_height = test_image.rows;
	const int image_width = test_image.cols;
	const int image_type = test_image.type();

	if (cap.isOpened())
	{
		while (1)
		{
			Mat original_image;
			cap >> original_image;
	        Mat bw_image = cv::Mat(image_height, image_width, CV_8UC1);
	        Mat image = cv::Mat(image_height, image_width, image_type);
//			TSLskinSegment(original_image, bw_image);
			original_image.copyTo(image/*, bw_image*/);
			//Mat image;
			//cap >> image;
			
			//resize(image, image, Size(480, 360));

			double t = (double)cvGetTickCount();
			switch (type)
			{
			case '1':
				hog_tracking.process(image, detect);
				hog_tracking.drawResult(original_image, 0);
				break;
			case '2':
				hog_like_tracking.process(image, detect);
				hog_like_tracking.drawResult(original_image, 0);
				break;
			case '3':
				lbp_tracking.process(image, detect);
				lbp_tracking.drawResult(original_image, 0);
				break;
			default:
				break;
			}

			t = (double)cvGetTickCount() - t;
			char show_chars[128] = {0};
			sprintf( show_chars, "%s detection time = %g ms\0", type_name, t/((double)cvGetTickFrequency()*1000.) );
			cv::putText(original_image, show_chars, Point(20, 50), 0, 0.75, CV_RGB(255, 0, 0), 1.5);
			imshow("test", original_image);

			switch (waitKey(30))
			{
			case '1':
				type = '1';
				strcpy(type_name, "hog");
				break;
			case '2':
				type = '2';
				strcpy(type_name, "hog_like");
				break;
			case '3':
				strcpy(type_name, "lbp");
				type = '3';
				break;
			case 'd':
				detect = true;
				break;
			case 't':
				detect = false;
				break;
			case 'c':
				return 0;
			default:
				break;
			}
		}
	}
	else
	{
		cout << "opening capture failed!" << endl;
	}
}