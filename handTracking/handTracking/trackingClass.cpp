#include "trackClass.h"

#include <time.h>
#include <iostream>
#include <vector>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;

Classifier::Classifier(CLASSIFIER classifier_type): m_classifier_type(classifier_type)
{
}

Classifier::~Classifier()
{
}

HOGLIKEClassifier::HOGLIKEClassifier():Classifier(HOG_LIKE)
{
	size = 50;
}
HOGLIKEClassifier::~HOGLIKEClassifier()
{
}

bool HOGLIKEClassifier::load(string xml_file)
{
	return m_classifier.load(xml_file);
}

void HOGLIKEClassifier::detectHand(Mat& image, vector<Rect>& hands)
{
	m_classifier.detectMultiScale(image, hands, 1.2, 1, 0, Size(40, 40));
}

LBPClassifier::LBPClassifier():Classifier(LBP)
{
	size = 50;
}
LBPClassifier::~LBPClassifier()
{
}

bool LBPClassifier::load(string xml_file)
{
	return m_classifier.load(xml_file);
}

void LBPClassifier::detectHand(Mat& image, vector<Rect>& hands)
{
	m_classifier.detectMultiScale(image, hands, 1.05, 1, 0, Size(40, 40));
}

HOGClassifier::HOGClassifier(): Classifier(HOG)
{
	size = 70;
}

HOGClassifier::~HOGClassifier()
{
}

bool HOGClassifier::load(string xml_file)
{
	return m_classifier.load(xml_file);
}

void HOGClassifier::detectHand(Mat& image, vector<Rect>& hands)
{
	m_classifier.detectMultiScale(image, hands, 1, Size(8, 8), Size(8, 8), 1.12, 3);
}

MotionHistory::MotionHistory(float alpha):
	m_is_ok(true), m_alpha(alpha), m_frame_index(-1)
{
	for (int i = 0; i < 2; i++)
	{
		m_pre_acc[i] = 0.0f;
		m_pre_vel[i] = 0.0f;
		m_pre_para[i] = 0.0f;
	}
}

MotionHistory::~MotionHistory()
{
}

void MotionHistory::Update(float cur_para[2])
{
	if (!m_is_ok)
	{
		return;
	}

	float cur_vel[2] = {0};

	if (m_frame_index >= 0)
	{
		cur_vel[0] = cur_para[0] - m_pre_para[0];
		cur_vel[1] = cur_para[1] - m_pre_para[1];
	}

	if (m_frame_index >= 1)
	{
		m_pre_acc[0] = cur_vel[0] - m_pre_vel[0];
		m_pre_acc[1] = cur_vel[1] - m_pre_vel[1];
	}

	m_pre_para[0] = cur_para[0];
	m_pre_para[1] = cur_para[1];

	if (m_frame_index >= 0)
	{
		m_pre_vel[0] = cur_vel[0];
		m_pre_vel[1] = cur_vel[1];
	}

	m_frame_index++;
	return;
}

void MotionHistory::Reset()
{
	m_frame_index = -1;
}

void MotionHistory::Predict(float cur_para[2])
{
	if (!m_is_ok)
	{
		return;
	}

	if (m_frame_index >= 0)
	{
		cur_para[0] = m_pre_para[0];
		cur_para[1] = m_pre_para[1];

		if (m_frame_index >= 1)
		{
			cur_para[0] += m_alpha * m_pre_vel[0];
			cur_para[1] += m_alpha * m_pre_vel[1];

			if (m_frame_index >= 2)
			{
				cur_para[0] += 1.0f * m_alpha * m_pre_acc[0];
				cur_para[1] += 1.0f * m_alpha * m_pre_acc[1];
			}
		}
	}
	return;
}

void MotionHistory::Initialize(float alpha /* = 0.6f */)
{
	m_is_ok = true;
	m_frame_index = -1;
	return;
}

MotionPredict::MotionPredict()
{
	count = 0;
	imgSize = Size(640, 480);
	hand_pos = Rect(0, 0, 0, 0);
}

MotionPredict::~MotionPredict()
{

}

void MotionPredict::update(Mat &frame, ResultRect cur_pos)
{
	Point tl_pt = cur_pos.points[0];
	Point br_pt = cur_pos.points[0];
	for (int i = 1; i < 4; i++)
	{
		if (cur_pos.points[i].x < tl_pt.x)
		{
			tl_pt.x = cur_pos.points[i].x;
		}
		if (cur_pos.points[i].y < tl_pt.y)
		{
			tl_pt.y = cur_pos.points[i].y;
		}
		if (cur_pos.points[i].x > br_pt.x)
		{
			br_pt.x = cur_pos.points[i].x;
		}
		if (cur_pos.points[i].y > br_pt.y)
		{
			br_pt.y = cur_pos.points[i].y;
		}
	}
	hand_pos.x = tl_pt.x;
	hand_pos.y = tl_pt.y;
	hand_pos.width = br_pt.x - tl_pt.x;
	hand_pos.height = br_pt.y - tl_pt.y;

	if (pre_frame.empty())
	{
		frame.copyTo(pre_frame);
	}
	else
	{
		cur_frame.copyTo(pre_frame);
	}
	frame.copyTo(cur_frame);

	imgSize = frame.size();

	hand_pos.x = hand_pos.x < 0 ? 0 : hand_pos.x;
	hand_pos.y = hand_pos.y < 0 ? 0 : hand_pos.y;
	hand_pos.x = hand_pos.br().x > imgSize.width ? (imgSize.width - hand_pos.width) : hand_pos.x;
	hand_pos.y = hand_pos.br().y > imgSize.height ? (imgSize.height - hand_pos.height) : hand_pos.y;
}

bool MotionPredict::predict(float cur_pos[2])
{
	Mat skin_seg(cur_frame.size(), CV_8UC1);
	skinSegment(cur_frame, skin_seg);
	Mat skin_img = skin_seg(hand_pos);

	Mat cur_gray, pre_gray;
	if (cur_frame.channels() > 1)
	{
		cvtColor(cur_frame, cur_gray, CV_RGB2GRAY);
		cvtColor(pre_frame, pre_gray, CV_RGB2GRAY);
	}
	Mat diff;
	Mat curImg = cur_gray(hand_pos);
	Mat preImg = pre_gray(hand_pos);
	absdiff(curImg, preImg, diff);
	threshold(diff, diff, 15, 255, CV_THRESH_BINARY);
	erode(diff,diff,Mat(),Point(-1,-1));
	dilate(diff,diff,Mat(),Point(-1,-1));
	imshow("diff", diff);

	Point2f diffcenter;
	Moments m = moments(diff,true);
	if (m.m00 == 0)
	{
		diffcenter.x = 0;
		diffcenter.y = 0;
	}
	else
	{
		diffcenter.x = m.m10 / m.m00;
		diffcenter.y = m.m01 / m.m00;
	}

	int diffNum(0);
	for (int y = 0; y < diff.rows; y++)
	{
		const uchar* Mi = diff.ptr<uchar>(y);
		for (int x = 0; x < diff.cols; x++)
		{
			if (Mi[x] != 0)		//learn the range
			{
				diffNum++;
			}
		}
	}

	bool diffEmpty(false);
	if (diffNum > 500)
	{
		diffEmpty = false;
	}
	else
	{
		diffEmpty = true;
	}

	int seg_num(0);
	for (int y = 0; y < skin_img.rows; y++)			//limit the range in skin segment
	{
		uchar* Mi = skin_img.ptr<uchar>(y);
		for (int x = 0; x < skin_img.cols; x++)
		{
			if (!diffEmpty)
			{
				double dist = sqrt((x-diffcenter.x)*(x-diffcenter.x) + (y-diffcenter.y)*(y-diffcenter.y));
				if (dist > 0.3 * sqrt(double(skin_img.rows*skin_img.rows + skin_img.cols*skin_img.cols)))
				{
					Mi[x] = 0;
				}
			}
			if (Mi[x] != 0)
			{
				seg_num++;
			}
		}
	}

	if (diffEmpty && seg_num < 1000)
	{
		return false;
	}
	if (diffEmpty || seg_num < 1000)
	{
		count++;
	}
	else if (!diffEmpty || seg_num > 1000)
	{
		count = 0;
	}
	if (count == 20)
	{
		return false;
	}

	Point2f center;
	if (diffEmpty)
	{
		center.x = 0;
		center.y = 0;
	}
	else
	{
		m = moments(skin_img, true);
		if (m.m00 == 0)
		{
			center.x = 0;
			center.y = 0;
		}
		else
		{
			center.x = m.m10 / m.m00;
			center.y = m.m01 / m.m00;
		}
	}

/*
	float dx(0.0);
	float dy(0.0);
	if (center.x == 0 && center.y == 0)
	{
		dx = 0;
		dy = 0;
	}
	else
	{
		dx = center.x - 0.5 * hand_pos.width;
		dy = center.y - 0.5 * hand_pos.height;
	}
	hand_pos.x += dx;
	hand_pos.y += dy;
	hand_pos.x = hand_pos.x < 0 ? 0 : hand_pos.x;
	hand_pos.y = hand_pos.y < 0 ? 0 : hand_pos.y;
	hand_pos.x = hand_pos.br().x > imgSize.width ? (imgSize.width - hand_pos.width) : hand_pos.x;
	hand_pos.y = hand_pos.br().y > imgSize.height ? (imgSize.height - hand_pos.height) : hand_pos.y;*/
	calcWin(skin_img);

	cur_pos[0] = hand_pos.x + 0.5*hand_pos.width;
	cur_pos[1] = hand_pos.y + 0.5*hand_pos.height;

	return true;
}

void MotionPredict::adaThres(Mat &frame)
{
	float y(0), cr(0), cb(0);
	int num(0);
	Mat ycrcb;
	vector<Mat> planes;

	cvtColor(frame, ycrcb, CV_RGB2YCrCb);
	Mat ycrcbROI = ycrcb(hand_pos);
	split(ycrcbROI, planes);
	MatIterator_<uchar> it_cr = planes[1].begin<uchar>(),
		it_cr_end = planes[1].end<uchar>(),
		it_cb = planes[2].begin<uchar>(),
		it_y = planes[0].begin<uchar>();

	for (; it_cr!=it_cr_end; it_y++,it_cr++,it_cb++)
	{
		if ((*it_cr)>95 && (*it_cr)<125 && (*it_cb)>130 && (*it_cb)<167)
		{
			y += *it_y;
			cr += *it_cr;
			cb += *it_cb;
			num++;
		}
	}

	thres[0] = y / num;
	thres[1] = cr / num;
	thres[2] = cb / num;
}

void MotionPredict::skinSegment(const Mat &src, Mat &dst)
{
	Mat ycrcb;
	vector<Mat> planes;

	cvtColor(src,ycrcb,CV_RGB2YCrCb);
	split(ycrcb, planes);
	MatIterator_<uchar> it_cr = planes[1].begin<uchar>(),
		it_cr_end = planes[1].end<uchar>(),
		it_cb = planes[2].begin<uchar>(),
		it_gray = dst.begin<uchar>();

	for (; it_cr != it_cr_end; it_cr++,it_cb++)
	{
		if (abs((*it_cr)-thres[1])<7 && abs((*it_cb)-thres[2])<7)
		{
			*it_gray = 255;
		}
		else
		{
			*it_gray = 0;
		}
		*it_gray++;
	}

	dilate(dst,dst,Mat(),Point(-1,-1));
	erode(dst,dst,Mat(),Point(-1,-1));
	dilate(dst,dst,Mat(),Point(-1,-1));
	erode(dst,dst,Mat(),Point(-1,-1));
}

void MotionPredict::calcWin(const Mat &img)
{
	Point ct(hand_pos.x + 0.5*img.cols, hand_pos.y + 0.5*img.rows);	//center point
	Point pt;
	Point pivot(imgSize.width / 2, imgSize.height * 0.85);
	double a2, b2, angle_;
	double c2((ct.x - pivot.x)*(ct.x - pivot.x) + (ct.y - pivot.y)*(ct.y - pivot.y));

	double m00(0), m10(0), m01(0);

	for (int y = 0; y < img.rows; y++)
	{
		const uchar* Mi = img.ptr<uchar>(y);
		for (int x = 0; x < img.cols; x++)
		{
			if (Mi[x] != 0)		//calculate the moments
			{
				pt.x = hand_pos.x + x;
				pt.y = hand_pos.y + y;
				a2 = (pt.x - pivot.x)*(pt.x - pivot.x) + (pt.y - pivot.y)*(pt.y - pivot.y);
				b2 = (pt.x - ct.x)*(pt.x - ct.x) + (pt.y - ct.y)*(pt.y - ct.y);
				angle_ = (b2 + c2 - a2) / (2 * sqrt(b2) * sqrt(c2));
				
				if (angle_ >= 0)
				{
					m00 += Mi[x];
					m10 += Mi[x] * x;
					m01 += Mi[x] * y;
				}
				else
				{
					m00 += Mi[x] * 2.5;
					m10 += Mi[x] * x * 2.5;
					m01 += Mi[x] * y * 2.5;
				}
			}
		}
	}

	if (m00 != 0)
	{
		hand_pos.x += (m10 / m00) - (0.5 * hand_pos.width);		//x + dx
		hand_pos.y += (m01 / m00) - (0.5 * hand_pos.height);		//y + dy
		hand_pos.x = hand_pos.x<0 ? 0 :hand_pos.x;
		hand_pos.y = hand_pos.y<0 ? 0 : hand_pos.y;
		hand_pos.x = (hand_pos.x + hand_pos.width) > imgSize.width ? (imgSize.width - hand_pos.width) : hand_pos.x;
		hand_pos.y = (hand_pos.y + hand_pos.height) > imgSize.height ? (imgSize.height - hand_pos.height) : hand_pos.y;
	}
}

Tracking::Tracking(): m_classifier(NULL), m_count(0)
{
}

Tracking::~Tracking()
{
	if (m_classifier != NULL)
	{
		delete m_classifier;
	}
}

int Tracking::initialize(CLASSIFIER classifier_type, string xml_file, const int particle_num, double* aff_sig)
{
	if (m_classifier != NULL)
	{
		delete m_classifier;
	}
	switch (classifier_type)
	{
	case HOG_LIKE:
		m_classifier = new HOGLIKEClassifier();
		break;
	case LBP:
		m_classifier = new LBPClassifier();
		break;
	case  HOG:
		m_classifier = new HOGClassifier();
		break;
	default:
		return -1;
	}

	if (!m_classifier->load(xml_file))
	{
		cout << "ERROR: Load classifier failed!" << endl;
		return -1;
	}
	m_particle_num = particle_num;
	m_aff_sig = Mat(1, 6, CV_64FC1, aff_sig).clone();
	m_rng = RNG(time(NULL));
	m_est = Mat::zeros(1, 6, CV_64FC1);
	m_status = Fail;

	return 0;
}

void Tracking:: warping(Mat& src, Mat& dst, double* param)
{
	double cth, sth, cph, sph, ccc, ccs, css, scc, scs, sss;
	Mat map_x, map_y;

	int width = m_classifier->size;
	int height = m_classifier->size;

	map_x.create(height, width, CV_32FC1);
	map_y.create(height, width, CV_32FC1);

	float* x = (float*)map_x.data;
	float* y = (float*)map_y.data;

	cth = cos(param[4]);
	sth = sin(param[4]);
	cph = cos(param[5]);
	sph = sin(param[5]);

	ccc = cth*cph*cph;
	ccs = cth*cph*sph;
	css = cth*sph*sph;

	scc = sth*cph*cph;
	scs = sth*cph*sph;
	sss = sth*sph*sph;

	for (int i = 0; i < height; i++)
	{
		for(int j = 0; j < width; j++)
		{
			x[i*width+j] = param[0] + ((float)j/width-0.5)*(param[2]*(ccc+scs)+param[3]*(css-scs))
				+ ((float)i/width-(float)height/width/2.0)*(param[2]*(ccs-scc)-param[3]*(ccs+sss));
			y[i*width+j] = param[1] + ((float)j/width-0.5)*(param[2]*(scc-ccs)+param[3]*(ccs+sss))
				+((float)i/width-(float)height/width/2.0)*(param[2]*(ccc+scs)+param[3]*(css-scs));
		}
	}

	remap(src, dst, map_x, map_y, INTER_NEAREST);
}

void Tracking::calResult(double* est_data, const Point2d* p_points, Point* p_out_points)
{
	double cth, sth, cph, sph, ccc, ccs, css, scc, scs, sss;
	cth = cos(est_data[4]);
	sth = sin(est_data[4]);
	cph = cos(est_data[5]);
	sph = sin(est_data[5]);

	ccc = cth * cph * cph;
	ccs = cth * cph * sph;
	css = cth * sph * sph;

	scc = sth*cph*cph;
	scs = sth*cph*sph;
	sss = sth*sph*sph;

	double width = m_classifier->size;
	double height = m_classifier->size;

	for(int i = 0; i < 4; i++)
	{
		p_out_points[i].x = cvRound(est_data[0] + (p_points[i].x/width-0.5)*((ccc + scs)*est_data[2]+(css - scs)*est_data[3])
			+ (p_points[i].y/width-0.5)*(est_data[2]*(ccs-scc)+est_data[3]*(-ccs-sss)));

		p_out_points[i].y = cvRound(est_data[1] + (p_points[i].x/width-0.5)*(est_data[2]*(scc-ccs)+est_data[3]*(ccs+sss))
			+(p_points[i].y/width-0.5)*(est_data[2]*(ccc+scs)+est_data[3]*(-scs+css)));
	}
}

void Tracking::drawResult(Mat& image, int method)
{
	if (m_status != Fail)
	{
		const Point* pts[] = {m_curr_result.points};
		int points_num = 4;

		switch (method)
		{
		case 0:
			polylines(image, pts, &points_num, 1, true, CV_RGB(0, 255, 0), 3);
			break;
		case 1:
			int x, y;
			x = (m_curr_result.points[0].x + m_curr_result.points[1].x
				+ m_curr_result.points[2].x + m_curr_result.points[3].x)/4;
			y = (m_curr_result.points[0].y + m_curr_result.points[1].y
				+ m_curr_result.points[2].y + m_curr_result.points[3].y)/4;
			float radius;
			radius = sqrtf((m_curr_result.points[0].x - m_curr_result.points[1].x) *
				(m_curr_result.points[0].x - m_curr_result.points[1].x) +
				(m_curr_result.points[0].y - m_curr_result.points[1].y) *
				(m_curr_result.points[0].y - m_curr_result.points[1].y))/2;

			circle(image, Point(x, y), cvRound(radius), CV_RGB(0, 233, 34), 2);
			break;
		}
	}
}
void Tracking::process(Mat& image, const bool only_detect)
{
	Mat image_copy;
	image.copyTo(image_copy);

	vector<Rect> hands;
	if (only_detect || m_status == Fail || m_status == Tracking_Unstable)
	{
		m_classifier->detectHand(image_copy, hands);

		if (!hands.empty())
		{
			m_status = Detect_Sucess;
			Rect biggest_hand = hands[0];

			for (int i = 1; i < hands.size(); i++)
			{
				if (hands[i].area() > biggest_hand.area())
				{
					biggest_hand = hands[i];
				}
			}
			double* est_data = (double*)m_est.data;

			est_data[2] = (double)biggest_hand.width*1.2;
			est_data[3] = (double)biggest_hand.height*1.2;
			est_data[0] = (double)biggest_hand.x + est_data[2]/2.0;
			est_data[1] = (double)biggest_hand.y + est_data[3]/2.0;
			est_data[4] = 0;
			est_data[5] = 0;

			m_curr_result.points[0] = Point(biggest_hand.x, biggest_hand.y);
			m_curr_result.points[1] = Point(biggest_hand.x, biggest_hand.y + biggest_hand.height);
			m_curr_result.points[2] = Point(biggest_hand.x + biggest_hand.width, biggest_hand.y + biggest_hand.height);
			m_curr_result.points[3] = Point(biggest_hand.x + biggest_hand.width, biggest_hand.y);

			for (int i = 0; i < 4; i++)
			{
				m_pre_result.points[i] = m_curr_result.points[i];
			}

			return;
		}
	}
	else
	{
		//m_status = STATUS::Fail;
		Mat warping_image;
		Mat est;
		double* est_data;

		Mat random = Mat::zeros(1, 6, CV_64FC1);
		int hands_number = 0;
		Point result_points[4];
		m_curr_result.points[0] = Point(0, 0);
		m_curr_result.points[1] = Point(0, 0);
		m_curr_result.points[2] = Point(0, 0);
		m_curr_result.points[3] = Point(0, 0);

		for (int i = 0; i < m_particle_num; i++)
		{
			m_rng.fill(random, RNG::NORMAL, Scalar(0), Scalar(1));

			m_aff_sig.at<double>(0) = m_est.at<double>(2) * 0.2;
			m_aff_sig.at<double>(1) = m_est.at<double>(3) * 0.2;
			est = m_est + random.mul(m_aff_sig);
			est_data = (double*)est.data;

			est_data[0] = (est_data[0] < 0) ? 0 : est_data[0];
			est_data[0] = (est_data[0] > image_copy.cols -1) ? image_copy.cols - 1 : est_data[0];
			est_data[1] = (est_data[1] < 0) ? 0 : est_data[1];
			est_data[1] = (est_data[1] > image_copy.rows -1) ? image_copy.rows - 1 : est_data[1];

			warping(image_copy, warping_image, est_data);

			imshow("warp image", warping_image);

			m_classifier->detectHand(warping_image, hands);

			if (!hands.empty())
			{
				//m_status = STATUS::Tracking_Sucess;
				m_est = est.clone();
				Rect biggest_hand = hands[0];

				for (int i = 1; i < hands.size(); i++)
				{
					if (hands[i].area() > biggest_hand.area())
					{
						biggest_hand = hands[i];
					}
				}

				rectangle(warping_image, biggest_hand, CV_RGB(233, 0, 0), 1);
				imshow("warp image", warping_image);

				double x, y, width, height;
				x = (double)biggest_hand.x;
				y = (double)biggest_hand.y;
				width = (double)biggest_hand.width;
				height = (double)biggest_hand.height;
				hands_number++;

				Point2d p_points[4] = {
					Point2d(x, y),
					Point2d(x, y + height),
					Point2d(x + width, y + height),
					Point2d(x + width, y)
				};
				//calResult(est_data, p_points, m_result.points);
				calResult(est_data, p_points, result_points);

				m_curr_result.points[0] += result_points[0];
				m_curr_result.points[1] += result_points[1];
				m_curr_result.points[2] += result_points[2];
				m_curr_result.points[3] += result_points[3];
				//return;
			}
		}

		float curr_position[2] = {0.0f};
		if (hands_number != 0)
		{
			m_status = Tracking_Sucess;
			m_count = 0;
			//float curr_position[2] = {0.0f};
			for (int i = 0; i < 4; i++)
			{
				//cout << "hands number: " << hands_number << endl;
				m_curr_result.points[i].x /= hands_number;
				m_curr_result.points[i].y /= hands_number;
				m_pre_result.points[i] = m_curr_result.points[i];
				curr_position[0] += m_curr_result.points[i].x;
				curr_position[1] += m_curr_result.points[i].y;
			}
			curr_position[0] /= 4;
			curr_position[1] /= 4;
//			m_motion.Update(curr_position);
			m_predict.update(image_copy, m_curr_result);
			m_predict.adaThres(image_copy);
		}
		else
		{
			m_count++;
			if (m_count == 10)
			{
				m_status = Tracking_Unstable;
				m_count = 0;
			}
			else
			{
				m_status = Tracking_Sucess;
			}
			float pre_position[2] = {0.0f};
			pre_position[0] = (m_pre_result.points[0].x + m_pre_result.points[1].x + 
				m_pre_result.points[2].x + m_pre_result.points[3].x) / 4;
			pre_position[1] = (m_pre_result.points[0].y + m_pre_result.points[1].y + 
				m_pre_result.points[2].y + m_pre_result.points[3].y) / 4;
			curr_position[0] = pre_position[0];
			curr_position[1] = pre_position[1];
//			m_motion.Predict(curr_position);
//			m_motion.Update(curr_position);
			bool predict_flag = m_predict.predict(curr_position);
			int dx, dy;
			dx = cvRound(curr_position[0] - pre_position[0]);
			dy = cvRound(curr_position[1] - pre_position[1]);

//			cout << "curr: " << curr_position[0] << endl;
//			cout << "pre: " << pre_position[0] << endl;
//			cout << "dx: " << dx << endl;
//			cout << "dy: " << dy << endl;
			for (int i = 0; i < 4; i++)
			{
				m_curr_result.points[i].x = m_pre_result.points[i].x + dx;
				m_curr_result.points[i].y = m_pre_result.points[i].y + dy;
				m_pre_result.points[i] = m_curr_result.points[i];
			}
//			m_status = Fail;
//			m_motion.Reset();
			m_predict.update(image_copy, m_curr_result);
			if (!predict_flag)
			{
				m_status = Fail;
			}
		}
	}

	cout<<"m_status: "<<m_status<<endl;
	return;
}

void TSLskinSegment(const cv::Mat& src, cv::Mat& dst)
{
	double r, g, b, T, S, rgbSum;
	vector<Mat> planes;
	split(src,planes);

	MatIterator_<uchar> it_B = planes[0].begin<uchar>(),
		it_B_end = planes[0].end<uchar>(),
		it_G = planes[1].begin<uchar>(),
		it_R = planes[2].begin<uchar>(),
		it_bw = dst.begin<uchar>();

	for (;it_B != it_B_end; ++it_B,++it_G,++it_R,++it_bw)
	{
		b = *it_B;
		g = *it_G;
		r = *it_R;
		rgbSum = b+g+r;
		b = b/rgbSum-0.33;
		r = r/rgbSum -0.33;
		g = g/rgbSum-0.33;

		if (fabs(g)<EPSILON)
		{
			T = 0;
		}
		else if (g>EPSILON)
		{
			T = (atan(r/g)/(2*CV_PI)+0.25)*300;
		}
		else
		{
			T = ((atan(r/g))/(2*CV_PI)+0.75)*300;
		}

		S = (sqrt((r*r+g*g)*1.8))*100;

		*it_bw = 255*(T>125&&T<185&&((1.033*T-114.8425)>S)&&((380.1575-1.967*T)>S));
	}
}