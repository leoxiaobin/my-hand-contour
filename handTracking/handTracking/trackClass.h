#ifndef TRACK_CLASS_H
#define TRACK_CLASS_H
#include "cascadedetect.h"
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>

const double EPSILON = 0.0000001;
using namespace cv;

typedef struct ResultRect
{
	Point points[4];
	//Point left_top;
	//Point left_bottom;
	//Point right_top;
	//Point right_bottom;
}ResultRect;

enum STATUS{
	Detect_Sucess,
	Tracking_Sucess,
	Tracking_Unstable,
	Fail
};

enum CLASSIFIER{
	HOG,
	HOG_LIKE,
	LBP,
	HARR
};

class Classifier
{
public:
	Classifier(CLASSIFIER classifier_type);
	virtual ~Classifier();
	virtual bool load(string xml_file) = 0;
	virtual void detectHand(Mat& image, vector<Rect>& hands) = 0;

	int size;
	CLASSIFIER m_classifier_type;
};

class HOGLIKEClassifier: public Classifier
{
public:
	HOGLIKEClassifier();
	virtual ~HOGLIKEClassifier();
	virtual bool load(string xml_file);
	virtual void detectHand(Mat& image, vector<Rect>& hands);

private:
	ACascadeClassifier m_classifier;
};

class LBPClassifier: public Classifier
{
public:
	LBPClassifier();
	virtual ~LBPClassifier();
	virtual bool load(string xml_file);
	virtual void detectHand(Mat& image, vector<Rect>& hands);

private:
	CascadeClassifier m_classifier;
};

class HOGClassifier: public Classifier
{
public:
	HOGClassifier();
	virtual ~HOGClassifier();
	virtual bool load(string xml_file);
	virtual void detectHand(Mat& image, vector<Rect>& hands);

private:
	HOGDescriptor m_classifier;
};

class MotionHistory
{
public:
	MotionHistory(float alpha = 0.6f);
	~MotionHistory();

	void Update(float cur_para[2]);
	void Reset();
	void Predict(float cur_para[2]);
	void Initialize(float alpha = 0.6f);

private:
	float m_pre_para[2];
	float m_pre_vel[2];
	float m_pre_acc[2];
	float m_alpha;
	int m_frame_index;
	bool m_is_ok;
};

class MotionPredict
{
public:
	MotionPredict();
	~MotionPredict();
	void adaThres(Mat &frame);
	void update(Mat &frame, ResultRect cur_pos);
	bool predict(float cur_pos[2]);

private:
	void skinSegment(const Mat &src, Mat &dst);
	void calcWin(const Mat &img);
	float thres[3];
	int count;
	Size imgSize;
	Mat cur_frame;
	Mat pre_frame;
	Rect hand_pos;
};

class Tracking
{
public:
	Tracking();
	virtual ~Tracking();
	int initialize(CLASSIFIER classifier_type, string xmlFile, const int particle_num, double* aff_sig);
	void process(Mat& image, const bool only_detect = false);
	void drawResult(Mat& image, int method = 1);
	bool getResult(Point* p_points);
	inline STATUS getStatus() {return m_status;}

private:
	void warping(Mat& src, Mat& dst, double* param);
	void calResult(double* est_data, const Point2d* p_points, Point* p_out_points);
	ResultRect m_curr_result;
	ResultRect m_pre_result;
	int m_count;
	RNG m_rng;
	Mat m_aff_sig;
	Mat m_est;
	int m_particle_num;
	STATUS m_status;
	Classifier* m_classifier;
	MotionHistory m_motion;
	MotionPredict m_predict;
};

void TSLskinSegment(const cv::Mat& src, cv::Mat& dst);
#endif