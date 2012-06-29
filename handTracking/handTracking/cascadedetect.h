#ifndef CASCADEDETECT_H
#define CASCADEDETECT_H

#define HAVE_TBB

#include "cv.h"
#include "highgui.h"
#include "cxcore.h"
#include "cvaux.h"
#include <stdio.h>
#include <string.h>
#include <vector>
#include <iostream>
#include "tbb/tbb_stddef.h"
#include "tbb/tbb.h"

using namespace std;
using namespace cv;

class AFeatureEvaluator
{
public:
	enum { HAAR = 0, LBP = 1, EHOG = 2, HOG = 3, HOG_4 = 4, HOG_6 = 6, HOG_8 = 8 };
	virtual ~AFeatureEvaluator();
	virtual bool read(const FileNode& node);
	virtual Ptr<AFeatureEvaluator> clone() const;
	virtual int getFeatureType() const;

	virtual bool setImage(const Mat&, Size origWinSize);
	virtual bool setWindow(Point p);

	virtual double calcOrd(int featureIdx) const;
	virtual int calcCat(int featureIdx) const;

	static Ptr<AFeatureEvaluator> create(int type);
};

class ACascadeClassifier
{
public:
	struct DTreeNode
	{
		int featureIdx;
		float threshold; // for ordered features only
		int left;
		int right;
	};

	struct DTree
	{
		int nodeCount;
	};

	struct Stage
	{
		int first;
		int ntrees;
		float threshold;
	};

	enum { BOOST = 0 };
	enum { DO_CANNY_PRUNING = CV_HAAR_DO_CANNY_PRUNING,
		SCALE_IMAGE = CV_HAAR_SCALE_IMAGE,
		FIND_BIGGEST_OBJECT = CV_HAAR_FIND_BIGGEST_OBJECT,
		DO_ROUGH_SEARCH = CV_HAAR_DO_ROUGH_SEARCH };

	ACascadeClassifier();
	ACascadeClassifier(const string& filename);
	~ACascadeClassifier();

	bool empty() const;
	bool load(const string& filename);
	bool read(const FileNode& node);
	void detectMultiScale( const Mat& image,
		vector<Rect>& objects,
		double scaleFactor=1.1,
		int minNeighbors=3, int flags=0,
		Size minSize=Size());

	bool setImage( Ptr<AFeatureEvaluator>&, const Mat& );
	int runAt( Ptr<AFeatureEvaluator>&, Point );

	bool is_stump_based;

	int stageType;
	int featureType;
	int ncategories;
	Size origWinSize;

	vector<Stage> stages;
	vector<DTree> classifiers;
	vector<DTreeNode> nodes;
	vector<float> leaves;
	vector<int> subsets;

	Ptr<AFeatureEvaluator> feval;
	Ptr<CvHaarClassifierCascade> oldCascade;
};

#endif