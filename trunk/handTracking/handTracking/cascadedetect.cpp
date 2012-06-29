//#include "_cv.h"
#include "cascadedetect.h"
#include <cstdio>
#include <stdio.h>
#include <iostream>

#define CC_CASCADE_PARAMS "cascadeParams"
#define CC_STAGE_TYPE     "stageType"
#define CC_FEATURE_TYPE   "featureType"
#define CC_HEIGHT         "height"
#define CC_WIDTH          "width"

#define CC_STAGE_NUM    "stageNum"
#define CC_STAGES       "stages"
#define CC_STAGE_PARAMS "stageParams"

#define CC_BOOST            "BOOST"
#define CC_MAX_DEPTH        "maxDepth"
#define CC_WEAK_COUNT       "maxWeakCount"
#define CC_STAGE_THRESHOLD  "stageThreshold"
#define CC_WEAK_CLASSIFIERS "weakClassifiers"
#define CC_INTERNAL_NODES   "internalNodes"
#define CC_LEAF_VALUES      "leafValues"

#define CC_FEATURES       "features"
#define CC_FEATURE_PARAMS "featureParams"
#define CC_MAX_CAT_COUNT  "maxCatCount"

#define CC_HAAR   "HAAR"
#define CC_RECTS  "rects"
#define CC_TILTED "tilted"

#define CC_LBP  "LBP"
#define CC_RECT "rect"

#define CC_EHOG "EHOG"
#define CC_HOG  "HOG"
#define CC_HOG_4 "HOG_4"
#define CC_HOG_6 "HOG_6"
#define CC_HOG_8 "HOG_8"

#define CV_SUM_PTRS( p0, p1, p2, p3, sum, rect, step )                    \
    /* (x, y) */                                                          \
    (p0) = sum + (rect).x + (step) * (rect).y,                            \
    /* (x + w, y) */                                                      \
    (p1) = sum + (rect).x + (rect).width + (step) * (rect).y,             \
    /* (x + w, y) */                                                      \
    (p2) = sum + (rect).x + (step) * ((rect).y + (rect).height),          \
    /* (x + w, y + h) */                                                  \
    (p3) = sum + (rect).x + (rect).width + (step) * ((rect).y + (rect).height)

#define CV_TILTED_PTRS( p0, p1, p2, p3, tilted, rect, step )                        \
    /* (x, y) */                                                                    \
    (p0) = tilted + (rect).x + (step) * (rect).y,                                   \
    /* (x - h, y + h) */                                                            \
    (p1) = tilted + (rect).x - (rect).height + (step) * ((rect).y + (rect).height), \
    /* (x + w, y + w) */                                                            \
    (p2) = tilted + (rect).x + (rect).width + (step) * ((rect).y + (rect).width),   \
    /* (x + w - h, y + w + h) */                                                    \
    (p3) = tilted + (rect).x + (rect).width - (rect).height                         \
           + (step) * ((rect).y + (rect).width + (rect).height)

#define CALC_SUM_(p0, p1, p2, p3, offset) \
    ((p0)[offset] - (p1)[offset] - (p2)[offset] + (p3)[offset])

#define CALC_SUM(rect,offset) CALC_SUM_((rect)[0], (rect)[1], (rect)[2], (rect)[3], offset)

using namespace std;

AFeatureEvaluator::~AFeatureEvaluator() {}
bool AFeatureEvaluator::read(const FileNode&) {return true;}
Ptr<AFeatureEvaluator> AFeatureEvaluator::clone() const { return Ptr<AFeatureEvaluator>(); }
int AFeatureEvaluator::getFeatureType() const {return -1;}
bool AFeatureEvaluator::setImage(const Mat&, Size) {return true;}
bool AFeatureEvaluator::setWindow(Point) { return true; }
double AFeatureEvaluator::calcOrd(int) const { return 0.; }
int AFeatureEvaluator::calcCat(int) const { return 0; }

//----------------------------------------------  EHOGEvaluator -------------------------------------
class EHOGEvaluator : public AFeatureEvaluator
{
public:
	struct Feature
	{
		Feature();
		Feature( int x, int y, int _cell_w, int _cell_h  ) :
		rect(x, y, _cell_w, _cell_h) {}

		int calc( int offset ) const;
		void updatePtrs( const vector<Mat>& intBins);
		bool read(const FileNode& node );

		Rect rect; // weight and height for block
		const double* p[28]; // fast
	};

	EHOGEvaluator();
	virtual ~EHOGEvaluator();

	virtual bool read( const FileNode& node );
	virtual Ptr<AFeatureEvaluator> clone() const;
	virtual int getFeatureType() const { return AFeatureEvaluator::EHOG; }

	virtual bool setImage(const Mat& image, Size _origWinSize);
	virtual bool setWindow(Point pt);

	int operator()(int featureIdx) const
	{ return featuresPtr[featureIdx].calc(offset); }
	virtual int calcCat(int featureIdx) const
	{ return (*this)(featureIdx); }
private:
	Size origWinSize;
	Ptr<vector<Feature> > features;
	Feature* featuresPtr; // optimization
	vector<Mat> intBins;
	Rect normrect;

	static const int nbins = 7;
	static const bool gammaCorrection = true;
	int offset;
};

class HOG_4Evaluator : public AFeatureEvaluator
{
public:
	struct Feature
	{
		Feature();
		Feature( int x, int y, int _cell_w, int _cell_h  ) :
		rect(x, y, _cell_w, _cell_h) {}

		int calc( int offset ) const;
		void updatePtrs( const vector<Mat>& intBins);
		bool read(const FileNode& node );

		Rect rect; // weight and height for block
		const double* p[20]; // fast
	};

	HOG_4Evaluator();
	virtual ~HOG_4Evaluator();

	virtual bool read( const FileNode& node );
	virtual Ptr<AFeatureEvaluator> clone() const;
	virtual int getFeatureType() const { return AFeatureEvaluator::HOG_4; }

	virtual bool setImage(const Mat& image, Size _origWinSize);
	virtual bool setWindow(Point pt);

	int operator()(int featureIdx) const
	{ return featuresPtr[featureIdx].calc(offset); }
	virtual int calcCat(int featureIdx) const
	{ return (*this)(featureIdx); }
private:
	Size origWinSize;
	Ptr<vector<Feature> > features;
	Feature* featuresPtr; // optimization
	vector<Mat> intBins;
	Rect normrect;

	static const int nbins = 5;
	static const bool gammaCorrection = true;
	int offset;
};

class HOG_6Evaluator : public AFeatureEvaluator
{
public:
	struct Feature
	{
		Feature();
		Feature( int x, int y, int _cell_w, int _cell_h  ) :
		rect(x, y, _cell_w, _cell_h) {}

		int calc( int offset ) const;
		void updatePtrs( const vector<Mat>& intBins);
		bool read(const FileNode& node );

		Rect rect; // weight and height for block
		const double* p[28]; // fast
	};

	HOG_6Evaluator();
	virtual ~HOG_6Evaluator();

	virtual bool read( const FileNode& node );
	virtual Ptr<AFeatureEvaluator> clone() const;
	virtual int getFeatureType() const { return AFeatureEvaluator::HOG_6; }

	virtual bool setImage(const Mat& image, Size _origWinSize);
	virtual bool setWindow(Point pt);

	int operator()(int featureIdx) const
	{ return featuresPtr[featureIdx].calc(offset); }
	virtual int calcCat(int featureIdx) const
	{ return (*this)(featureIdx); }
private:
	Size origWinSize;
	Ptr<vector<Feature> > features;
	Feature* featuresPtr; // optimization
	vector<Mat> intBins;
	Rect normrect;

	static const int nbins = 7;
	static const bool gammaCorrection = true;
	int offset;
};

class HOG_8Evaluator : public AFeatureEvaluator
{
public:
	struct Feature
	{
		Feature();
		Feature( int x, int y, int _cell_w, int _cell_h  ) :
		rect(x, y, _cell_w, _cell_h) {}

		int calc( int offset ) const;
		void updatePtrs( const vector<Mat>& intBins);
		bool read(const FileNode& node );

		Rect rect; // weight and height for block
		const double* p[36]; // fast
	};

	HOG_8Evaluator();
	virtual ~HOG_8Evaluator();

	virtual bool read( const FileNode& node );
	virtual Ptr<AFeatureEvaluator> clone() const;
	virtual int getFeatureType() const { return AFeatureEvaluator::HOG_8; }

	virtual bool setImage(const Mat& image, Size _origWinSize);
	virtual bool setWindow(Point pt);

	int operator()(int featureIdx) const
	{ return featuresPtr[featureIdx].calc(offset); }
	virtual int calcCat(int featureIdx) const
	{ return (*this)(featureIdx); }
private:
	Size origWinSize;
	Ptr<vector<Feature> > features;
	Feature* featuresPtr; // optimization
	vector<Mat> intBins;
	Rect normrect;

	static const int nbins = 9;
	static const bool gammaCorrection = true;
	int offset;
};

inline EHOGEvaluator::Feature :: Feature()
{
	rect = Rect();
	for( int i = 0; i < 28; i++ )
		p[i] = 0;
}

inline HOG_4Evaluator::Feature :: Feature()
{
	rect = Rect();
	for( int i = 0; i < 20; i++ )
		p[i] = 0;
}

inline HOG_6Evaluator::Feature :: Feature()
{
	rect = Rect();
	for( int i = 0; i < 28; i++ )
		p[i] = 0;
}

inline HOG_8Evaluator::Feature :: Feature()
{
	rect = Rect();
	for( int i = 0; i < 36; i++ )
		p[i] = 0;
}

inline int EHOGEvaluator::Feature :: calc( int offset ) const
{
	double cval = CALC_SUM_( p[12], p[13], p[14], p[15], offset );

	//return (int)((CALC_SUM_( p[0],  p[1],  p[2],  p[3],  offset ) >= cval ? 128 : 0) |
	//	   (CALC_SUM_( p[4],  p[5],  p[6],  p[7],  offset ) >= cval ? 64 : 0) |
	//	   (CALC_SUM_( p[8],  p[9],  p[10], p[11], offset ) >= cval ? 32 : 0) |
	//	   (CALC_SUM_( p[12], p[13], p[14], p[15], offset ) >= cval ? 16 : 0) |
	//	   (CALC_SUM_( p[20], p[21], p[22], p[23], offset ) >= cval ? 8 : 0)|
	//	   (CALC_SUM_( p[24], p[25], p[26], p[27], offset ) >= cval ? 4 : 0)|
	//	   (CALC_SUM_( p[28], p[29], p[30], p[31], offset ) >= cval ? 2 : 0)|
	//	   (CALC_SUM_( p[32], p[33], p[34], p[35], offset ) >= cval ? 1 : 0));
	//return (int)((CALC_SUM_( p[0],  p[1],  p[2],  p[3],  offset ) >= cval ? 8 : 0) |
	//	(CALC_SUM_( p[4],  p[5],  p[6],  p[7],  offset ) >= cval ? 4 : 0) |
	//	(CALC_SUM_( p[12], p[13], p[14], p[15], offset ) >= cval ? 2 : 0) |
	//	(CALC_SUM_( p[16], p[17], p[18], p[19], offset ) >= cval ? 1 : 0));
	return (int)((CALC_SUM_( p[0],  p[1],  p[2],  p[3],  offset ) >= cval ? 32 : 0) |
		(CALC_SUM_( p[4],  p[5],  p[6],  p[7],  offset ) >= cval ? 16 : 0) |
		(CALC_SUM_( p[8],  p[9],  p[10], p[11], offset ) >= cval ? 8 : 0) |
		(CALC_SUM_( p[16], p[17], p[18], p[19], offset ) >= cval ? 4 : 0)|
		(CALC_SUM_( p[20], p[21], p[22], p[23], offset ) >= cval ? 2 : 0)|
		(CALC_SUM_( p[24], p[25], p[26], p[27], offset ) >= cval ? 1 : 0));
}

inline int HOG_4Evaluator::Feature :: calc( int offset ) const
{
	double cval = CALC_SUM_( p[8], p[9], p[10], p[11], offset );

	return (int)((CALC_SUM_( p[0],  p[1],  p[2],  p[3],  offset ) >= cval ? 8 : 0) |
		(CALC_SUM_( p[4],  p[5],  p[6],  p[7],  offset ) >= cval ? 4 : 0) |
		(CALC_SUM_( p[12], p[13], p[14], p[15], offset ) >= cval ? 2 : 0) |
		(CALC_SUM_( p[16], p[17], p[18], p[19], offset ) >= cval ? 1 : 0));
}

inline int HOG_6Evaluator::Feature :: calc( int offset ) const
{
	double cval = CALC_SUM_( p[12], p[13], p[14], p[15], offset );

	return (int)((CALC_SUM_( p[0],  p[1],  p[2],  p[3],  offset ) >= cval ? 32 : 0) |
		(CALC_SUM_( p[4],  p[5],  p[6],  p[7],  offset ) >= cval ? 16 : 0) |
		(CALC_SUM_( p[8],  p[9],  p[10], p[11], offset ) >= cval ? 8 : 0) |
		(CALC_SUM_( p[16], p[17], p[18], p[19], offset ) >= cval ? 4 : 0)|
		(CALC_SUM_( p[20], p[21], p[22], p[23], offset ) >= cval ? 2 : 0)|
		(CALC_SUM_( p[24], p[25], p[26], p[27], offset ) >= cval ? 1 : 0));
}

inline int HOG_8Evaluator::Feature :: calc( int offset ) const
{
	double cval = CALC_SUM_( p[16], p[17], p[18], p[19], offset );

	return (int)((CALC_SUM_( p[0],  p[1],  p[2],  p[3],  offset ) >= cval ? 128 : 0) |
		   (CALC_SUM_( p[4],  p[5],  p[6],  p[7],  offset ) >= cval ? 64 : 0) |
		   (CALC_SUM_( p[8],  p[9],  p[10], p[11], offset ) >= cval ? 32 : 0) |
		   (CALC_SUM_( p[12], p[13], p[14], p[15], offset ) >= cval ? 16 : 0) |
		   (CALC_SUM_( p[20], p[21], p[22], p[23], offset ) >= cval ? 8 : 0)|
		   (CALC_SUM_( p[24], p[25], p[26], p[27], offset ) >= cval ? 4 : 0)|
		   (CALC_SUM_( p[28], p[29], p[30], p[31], offset ) >= cval ? 2 : 0)|
		   (CALC_SUM_( p[32], p[33], p[34], p[35], offset ) >= cval ? 1 : 0));
}

inline void EHOGEvaluator::Feature :: updatePtrs( const vector<Mat>& intBins )
{
	//const double* ptr0 = (const double*)intBins[0].data;
	//const double* ptr1 = (const double*)intBins[1].data;
	//const double* ptr2 = (const double*)intBins[2].data;
	//const double* ptr3 = (const double*)intBins[3].data;
	//const double* ptr4 = (const double*)intBins[4].data;
	//const double* ptr5 = (const double*)intBins[5].data;
	//const double* ptr6 = (const double*)intBins[6].data;
	//const double* ptr7 = (const double*)intBins[7].data;
	//const double* ptr8 = (const double*)intBins[8].data;
	const double* ptr0 = (const double*)intBins[0].data;
	const double* ptr1 = (const double*)intBins[1].data;
	const double* ptr2 = (const double*)intBins[2].data;
	const double* ptr3 = (const double*)intBins[3].data;
	const double* ptr4 = (const double*)intBins[4].data;
	const double* ptr5 = (const double*)intBins[5].data;
	const double* ptr6 = (const double*)intBins[6].data;

	size_t step = intBins[0].step/sizeof(ptr0[0]);
	Rect tr = rect;

	//CV_SUM_PTRS( p[0],  p[1],  p[2],  p[3],  ptr0, tr, step );
	//CV_SUM_PTRS( p[4],  p[5],  p[6],  p[7],  ptr1, tr, step );
	//CV_SUM_PTRS( p[8],  p[9],  p[10], p[11], ptr2, tr, step );
	//CV_SUM_PTRS( p[12], p[13], p[14], p[15], ptr3, tr, step );
	//CV_SUM_PTRS( p[16], p[17], p[18], p[19], ptr4, tr, step );
	//CV_SUM_PTRS( p[20], p[21], p[22], p[23], ptr5, tr, step );
	//CV_SUM_PTRS( p[24], p[25], p[26], p[27], ptr6, tr, step );
	//CV_SUM_PTRS( p[28], p[29], p[30], p[31], ptr7, tr, step );
	//CV_SUM_PTRS( p[32], p[33], p[34], p[35], ptr8, tr, step );
	CV_SUM_PTRS( p[0],  p[1],  p[2],  p[3],  ptr0, tr, step );
	CV_SUM_PTRS( p[4],  p[5],  p[6],  p[7],  ptr1, tr, step );
	CV_SUM_PTRS( p[8],  p[9],  p[10], p[11], ptr2, tr, step );
	CV_SUM_PTRS( p[12], p[13], p[14], p[15], ptr3, tr, step );
	CV_SUM_PTRS( p[16], p[17], p[18], p[19], ptr4, tr, step );
	CV_SUM_PTRS( p[20], p[21], p[22], p[23], ptr5, tr, step );
	CV_SUM_PTRS( p[24], p[25], p[26], p[27], ptr6, tr, step );
}

inline void HOG_4Evaluator::Feature :: updatePtrs( const vector<Mat>& intBins )
{
	const double* ptr0 = (const double*)intBins[0].data;
	const double* ptr1 = (const double*)intBins[1].data;
	const double* ptr2 = (const double*)intBins[2].data;
	const double* ptr3 = (const double*)intBins[3].data;
	const double* ptr4 = (const double*)intBins[4].data;

	size_t step = intBins[0].step/sizeof(ptr0[0]);
	Rect tr = rect;

	CV_SUM_PTRS( p[0],  p[1],  p[2],  p[3],  ptr0, tr, step );
	CV_SUM_PTRS( p[4],  p[5],  p[6],  p[7],  ptr1, tr, step );
	CV_SUM_PTRS( p[8],  p[9],  p[10], p[11], ptr2, tr, step );
	CV_SUM_PTRS( p[12], p[13], p[14], p[15], ptr3, tr, step );
	CV_SUM_PTRS( p[16], p[17], p[18], p[19], ptr4, tr, step );
}

inline void HOG_6Evaluator::Feature :: updatePtrs( const vector<Mat>& intBins )
{
	const double* ptr0 = (const double*)intBins[0].data;
	const double* ptr1 = (const double*)intBins[1].data;
	const double* ptr2 = (const double*)intBins[2].data;
	const double* ptr3 = (const double*)intBins[3].data;
	const double* ptr4 = (const double*)intBins[4].data;
	const double* ptr5 = (const double*)intBins[5].data;
	const double* ptr6 = (const double*)intBins[6].data;

	size_t step = intBins[0].step/sizeof(ptr0[0]);
	Rect tr = rect;

	CV_SUM_PTRS( p[0],  p[1],  p[2],  p[3],  ptr0, tr, step );
	CV_SUM_PTRS( p[4],  p[5],  p[6],  p[7],  ptr1, tr, step );
	CV_SUM_PTRS( p[8],  p[9],  p[10], p[11], ptr2, tr, step );
	CV_SUM_PTRS( p[12], p[13], p[14], p[15], ptr3, tr, step );
	CV_SUM_PTRS( p[16], p[17], p[18], p[19], ptr4, tr, step );
	CV_SUM_PTRS( p[20], p[21], p[22], p[23], ptr5, tr, step );
	CV_SUM_PTRS( p[24], p[25], p[26], p[27], ptr6, tr, step );
}

inline void HOG_8Evaluator::Feature :: updatePtrs( const vector<Mat>& intBins )
{
	const double* ptr0 = (const double*)intBins[0].data;
	const double* ptr1 = (const double*)intBins[1].data;
	const double* ptr2 = (const double*)intBins[2].data;
	const double* ptr3 = (const double*)intBins[3].data;
	const double* ptr4 = (const double*)intBins[4].data;
	const double* ptr5 = (const double*)intBins[5].data;
	const double* ptr6 = (const double*)intBins[6].data;
	const double* ptr7 = (const double*)intBins[7].data;
	const double* ptr8 = (const double*)intBins[8].data;

	size_t step = intBins[0].step/sizeof(ptr0[0]);
	Rect tr = rect;

	CV_SUM_PTRS( p[0],  p[1],  p[2],  p[3],  ptr0, tr, step );
	CV_SUM_PTRS( p[4],  p[5],  p[6],  p[7],  ptr1, tr, step );
	CV_SUM_PTRS( p[8],  p[9],  p[10], p[11], ptr2, tr, step );
	CV_SUM_PTRS( p[12], p[13], p[14], p[15], ptr3, tr, step );
	CV_SUM_PTRS( p[16], p[17], p[18], p[19], ptr4, tr, step );
	CV_SUM_PTRS( p[20], p[21], p[22], p[23], ptr5, tr, step );
	CV_SUM_PTRS( p[24], p[25], p[26], p[27], ptr6, tr, step );
	CV_SUM_PTRS( p[28], p[29], p[30], p[31], ptr7, tr, step );
	CV_SUM_PTRS( p[32], p[33], p[34], p[35], ptr8, tr, step );
}

bool EHOGEvaluator::Feature :: read(const FileNode& node )
{
	FileNode rnode = node[CC_RECT];
	FileNodeIterator it = rnode.begin();
	it >> rect.x >> rect.y >> rect.width >> rect.height;
	return true;
}

bool HOG_4Evaluator::Feature :: read(const FileNode& node )
{
	FileNode rnode = node[CC_RECT];
	FileNodeIterator it = rnode.begin();
	it >> rect.x >> rect.y >> rect.width >> rect.height;
	return true;
}

bool HOG_6Evaluator::Feature :: read(const FileNode& node )
{
	FileNode rnode = node[CC_RECT];
	FileNodeIterator it = rnode.begin();
	it >> rect.x >> rect.y >> rect.width >> rect.height;
	return true;
}

bool HOG_8Evaluator::Feature :: read(const FileNode& node )
{
	FileNode rnode = node[CC_RECT];
	FileNodeIterator it = rnode.begin();
	it >> rect.x >> rect.y >> rect.width >> rect.height;
	return true;
}

EHOGEvaluator::EHOGEvaluator()
{
	features = new vector<Feature>();
}

HOG_4Evaluator::HOG_4Evaluator()
{
	features = new vector<Feature>();
}

HOG_6Evaluator::HOG_6Evaluator()
{
	features = new vector<Feature>();
}

HOG_8Evaluator::HOG_8Evaluator()
{
	features = new vector<Feature>();
}

EHOGEvaluator::~EHOGEvaluator()
{
}

HOG_4Evaluator::~HOG_4Evaluator()
{
}

HOG_6Evaluator::~HOG_6Evaluator()
{
}

HOG_8Evaluator::~HOG_8Evaluator()
{
}
bool EHOGEvaluator::read( const FileNode& node )
{
	features->resize(node.size());
	featuresPtr = &(*features)[0];
	FileNodeIterator it = node.begin(), it_end = node.end();
	for(int i = 0; it != it_end; ++it, i++)
	{
		if(!featuresPtr[i].read(*it))
			return false;
	}
	return true;
}

bool HOG_4Evaluator::read( const FileNode& node )
{
	features->resize(node.size());
	featuresPtr = &(*features)[0];
	FileNodeIterator it = node.begin(), it_end = node.end();
	for(int i = 0; it != it_end; ++it, i++)
	{
		if(!featuresPtr[i].read(*it))
			return false;
	}
	return true;
}

bool HOG_6Evaluator::read( const FileNode& node )
{
	features->resize(node.size());
	featuresPtr = &(*features)[0];
	FileNodeIterator it = node.begin(), it_end = node.end();
	for(int i = 0; it != it_end; ++it, i++)
	{
		if(!featuresPtr[i].read(*it))
			return false;
	}
	return true;
}

bool HOG_8Evaluator::read( const FileNode& node )
{
	features->resize(node.size());
	featuresPtr = &(*features)[0];
	FileNodeIterator it = node.begin(), it_end = node.end();
	for(int i = 0; it != it_end; ++it, i++)
	{
		if(!featuresPtr[i].read(*it))
			return false;
	}
	return true;
}

Ptr<AFeatureEvaluator> EHOGEvaluator::clone() const
{
	EHOGEvaluator* ret = new EHOGEvaluator;
	ret->origWinSize = origWinSize;
	ret->features = features;
	ret->featuresPtr = &(*ret->features)[0];
	ret->normrect = normrect;
	ret->intBins = intBins;
	ret->offset = offset;
	return ret;
}

Ptr<AFeatureEvaluator> HOG_4Evaluator::clone() const
{
	HOG_4Evaluator* ret = new HOG_4Evaluator;
	ret->origWinSize = origWinSize;
	ret->features = features;
	ret->featuresPtr = &(*ret->features)[0];
	ret->normrect = normrect;
	ret->intBins = intBins;
	ret->offset = offset;
	return ret;
}

Ptr<AFeatureEvaluator> HOG_6Evaluator::clone() const
{
	HOG_6Evaluator* ret = new HOG_6Evaluator;
	ret->origWinSize = origWinSize;
	ret->features = features;
	ret->featuresPtr = &(*ret->features)[0];
	ret->normrect = normrect;
	ret->intBins = intBins;
	ret->offset = offset;
	return ret;
}

Ptr<AFeatureEvaluator> HOG_8Evaluator::clone() const
{
	HOG_8Evaluator* ret = new HOG_8Evaluator;
	ret->origWinSize = origWinSize;
	ret->features = features;
	ret->featuresPtr = &(*ret->features)[0];
	ret->normrect = normrect;
	ret->intBins = intBins;
	ret->offset = offset;
	return ret;
}

class ParallelIntegral
{
public:
	vector<cv::Mat> imgs;
	vector<cv::Mat> sums;
	ParallelIntegral(vector<cv::Mat>& _imgs, vector<cv::Mat>& _sums ) : imgs(_imgs),sums(_sums)
	{}
	void operator()(const tbb::blocked_range<int>& range) const
	{
		//int  = range.begin();
		//int j = range.end();
		vector<Mat> imgs_ = imgs;
		vector<Mat> sums_ = sums;
		for (int i = range.begin(); i<range.end(); i++)
		{
			integral((imgs_)[i],(sums_)[i]);
		}
	}
};

bool EHOGEvaluator::setImage( const Mat& image, Size _origWinSize )
{
	int rn = image.rows+1, cn = image.cols+1;
	Size imgSize;
	imgSize.width = image.cols, imgSize.height = image.rows;
	origWinSize = _origWinSize;

	if( image.cols < origWinSize.width || image.rows < origWinSize.height )
		return false;

	//---------------------------compute gradient----------------------------//
	int i, x, y;

	Mat image2;
	equalizeHist(image, image2);

	vector<Mat> bd;
	intBins.clear();

	for( i = 0; i < nbins; i++)
	{
		Mat buf1 = Mat::zeros(imgSize.height, imgSize.width, CV_32FC1);
		bd.push_back(buf1);
		Mat buf2 = Mat::zeros(rn, cn, CV_64FC1);
		intBins.push_back(buf2);
	}

	Size wholeSize;
	Point roiofs;
	image2.locateROI(wholeSize, roiofs);

	Mat_<float> _lut(1, 256);
	const float* lut = &_lut(0, 0);

	if(gammaCorrection)
		for(i = 0; i < 256; i++)
			_lut(0 ,i) = std :: sqrt((float)i);
	else
		for(i = 0; i < 256; i++)
			_lut(0, i) = (float)i;

	//get the image data's address after interpolated
	AutoBuffer<int> mapbuf(imgSize.width + imgSize.height + 4);
	int* xmap = (int*)mapbuf + 1;
	int* ymap = xmap + imgSize.width + 2;

	const int borderType = (int)BORDER_REFLECT_101;

	for( x = -1; x < imgSize.width + 1; x++ )
		xmap[x] = borderInterpolate(x + roiofs.x,
		wholeSize.width, borderType);
	for( y = -1; y < imgSize.height + 1; y++ )
		ymap[y] = borderInterpolate(y + roiofs.y,
		wholeSize.height, borderType);

	// x- & y- derivatives for the whole row
	int width = imgSize.width;
	AutoBuffer<float> _dbuf(width*4);
	float* dbuf = _dbuf;
	Mat Dx(1, width, CV_32F, dbuf);
	Mat Dy(1, width, CV_32F, dbuf + width);
	Mat Mag(1, width, CV_32F, dbuf + width*2);
	Mat Angle(1, width, CV_32F, dbuf + width*3);

	//int _nbins = nbins;
	float angleScale = (float)(nbins/CV_PI);//angle per bin

	//double t = (double)cvGetTickCount();
	for( y = 0; y < imgSize.height; y++ )
	{
		const uchar* imgPtr = image2.data + image2.step*ymap[y];
		const uchar* prevPtr = image2.data + image2.step*ymap[y-1];
		const uchar* nextPtr = image2.data + image2.step*ymap[y+1];

		//only one channel, compute them
		for( x = 0; x < width; x++ )
		{
			int x1 = xmap[x];
			dbuf[x] = (float)(lut[imgPtr[xmap[x+1]]] - lut[imgPtr[xmap[x-1]]]);
			dbuf[width + x] = (float)(lut[nextPtr[x1]] - lut[prevPtr[x1]]);
		}

		//compute the mag and angle using the dx and dy
		cartToPolar( Dx, Dy, Mag, Angle, false );

		for( x = 0; x < width; x++ )
		{
			//for the sake of voting to the bins
			float mag = dbuf[x+width*2], angle = dbuf[x+width*3]*angleScale - 0.5f;
			int hidx = cvFloor(angle);
			angle -= hidx;
			//in order to normalize the hidx
			if( hidx < 0 )
				hidx += nbins;
			else if( hidx >= nbins )
				hidx -= nbins;
			assert( (unsigned)hidx < (unsigned)nbins );

			float* tmp = (float*)bd[hidx].ptr(y);
			tmp[x] = mag*(1.f - angle);//vote to the lower bin
			hidx++;
			hidx &= hidx < nbins ? -1 : 0;//if hidx == 8, hidx+1 = 0
			tmp = (float*)bd[hidx].ptr(y);
			tmp[x] = mag*angle;//vote to the higher bin
		}
	}
	//t = (double)cvGetTickCount() - t;
	//cout << "Integral Compute time = " << t*1000./cv::getTickFrequency() << "ms" << endl;

	integral(bd[0], intBins[0]);
	integral(bd[1], intBins[1]);
	integral(bd[2], intBins[2]);
	integral(bd[3], intBins[3]);
	integral(bd[4], intBins[4]);
	integral(bd[5], intBins[5]);
	integral(bd[6], intBins[6]);
	//integral(bd[7], intBins[7]);
	//integral(bd[8], intBins[8]);

	//tbb::parallel_for(tbb::blocked_range<int>(0,8),ParallelIntegral(bd,intBins));

	size_t fi, nfeatures = features->size();
	//cout << nfeatures << endl;

	for( fi = 0; fi < nfeatures; fi++ )
	{
		//cout << fi << endl;
		featuresPtr[fi].updatePtrs( intBins );
	}

	//bd[0].release();
	//bd[1].release();
	//bd[2].release();
	//bd[3].release();
	//bd[4].release();
	//bd[5].release();
	//bd[6].release();
	//bd[7].release();
	//bd[8].release();
	//bd.clear();
	return true;
}

bool HOG_4Evaluator::setImage( const Mat& image, Size _origWinSize )
{
	int rn = image.rows+1, cn = image.cols+1;
	Size imgSize;
	imgSize.width = image.cols, imgSize.height = image.rows;
	origWinSize = _origWinSize;

	if( image.cols < origWinSize.width || image.rows < origWinSize.height )
		return false;

	//---------------------------compute gradient----------------------------//
	int i, x, y;

	Mat image2;
	equalizeHist(image, image2);

	vector<Mat> bd;
	intBins.clear();

	for( i = 0; i < nbins; i++)
	{
		Mat buf1 = Mat::zeros(imgSize.height, imgSize.width, CV_32FC1);
		bd.push_back(buf1);
		Mat buf2 = Mat::zeros(rn, cn, CV_64FC1);
		intBins.push_back(buf2);
	}

	Size wholeSize;
	Point roiofs;
	image2.locateROI(wholeSize, roiofs);

	Mat_<float> _lut(1, 256);
	const float* lut = &_lut(0, 0);

	if(gammaCorrection)
		for(i = 0; i < 256; i++)
			_lut(0 ,i) = std :: sqrt((float)i);
	else
		for(i = 0; i < 256; i++)
			_lut(0, i) = (float)i;

	//get the image data's address after interpolated
	AutoBuffer<int> mapbuf(imgSize.width + imgSize.height + 4);
	int* xmap = (int*)mapbuf + 1;
	int* ymap = xmap + imgSize.width + 2;

	const int borderType = (int)BORDER_REFLECT_101;

	for( x = -1; x < imgSize.width + 1; x++ )
		xmap[x] = borderInterpolate(x + roiofs.x,
		wholeSize.width, borderType);
	for( y = -1; y < imgSize.height + 1; y++ )
		ymap[y] = borderInterpolate(y + roiofs.y,
		wholeSize.height, borderType);

	// x- & y- derivatives for the whole row
	int width = imgSize.width;
	AutoBuffer<float> _dbuf(width*4);
	float* dbuf = _dbuf;
	Mat Dx(1, width, CV_32F, dbuf);
	Mat Dy(1, width, CV_32F, dbuf + width);
	Mat Mag(1, width, CV_32F, dbuf + width*2);
	Mat Angle(1, width, CV_32F, dbuf + width*3);

	//int _nbins = nbins;
	float angleScale = (float)(nbins/CV_PI);//angle per bin

	//double t = (double)cvGetTickCount();
	for( y = 0; y < imgSize.height; y++ )
	{
		const uchar* imgPtr = image2.data + image2.step*ymap[y];
		const uchar* prevPtr = image2.data + image2.step*ymap[y-1];
		const uchar* nextPtr = image2.data + image2.step*ymap[y+1];

		//only one channel, compute them
		for( x = 0; x < width; x++ )
		{
			int x1 = xmap[x];
			dbuf[x] = (float)(lut[imgPtr[xmap[x+1]]] - lut[imgPtr[xmap[x-1]]]);
			dbuf[width + x] = (float)(lut[nextPtr[x1]] - lut[prevPtr[x1]]);
		}

		//compute the mag and angle using the dx and dy
		cartToPolar( Dx, Dy, Mag, Angle, false );

		for( x = 0; x < width; x++ )
		{
			//for the sake of voting to the bins
			float mag = dbuf[x+width*2], angle = dbuf[x+width*3]*angleScale - 0.5f;
			int hidx = cvFloor(angle);
			angle -= hidx;
			//in order to normalize the hidx
			if( hidx < 0 )
				hidx += nbins;
			else if( hidx >= nbins )
				hidx -= nbins;
			assert( (unsigned)hidx < (unsigned)nbins );

			float* tmp = (float*)bd[hidx].ptr(y);
			tmp[x] = mag*(1.f - angle);//vote to the lower bin
			hidx++;
			hidx &= hidx < nbins ? -1 : 0;//if hidx == 8, hidx+1 = 0
			tmp = (float*)bd[hidx].ptr(y);
			tmp[x] = mag*angle;//vote to the higher bin
		}
	}
	//t = (double)cvGetTickCount() - t;
	//cout << "Integral Compute time = " << t*1000./cv::getTickFrequency() << "ms" << endl;

	integral(bd[0], intBins[0]);
	integral(bd[1], intBins[1]);
	integral(bd[2], intBins[2]);
	integral(bd[3], intBins[3]);
	integral(bd[4], intBins[4]);

	//tbb::parallel_for(tbb::blocked_range<int>(0,8),ParallelIntegral(bd,intBins));

	size_t fi, nfeatures = features->size();
	//cout << nfeatures << endl;

	for( fi = 0; fi < nfeatures; fi++ )
	{
		//cout << fi << endl;
		featuresPtr[fi].updatePtrs( intBins );
	}

	return true;
}

bool HOG_6Evaluator::setImage( const Mat& image, Size _origWinSize )
{
	int rn = image.rows+1, cn = image.cols+1;
	Size imgSize;
	imgSize.width = image.cols, imgSize.height = image.rows;
	origWinSize = _origWinSize;

	if( image.cols < origWinSize.width || image.rows < origWinSize.height )
		return false;

	//---------------------------compute gradient----------------------------//
	int i, x, y;

	Mat image2;
	equalizeHist(image, image2);

	vector<Mat> bd;
	intBins.clear();

	for( i = 0; i < nbins; i++)
	{
		Mat buf1 = Mat::zeros(imgSize.height, imgSize.width, CV_32FC1);
		bd.push_back(buf1);
		Mat buf2 = Mat::zeros(rn, cn, CV_64FC1);
		intBins.push_back(buf2);
	}

	Size wholeSize;
	Point roiofs;
	image2.locateROI(wholeSize, roiofs);

	Mat_<float> _lut(1, 256);
	const float* lut = &_lut(0, 0);

	if(gammaCorrection)
		for(i = 0; i < 256; i++)
			_lut(0 ,i) = std :: sqrt((float)i);
	else
		for(i = 0; i < 256; i++)
			_lut(0, i) = (float)i;

	//get the image data's address after interpolated
	AutoBuffer<int> mapbuf(imgSize.width + imgSize.height + 4);
	int* xmap = (int*)mapbuf + 1;
	int* ymap = xmap + imgSize.width + 2;

	const int borderType = (int)BORDER_REFLECT_101;

	for( x = -1; x < imgSize.width + 1; x++ )
		xmap[x] = borderInterpolate(x + roiofs.x,
		wholeSize.width, borderType);
	for( y = -1; y < imgSize.height + 1; y++ )
		ymap[y] = borderInterpolate(y + roiofs.y,
		wholeSize.height, borderType);

	// x- & y- derivatives for the whole row
	int width = imgSize.width;
	AutoBuffer<float> _dbuf(width*4);
	float* dbuf = _dbuf;
	Mat Dx(1, width, CV_32F, dbuf);
	Mat Dy(1, width, CV_32F, dbuf + width);
	Mat Mag(1, width, CV_32F, dbuf + width*2);
	Mat Angle(1, width, CV_32F, dbuf + width*3);

	//int _nbins = nbins;
	float angleScale = (float)(nbins/CV_PI);//angle per bin

	//double t = (double)cvGetTickCount();
	for( y = 0; y < imgSize.height; y++ )
	{
		const uchar* imgPtr = image2.data + image2.step*ymap[y];
		const uchar* prevPtr = image2.data + image2.step*ymap[y-1];
		const uchar* nextPtr = image2.data + image2.step*ymap[y+1];

		//only one channel, compute them
		for( x = 0; x < width; x++ )
		{
			int x1 = xmap[x];
			dbuf[x] = (float)(lut[imgPtr[xmap[x+1]]] - lut[imgPtr[xmap[x-1]]]);
			dbuf[width + x] = (float)(lut[nextPtr[x1]] - lut[prevPtr[x1]]);
		}

		//compute the mag and angle using the dx and dy
		cartToPolar( Dx, Dy, Mag, Angle, false );

		for( x = 0; x < width; x++ )
		{
			//for the sake of voting to the bins
			float mag = dbuf[x+width*2], angle = dbuf[x+width*3]*angleScale - 0.5f;
			int hidx = cvFloor(angle);
			angle -= hidx;
			//in order to normalize the hidx
			if( hidx < 0 )
				hidx += nbins;
			else if( hidx >= nbins )
				hidx -= nbins;
			assert( (unsigned)hidx < (unsigned)nbins );

			float* tmp = (float*)bd[hidx].ptr(y);
			tmp[x] = mag*(1.f - angle);//vote to the lower bin
			hidx++;
			hidx &= hidx < nbins ? -1 : 0;//if hidx == 8, hidx+1 = 0
			tmp = (float*)bd[hidx].ptr(y);
			tmp[x] = mag*angle;//vote to the higher bin
		}
	}
	//t = (double)cvGetTickCount() - t;
	//cout << "Integral Compute time = " << t*1000./cv::getTickFrequency() << "ms" << endl;

	integral(bd[0], intBins[0]);
	integral(bd[1], intBins[1]);
	integral(bd[2], intBins[2]);
	integral(bd[3], intBins[3]);
	integral(bd[4], intBins[4]);
	integral(bd[5], intBins[5]);
	integral(bd[6], intBins[6]);

	//tbb::parallel_for(tbb::blocked_range<int>(0,8),ParallelIntegral(bd,intBins));

	size_t fi, nfeatures = features->size();
	//cout << nfeatures << endl;

	for( fi = 0; fi < nfeatures; fi++ )
	{
		//cout << fi << endl;
		featuresPtr[fi].updatePtrs( intBins );
	}

	return true;
}

bool HOG_8Evaluator::setImage( const Mat& image, Size _origWinSize )
{
	int rn = image.rows+1, cn = image.cols+1;
	Size imgSize;
	imgSize.width = image.cols, imgSize.height = image.rows;
	origWinSize = _origWinSize;

	if( image.cols < origWinSize.width || image.rows < origWinSize.height )
		return false;

	//---------------------------compute gradient----------------------------//
	int i, x, y;

	Mat image2;
	equalizeHist(image, image2);

	vector<Mat> bd;
	intBins.clear();

	for( i = 0; i < nbins; i++)
	{
		Mat buf1 = Mat::zeros(imgSize.height, imgSize.width, CV_32FC1);
		bd.push_back(buf1);
		Mat buf2 = Mat::zeros(rn, cn, CV_64FC1);
		intBins.push_back(buf2);
	}

	Size wholeSize;
	Point roiofs;
	image2.locateROI(wholeSize, roiofs);

	Mat_<float> _lut(1, 256);
	const float* lut = &_lut(0, 0);

	if(gammaCorrection)
		for(i = 0; i < 256; i++)
			_lut(0 ,i) = std :: sqrt((float)i);
	else
		for(i = 0; i < 256; i++)
			_lut(0, i) = (float)i;

	//get the image data's address after interpolated
	AutoBuffer<int> mapbuf(imgSize.width + imgSize.height + 4);
	int* xmap = (int*)mapbuf + 1;
	int* ymap = xmap + imgSize.width + 2;

	const int borderType = (int)BORDER_REFLECT_101;

	for( x = -1; x < imgSize.width + 1; x++ )
		xmap[x] = borderInterpolate(x + roiofs.x,
		wholeSize.width, borderType);
	for( y = -1; y < imgSize.height + 1; y++ )
		ymap[y] = borderInterpolate(y + roiofs.y,
		wholeSize.height, borderType);

	// x- & y- derivatives for the whole row
	int width = imgSize.width;
	AutoBuffer<float> _dbuf(width*4);
	float* dbuf = _dbuf;
	Mat Dx(1, width, CV_32F, dbuf);
	Mat Dy(1, width, CV_32F, dbuf + width);
	Mat Mag(1, width, CV_32F, dbuf + width*2);
	Mat Angle(1, width, CV_32F, dbuf + width*3);

	//int _nbins = nbins;
	float angleScale = (float)(nbins/CV_PI);//angle per bin

	//double t = (double)cvGetTickCount();
	for( y = 0; y < imgSize.height; y++ )
	{
		const uchar* imgPtr = image2.data + image2.step*ymap[y];
		const uchar* prevPtr = image2.data + image2.step*ymap[y-1];
		const uchar* nextPtr = image2.data + image2.step*ymap[y+1];

		//only one channel, compute them
		for( x = 0; x < width; x++ )
		{
			int x1 = xmap[x];
			dbuf[x] = (float)(lut[imgPtr[xmap[x+1]]] - lut[imgPtr[xmap[x-1]]]);
			dbuf[width + x] = (float)(lut[nextPtr[x1]] - lut[prevPtr[x1]]);
		}

		//compute the mag and angle using the dx and dy
		cartToPolar( Dx, Dy, Mag, Angle, false );

		for( x = 0; x < width; x++ )
		{
			//for the sake of voting to the bins
			float mag = dbuf[x+width*2], angle = dbuf[x+width*3]*angleScale - 0.5f;
			int hidx = cvFloor(angle);
			angle -= hidx;
			//in order to normalize the hidx
			if( hidx < 0 )
				hidx += nbins;
			else if( hidx >= nbins )
				hidx -= nbins;
			assert( (unsigned)hidx < (unsigned)nbins );

			float* tmp = (float*)bd[hidx].ptr(y);
			tmp[x] = mag*(1.f - angle);//vote to the lower bin
			hidx++;
			hidx &= hidx < nbins ? -1 : 0;//if hidx == 8, hidx+1 = 0
			tmp = (float*)bd[hidx].ptr(y);
			tmp[x] = mag*angle;//vote to the higher bin
		}
	}
	//t = (double)cvGetTickCount() - t;
	//cout << "Integral Compute time = " << t*1000./cv::getTickFrequency() << "ms" << endl;

	integral(bd[0], intBins[0]);
	integral(bd[1], intBins[1]);
	integral(bd[2], intBins[2]);
	integral(bd[3], intBins[3]);
	integral(bd[4], intBins[4]);
	integral(bd[5], intBins[5]);
	integral(bd[6], intBins[6]);
	integral(bd[7], intBins[7]);
	integral(bd[8], intBins[8]);

	//tbb::parallel_for(tbb::blocked_range<int>(0,8),ParallelIntegral(bd,intBins));

	size_t fi, nfeatures = features->size();
	//cout << nfeatures << endl;

	for( fi = 0; fi < nfeatures; fi++ )
	{
		//cout << fi << endl;
		featuresPtr[fi].updatePtrs( intBins );
	}

	return true;
}
bool EHOGEvaluator::setWindow( Point pt )
{
	if( pt.x < 0 || pt.y < 0 ||
		pt.x + origWinSize.width >= intBins[0].cols-2 ||
		pt.y + origWinSize.height >= intBins[0].rows-2 )
		return false;
	offset = pt.y * ((double)intBins[0].step/sizeof(double)) + pt.x;
	return true;
}

bool HOG_4Evaluator::setWindow( Point pt )
{
	if( pt.x < 0 || pt.y < 0 ||
		pt.x + origWinSize.width >= intBins[0].cols-2 ||
		pt.y + origWinSize.height >= intBins[0].rows-2 )
		return false;
	offset = pt.y * ((double)intBins[0].step/sizeof(double)) + pt.x;
	return true;
}

bool HOG_6Evaluator::setWindow( Point pt )
{
	if( pt.x < 0 || pt.y < 0 ||
		pt.x + origWinSize.width >= intBins[0].cols-2 ||
		pt.y + origWinSize.height >= intBins[0].rows-2 )
		return false;
	offset = pt.y * ((double)intBins[0].step/sizeof(double)) + pt.x;
	return true;
}

bool HOG_8Evaluator::setWindow( Point pt )
{
	if( pt.x < 0 || pt.y < 0 ||
		pt.x + origWinSize.width >= intBins[0].cols-2 ||
		pt.y + origWinSize.height >= intBins[0].rows-2 )
		return false;
	offset = pt.y * ((double)intBins[0].step/sizeof(double)) + pt.x;
	return true;
}

//--------------------------------------- Feature Evaluator ----------------------------------------------
Ptr<AFeatureEvaluator> AFeatureEvaluator::create(int featureType)
{
    return
		featureType == EHOG ? Ptr<AFeatureEvaluator>(new EHOGEvaluator) :
		featureType == HOG ? Ptr<AFeatureEvaluator>(new EHOGEvaluator) :
		featureType == HOG_4 ? Ptr<AFeatureEvaluator>(new HOG_4Evaluator) :
		featureType == HOG_6 ? Ptr<AFeatureEvaluator>(new HOG_6Evaluator) :
		featureType == HOG_8 ? Ptr<AFeatureEvaluator>(new HOG_8Evaluator) : Ptr<AFeatureEvaluator>();
}

//---------------------------------------- Classifier Cascade --------------------------------------------

ACascadeClassifier::ACascadeClassifier()
{
}

ACascadeClassifier::ACascadeClassifier(const string& filename)
{ load(filename); }

ACascadeClassifier::~ACascadeClassifier()
{
}

bool ACascadeClassifier::empty() const
{
    return oldCascade.empty() && stages.empty();
}

bool ACascadeClassifier::load(const string& filename)
{
    oldCascade.release();

	FileStorage fs(filename, FileStorage::READ);

	if( !fs.isOpened() )
        return false;

    if( read(fs.getFirstTopLevelNode()) )
        return true;

    fs.release();

    oldCascade = Ptr<CvHaarClassifierCascade>((CvHaarClassifierCascade*)cvLoad(filename.c_str(), 0, 0, 0));
    return !oldCascade.empty();
}

template<class AFEval>
inline int predictOrdered( ACascadeClassifier& cascade, Ptr<AFeatureEvaluator> &_feval )
{
    int si, nstages = (int)cascade.stages.size();
    int nodeOfs = 0, leafOfs = 0;
    AFEval& feval = (AFEval&)*_feval;
    float* cascadeLeaves = &cascade.leaves[0];
    ACascadeClassifier::DTreeNode* cascadeNodes = &cascade.nodes[0];
    ACascadeClassifier::DTree* cascadeWeaks = &cascade.classifiers[0];
    ACascadeClassifier::Stage* cascadeStages = &cascade.stages[0];

    for( si = 0; si < nstages; si++ )
    {
        ACascadeClassifier::Stage& stage = cascadeStages[si];
        int wi, ntrees = stage.ntrees;
        double sum = 0;

        for( wi = 0; wi < ntrees; wi++ )
        {
            ACascadeClassifier::DTree& weak = cascadeWeaks[stage.first + wi];
            int idx = 0, root = nodeOfs;

            do
            {
                CascadeClassifier::DTreeNode& node = cascadeNodes[root + idx];
                double val = feval(node.featureIdx);
                idx = val < node.threshold ? node.left : node.right;
            }
            while( idx > 0 );
            sum += cascadeLeaves[leafOfs - idx];
            nodeOfs += weak.nodeCount;
            leafOfs += weak.nodeCount + 1;
        }
        if( sum < stage.threshold )
            return -si;
    }
    return 1;
}

template<class AFEval>
inline int predictCategorical( ACascadeClassifier& cascade, Ptr<AFeatureEvaluator> &_feval )
{
	int si, nstages = (int)cascade.stages.size();
	int nodeOfs = 0, leafOfs = 0;
	AFEval& feval = (AFEval&)*_feval;
	size_t subsetSize = (cascade.ncategories + 31)/32;
	int* cascadeSubsets = &cascade.subsets[0];
	float* cascadeLeaves = &cascade.leaves[0];
	ACascadeClassifier::DTreeNode* cascadeNodes = &cascade.nodes[0];
    ACascadeClassifier::DTree* cascadeWeaks = &cascade.classifiers[0];
    ACascadeClassifier::Stage* cascadeStages = &cascade.stages[0];

	double sum;
    for( si = 0; si < nstages; si++ )
    {
        ACascadeClassifier::Stage& stage = cascadeStages[si];
        int wi, ntrees = stage.ntrees;
        /*double */sum = 0;

        for( wi = 0; wi < ntrees; wi++ )
        {
            ACascadeClassifier::DTree& weak = cascadeWeaks[stage.first + wi];
            int idx = 0, root = nodeOfs;
            do
            {
                ACascadeClassifier::DTreeNode& node = cascadeNodes[root + idx];
                int c = feval(node.featureIdx);
                const int* subset = &cascadeSubsets[(root + idx)*subsetSize];
                idx = (subset[c>>5] & (1 << (c & 31))) ? node.left : node.right;
            }
            while( idx > 0 );
            sum += cascadeLeaves[leafOfs - idx];
            nodeOfs += weak.nodeCount;
            leafOfs += weak.nodeCount + 1;
        }

        if( sum < stage.threshold )
            return -si;
    }
    return 1;
}

template<class AFEval>
inline int predictOrderedStump( ACascadeClassifier& cascade, Ptr<AFeatureEvaluator> &_feval )
{
    int si, nstages = (int)cascade.stages.size();
    int nodeOfs = 0, leafOfs = 0;
    AFEval& feval = (AFEval&)*_feval;
    float* cascadeLeaves = &cascade.leaves[0];
    ACascadeClassifier::DTreeNode* cascadeNodes = &cascade.nodes[0];
    ACascadeClassifier::Stage* cascadeStages = &cascade.stages[0];
    for( si = 0; si < nstages; si++ )
    {
        ACascadeClassifier::Stage& stage = cascadeStages[si];
        int wi, ntrees = stage.ntrees;
        double sum = 0;
        for( wi = 0; wi < ntrees; wi++, nodeOfs++, leafOfs+= 2 )
        {
            ACascadeClassifier::DTreeNode& node = cascadeNodes[nodeOfs];
            double val = feval(node.featureIdx);
            sum += cascadeLeaves[ val < node.threshold ? leafOfs : leafOfs+1 ];
        }
        if( sum < stage.threshold )
            return -si;
    }
    return 1;
}

template<class AFEval>
inline int predictCategoricalStump( ACascadeClassifier& cascade, Ptr<AFeatureEvaluator> &_feval )
{
    int si, nstages = (int)cascade.stages.size();
    int nodeOfs = 0, leafOfs = 0;
    AFEval& feval = (AFEval&)*_feval;
    size_t subsetSize = (cascade.ncategories + 31)/32;
    int* cascadeSubsets = &cascade.subsets[0];
    float* cascadeLeaves = &cascade.leaves[0];
    ACascadeClassifier::DTreeNode* cascadeNodes = &cascade.nodes[0];
    ACascadeClassifier::Stage* cascadeStages = &cascade.stages[0];

    for( si = 0; si < nstages; si++ )
    {
        ACascadeClassifier::Stage& stage = cascadeStages[si];
        int wi, ntrees = stage.ntrees;
        double sum = 0;

        for( wi = 0; wi < ntrees; wi++ )
        {
            ACascadeClassifier::DTreeNode& node = cascadeNodes[nodeOfs];
            int c = feval(node.featureIdx);
            const int* subset = &cascadeSubsets[nodeOfs*subsetSize];
            sum += cascadeLeaves[ subset[c>>5] & (1 << (c & 31)) ? leafOfs : leafOfs+1];
            nodeOfs++;
            leafOfs += 2;
        }
        if( sum < stage.threshold )
            return -si;
    }
    return 1;
}

int ACascadeClassifier::runAt( Ptr<AFeatureEvaluator> &_feval, Point pt )
{
    CV_Assert( oldCascade.empty() );
    /*if( !oldCascade.empty() )
        return cvRunHaarClassifierCascade(oldCascade, pt, 0);*/

    assert(featureType == AFeatureEvaluator::HOG_4 ||
		   featureType == AFeatureEvaluator::HOG_6 ||
		   featureType == AFeatureEvaluator::HOG_8);

	if (featureType == AFeatureEvaluator::HOG_4)
	{
		return !_feval->setWindow(pt) ? -1 :
			is_stump_based ? (
			predictCategoricalStump<HOG_4Evaluator>( *this, _feval ) ) :
		(
			predictCategorical<HOG_4Evaluator>( *this, _feval ));
	}
	else if (featureType == AFeatureEvaluator::HOG_6)
	{
		return !_feval->setWindow(pt) ? -1 :
			is_stump_based ? (
			predictCategoricalStump<HOG_6Evaluator>( *this, _feval ) ) :
		(
			predictCategorical<HOG_6Evaluator>( *this, _feval ));
	}
	else if(featureType == AFeatureEvaluator::HOG_8)
	{
		return !_feval->setWindow(pt) ? -1 :
			is_stump_based ? (
			predictCategoricalStump<HOG_8Evaluator>( *this, _feval ) ) :
		(
			predictCategorical<HOG_8Evaluator>( *this, _feval ));
	}
}

bool ACascadeClassifier::setImage( Ptr<AFeatureEvaluator> &_feval, const Mat& image )
{
    /*if( !oldCascade.empty() )
    {
        Mat sum(image.rows+1, image.cols+1, CV_32S);
        Mat tilted(image.rows+1, image.cols+1, CV_32S);
        Mat sqsum(image.rows+1, image.cols+1, CV_64F);
        integral(image, sum, sqsum, tilted);
        CvMat _sum = sum, _sqsum = sqsum, _tilted = tilted;
        cvSetImagesForHaarClassifierCascade( oldCascade, &_sum, &_sqsum, &_tilted, 1. );
        return true;
    }*/
    return empty() ? false : _feval->setImage(image, origWinSize );
}

struct ACascadeClassifierInvoker
{
    ACascadeClassifierInvoker( ACascadeClassifier& _cc, Size _sz1, int _stripSize, int _yStep, double _factor, ConcurrentRectVector& _vec )
    {
        cc = &_cc;
        sz1 = _sz1;
        stripSize = _stripSize;
        yStep = _yStep;
        factor = _factor;
        vec = &_vec;
    }

    void operator()(const BlockedRange& range) const
    {
        Ptr<AFeatureEvaluator> feval = cc->feval->clone();
        int y1 = range.begin()*stripSize, y2 = min(range.end()*stripSize, sz1.height);
        Size winSize(cvRound(cc->origWinSize.width*factor), cvRound(cc->origWinSize.height*factor));

        for( int y = y1; y < y2; y += yStep )
            for( int x = 0; x < sz1.width; x += yStep )
            {
                int r = cc->runAt(feval, Point(x, y));
                if( r > 0 )
                    vec->push_back(Rect(cvRound(x*factor), cvRound(y*factor),
                                        winSize.width, winSize.height));
                if( r == 0 )
                    x += yStep;
            }
    }

    ACascadeClassifier* cc;
    Size sz1;
    int stripSize, yStep;
    double factor;
    ConcurrentRectVector* vec;
};

struct getRect { Rect operator ()(const CvAvgComp& e) const { return e.rect; } };

void ACascadeClassifier::detectMultiScale( const Mat& image, vector<Rect>& objects,
                                          double scaleFactor, int minNeighbors,
                                          int flags, Size minSize )
{
    const double GROUP_EPS = 0.5;

    CV_Assert( scaleFactor > 1 && image.depth() == CV_8U );

    if( empty() )
        return;

    if( !oldCascade.empty() )
    {
        MemStorage storage(cvCreateMemStorage(0));
        CvMat _image = image;
        CvSeq* _objects = cvHaarDetectObjects( &_image, oldCascade, storage, scaleFactor,
                                              minNeighbors, flags, minSize );
        vector<CvAvgComp> vecAvgComp;
        Seq<CvAvgComp>(_objects).copyTo(vecAvgComp);
        objects.resize(vecAvgComp.size());
        std::transform(vecAvgComp.begin(), vecAvgComp.end(), objects.begin(), getRect());
        return;
    }

    objects.clear();

    Mat img = image, imgbuf(image.rows+1, image.cols+1, CV_8U);

    if( img.channels() > 1 )
    {
        Mat temp;
        cvtColor(img, temp, CV_BGR2GRAY);
        img = temp;
    }

    ConcurrentRectVector allCandidates;

    for( double factor = 1; ; factor *= scaleFactor )
    {
        int stripCount, stripSize;
        Size winSize( cvRound(origWinSize.width*factor), cvRound(origWinSize.height*factor) );
        Size sz( cvRound( img.cols/factor ), cvRound( img.rows/factor ) );
        Size sz1( sz.width - origWinSize.width, sz.height - origWinSize.height );

        if( sz1.width <= 0 || sz1.height <= 0 )
            break;
        if( winSize.width < minSize.width || winSize.height < minSize.height )
            continue;

        int yStep = factor > 2. ? 1 : 2;
    #ifdef HAVE_TBB
        const int PTS_PER_THREAD = 100;
        stripCount = max(((sz1.height*sz1.width)/(yStep*yStep) + PTS_PER_THREAD/2)/PTS_PER_THREAD, 1);
        stripSize = (sz1.height + stripCount - 1)/stripCount;
        stripSize = (stripSize/yStep)*yStep;
    #else
        stripCount = 1;
        stripSize = sz1.height;
    #endif

        Mat img1( sz, CV_8U, imgbuf.data );
        resize( img, img1, sz, 0, 0, CV_INTER_LINEAR );
        if( !feval->setImage( img1, origWinSize ) )
            break;

        parallel_for(BlockedRange(0, stripCount), ACascadeClassifierInvoker(*this, sz1, stripSize, yStep, factor, allCandidates));
    }

    objects.resize(allCandidates.size());
    std::copy(allCandidates.begin(), allCandidates.end(), objects.begin());
    groupRectangles( objects, minNeighbors, GROUP_EPS );
}

bool ACascadeClassifier::read(const FileNode& root)
{
    // load stage params
    string stageTypeStr = (string)root[CC_STAGE_TYPE];
    if( stageTypeStr == CC_BOOST )
        stageType = BOOST;
    else
        return false;

    string featureTypeStr = (string)root[CC_FEATURE_TYPE];
    if( featureTypeStr == CC_HAAR )
        featureType = AFeatureEvaluator::HAAR;
    else if( featureTypeStr == CC_LBP )
        featureType = AFeatureEvaluator::LBP;
    else if( featureTypeStr == CC_EHOG)
		featureType = AFeatureEvaluator::EHOG;
	else if( featureTypeStr == CC_HOG)
		featureType = AFeatureEvaluator::EHOG;
	else if( featureTypeStr == CC_HOG_4)
		featureType = AFeatureEvaluator::HOG_4;
	else if( featureTypeStr == CC_HOG_6)
		featureType = AFeatureEvaluator::HOG_6;
	else if( featureTypeStr == CC_HOG_8)
		featureType = AFeatureEvaluator::HOG_8;
	else
        return false;

    origWinSize.width = (int)root[CC_WIDTH];
    origWinSize.height = (int)root[CC_HEIGHT];
    CV_Assert( origWinSize.height > 0 && origWinSize.width > 0 );

    is_stump_based = (int)(root[CC_STAGE_PARAMS][CC_MAX_DEPTH]) == 1 ? true : false;

    // load feature params
    FileNode fn = root[CC_FEATURE_PARAMS];
    if( fn.empty() )
        return false;

    ncategories = fn[CC_MAX_CAT_COUNT];
    int subsetSize = (ncategories + 31)/32,
        nodeStep = 3 + ( ncategories>0 ? subsetSize : 1 );

    // load stages
    fn = root[CC_STAGES];
    if( fn.empty() )
        return false;

    stages.reserve(fn.size());
    classifiers.clear();
    nodes.clear();

    FileNodeIterator it = fn.begin(), it_end = fn.end();

    for( int si = 0; it != it_end; si++, ++it )
    {
        FileNode fns = *it;
        Stage stage;
        stage.threshold = fns[CC_STAGE_THRESHOLD];
        fns = fns[CC_WEAK_CLASSIFIERS];
        if(fns.empty())
            return false;
        stage.ntrees = (int)fns.size();
        stage.first = (int)classifiers.size();
        stages.push_back(stage);
        classifiers.reserve(stages[si].first + stages[si].ntrees);

        FileNodeIterator it1 = fns.begin(), it1_end = fns.end();
        for( ; it1 != it1_end; ++it1 ) // weak trees
        {
            FileNode fnw = *it1;
            FileNode internalNodes = fnw[CC_INTERNAL_NODES];
            FileNode leafValues = fnw[CC_LEAF_VALUES];
            if( internalNodes.empty() || leafValues.empty() )
                return false;
            DTree tree;
            tree.nodeCount = (int)internalNodes.size()/nodeStep;
            classifiers.push_back(tree);

            nodes.reserve(nodes.size() + tree.nodeCount);
            leaves.reserve(leaves.size() + leafValues.size());
            if( subsetSize > 0 )
                subsets.reserve(subsets.size() + tree.nodeCount*subsetSize);

            FileNodeIterator it2 = internalNodes.begin(), it2_end = internalNodes.end();

            for( ; it2 != it2_end; ) // nodes
            {
                DTreeNode node;
                node.left = (int)*it2; ++it2;
                node.right = (int)*it2; ++it2;
                node.featureIdx = (int)*it2; ++it2;
                if( subsetSize > 0 )
                {
                    for( int j = 0; j < subsetSize; j++, ++it2 )
                        subsets.push_back((int)*it2);
                    node.threshold = 0.f;
                }
                else
                {
                    node.threshold = (float)*it2; ++it2;
                }
                nodes.push_back(node);
            }

            it2 = leafValues.begin(), it2_end = leafValues.end();

            for( ; it2 != it2_end; ++it2 ) // leaves
                leaves.push_back((float)*it2);
        }
    }

    // load features
    feval = AFeatureEvaluator::create(featureType);
    fn = root[CC_FEATURES];
    if( fn.empty() )
        return false;

    return feval->read(fn);
}

 // namespace cv

/* End of file. */