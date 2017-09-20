#ifndef _IMAGEANALYZE_H_
#define _IMAGEANALYZE_H_

#include <opencv2/opencv.hpp>
#include "LabelManager.h"

//Mat use BGR channels order

struct PixelUnit
{
    cv::Vec3f m_normBGR; //normalize R,G,B
    float m_similarity[8]; //projection similarity in 8 directions;
    int m_label;
};

struct Coordinates{
    int i;
    int j;
};

struct PossibleRelative{
    int m_label;
    float m_similarity;
};

class ImageAnalyzer
{
public:
    ImageAnalyzer();
    ~ImageAnalyzer();

    std::string m_originalFileName;
    float m_similarThreshold;
    bool m_preProcess;
    int m_minClusterSize;

    int readImageInialize();
    void preprocessMat();
    void clusterMat();
    void setInvertColor(const int invertColor);
    long m_K; //K in opencv::kmeans;


private:
    cv::Mat m_originalMat;
    cv::Mat m_Mat; //m_Mat was used with mean filter
    cv::Mat m_clusterMat; //for clustering result
    cv::Mat m_kmeansMat;  //for kmeans method
    int m_nRows;
    int m_nCols;

    cv::Scalar m_mean;
    cv::Scalar m_std;

    float m_generalMinMaxSimilarity; //For whole image
    float m_generalAvgMaxSimilarity; //For whole image

    PixelUnit ** m_pixelMatrix;

    std::string m_WndOriginal;
    std::string m_WndAfterFilter;
    std::string m_WndClustring;
    std::string m_WndKmeans;

    LabelManager m_labelMgr;


    void createPixelMatrix();
    void deletePixelMatrix();

    void upateMeanStdDev();
    void updateNormalizeBGR();
    void calculateSimilarity();
    void calculateMinMaxSimilarity();//get general min similarity among all local max
    float getLocalMaxSimilarity(const float* similarityArray );
    float getLocalMaxSimilarity(const int i, const int j);
    int getLocalMaxDirection(const float* similarityArray);


    void BFSExpandLabel(int label,const int i,const int j);
    Coordinates getPeerPos(const Coordinates pos,const int k);
    Coordinates getPeerPos(const int i,const int j,const int k);
    void updateSimilarityThreshold();
    int mergeLabel(const int label1,const int label2); //return minL;

    void showResultImage();
    void statisticsCluster();
    void saveClusterImage();

    void mergeSingularity();
    PossibleRelative getPossibleRelative(const int label, const int i, const int j);
    PossibleRelative getMaxPossibleRelative(const int label, const int range);
    void mergeSmallCluster(int clusterSize);

    float projectionSimilarity(const cv::Vec3f& a,const cv::Vec3f& b);
    void Kmeans(const int K);
    void saveKmeansImage(const int K);
};


#endif
