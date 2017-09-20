#ifndef _LABELMANAGER_H_
#define _LABELMANAGER_H_

#include <map>
#include <opencv2/opencv.hpp>


struct Label{
    int m_avgR;
    int m_avgG;
    int m_avgB;
    cv::Vec3b m_avgColor;
    float m_avgSimilarity;
    float m_minSimilarity;

    long m_numPixel;
    long m_accuR; //accumulative R
    long m_accuG;
    long m_accuB;
    float m_accuSimilarity; //its number = m_numPixel -1;

};


//Label starts from 1.
class LabelManager{
public:
    LabelManager();
    ~LabelManager();
    int m_invertColor;

    void linkMat(cv::Mat mat);

    int getLeastVacancyLabel();
    float getMinSimilarity(const int label);
    void updateLabel(const int label, const int i, const int j, const float similar);
    void updateLabel(const int label, const float similar);
    void mergeLabel(const int minL, const int maxL);
    void computerAvgRGB();
    cv::Vec3b getColor(const int label);
    void statisticsCluster(const std::string filename);
    int getOneLabel(const int clusterSize);
    void deleteZerosPixelLabel();
    int getNumCluster();


private:
    std::map<int,Label> m_labelMap;
    cv::Mat  m_Mat;
    cv::Vec3b getRandomColor();
    cv::RNG m_randomNumGenerator;
};







#endif
