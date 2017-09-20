#include "LabelManager.h"

LabelManager::LabelManager(){
    m_labelMap.clear();
    m_invertColor = 0;
}


LabelManager::~LabelManager(){
    m_labelMap.clear();
}

void LabelManager::linkMat(cv::Mat mat){
    m_Mat = mat;
}


int LabelManager::getLeastVacancyLabel(){
    int label = 1;
    while (1 == m_labelMap.count(label)){
        label++;
    }
    return label;
}

float LabelManager::getMinSimilarity(const int label){
    //return m_labelMap[label].m_minSimilarity;
    Label* labelUnit = &(m_labelMap[label]);
    long nPixel = labelUnit->m_numPixel;
    if (nPixel > 1) labelUnit->m_avgSimilarity = labelUnit->m_accuSimilarity/(nPixel-1);
    else labelUnit->m_avgSimilarity = labelUnit->m_minSimilarity;
    return (labelUnit->m_avgSimilarity- 0.0115);

}

void LabelManager::updateLabel(const int label, const int i, const int j, const float similarity ){

     if (0 == m_labelMap.count(label)){
        Label* labelUnit = &(m_labelMap[label]);
        labelUnit->m_numPixel = 1;
        labelUnit->m_accuB = m_Mat.at<cv::Vec3b>(i,j).val[0];
        labelUnit->m_accuG = m_Mat.at<cv::Vec3b>(i,j).val[1];
        labelUnit->m_accuR = m_Mat.at<cv::Vec3b>(i,j).val[2];
        labelUnit->m_accuSimilarity = 0;
        labelUnit->m_minSimilarity = similarity;
     }
     else{
        Label* labelUnit = &(m_labelMap[label]);
        labelUnit->m_numPixel++;
        labelUnit->m_accuB += m_Mat.at<cv::Vec3b>(i,j).val[0];
        labelUnit->m_accuG += m_Mat.at<cv::Vec3b>(i,j).val[1];
        labelUnit->m_accuR += m_Mat.at<cv::Vec3b>(i,j).val[2];
        labelUnit->m_accuSimilarity += similarity;
        if (labelUnit->m_minSimilarity > similarity) labelUnit->m_minSimilarity = similarity;

     }
 }

void LabelManager::updateLabel(const int label, const float similar){
     Label* labelUnit = &(m_labelMap[label]);
     labelUnit->m_accuSimilarity += similar;
     if (labelUnit->m_minSimilarity > similar) labelUnit->m_minSimilarity = similar;
}


void LabelManager::mergeLabel(const int minL, const int maxL){
    Label* labelMinL = &(m_labelMap[minL]);
    Label* labelMaxL = &(m_labelMap[maxL]);
    labelMinL->m_numPixel += labelMaxL->m_numPixel;
    labelMinL->m_accuB += labelMaxL->m_accuB;
    labelMinL->m_accuG += labelMaxL->m_accuG;
    labelMinL->m_accuR += labelMaxL->m_accuR;
    labelMinL->m_accuSimilarity += labelMaxL->m_accuSimilarity;
    if (labelMinL->m_minSimilarity > labelMaxL->m_minSimilarity) labelMinL->m_minSimilarity = labelMaxL->m_minSimilarity;
    m_labelMap.erase(maxL);
}

void LabelManager::computerAvgRGB()
{
    std::map<int,Label>::iterator iter = m_labelMap.begin();
    while (iter != m_labelMap.end())
    {
        long nPixel = iter->second.m_numPixel;
        iter->second.m_avgR = iter->second.m_accuR/nPixel;
        iter->second.m_avgG = iter->second.m_accuG/nPixel;
        iter->second.m_avgB = iter->second.m_accuB/nPixel;
        if (nPixel > 1) iter->second.m_avgSimilarity = iter->second.m_accuSimilarity/(nPixel-1);
        else iter->second.m_avgSimilarity = 0;

        if (0 == m_invertColor){
            iter->second.m_avgColor.val[0] = iter->second.m_avgB;
            iter->second.m_avgColor.val[1] = iter->second.m_avgG;
            iter->second.m_avgColor.val[2] = iter->second.m_avgR;
        }
        else if (1 == m_invertColor){
            iter->second.m_avgColor.val[0] = 255 - iter->second.m_avgB;
            iter->second.m_avgColor.val[1] = 255 - iter->second.m_avgG;
            iter->second.m_avgColor.val[2] = 255 - iter->second.m_avgR;

        }
        else {
            iter->second.m_avgColor = getRandomColor();

        }


        iter++;
    }

}

cv::Vec3b LabelManager::getColor(const int label){
    return m_labelMap[label].m_avgColor;
}

cv::Vec3b LabelManager::getRandomColor(){
    return  cv::Vec3b(m_randomNumGenerator.uniform(0, 255),
                      m_randomNumGenerator.uniform(0, 255),
                      m_randomNumGenerator.uniform(0, 255));


}

void LabelManager::statisticsCluster(const std::string filename)
{
    FILE* pFile = fopen(filename.c_str(), "w");

    std::string tableHead("ClusterID,NumPixel,minSimilarity,AvgSimilarity, AvgR,AvgG,AvgB\n");
    fwrite(tableHead.c_str(), sizeof(char), tableHead.size(), pFile);

    float totalSimilarity = 0;
    float globalMinSimilarity = 1;
    long numCluster = 0;
    long numPixel = 0;


    char outLine[50];
    int length = 0;
    std::map<int,Label>::iterator iter = m_labelMap.begin();
    while (iter != m_labelMap.end())
    {
        length = sprintf(outLine, "%d,%lu,%.4f,%.4f,%d,%d,%d\n",
                         iter->first,iter->second.m_numPixel,iter->second.m_minSimilarity, iter->second.m_avgSimilarity,
                         iter->second.m_avgR,iter->second.m_avgG,iter->second.m_avgB);
        outLine[length] = '\0';
        fwrite(outLine, sizeof(char), length, pFile);

        numCluster++;
        numPixel += iter->second.m_numPixel;
        totalSimilarity += iter->second.m_avgSimilarity;
        if (iter->second.m_minSimilarity < globalMinSimilarity) globalMinSimilarity = iter->second.m_minSimilarity;

        iter++;
    }

    //print sum row
    totalSimilarity = totalSimilarity/numCluster;
    length = sprintf(outLine, "Sum,%lu,%.4f,%.4f\n",numPixel, globalMinSimilarity,totalSimilarity);
    outLine[length] = '\0';
    fwrite(outLine, sizeof(char), length, pFile);

    fflush(pFile);
    fclose(pFile);
    pFile = NULL;

    printf("Output: %s\n", filename.c_str());
}

 int LabelManager::getOneLabel(const int clusterSize){

    std::map<int,Label>::iterator iter = m_labelMap.begin();
    while (iter != m_labelMap.end())
    {
        if (clusterSize == iter->second.m_numPixel) return iter->first;
        iter++;
    }
    return 0; //Can't find the clusterSize;

 }

void LabelManager::deleteZerosPixelLabel()
{
    std::map<int,Label>::iterator iter = m_labelMap.begin();
    while (iter != m_labelMap.end())
    {
        if (0 == iter->second.m_numPixel)
        {
            iter = m_labelMap.erase(iter);

        }
        else
        {
            iter++;
        }

    }

}

int LabelManager::getNumCluster(){
    return (long)m_labelMap.size();

}






