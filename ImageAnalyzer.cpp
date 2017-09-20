#include "ImageAnalyzer.h"
#include <list>


 ImageAnalyzer::ImageAnalyzer(){
    m_pixelMatrix = NULL;
    m_nRows = 0;
    m_nCols = 0;
    m_similarThreshold = -1;
    m_generalMinMaxSimilarity = 0; //For whole image
    m_generalAvgMaxSimilarity = 0; //For whole image
    m_preProcess = 1;
    m_minClusterSize = 9;
    m_K = -1;
 }

 ImageAnalyzer::~ImageAnalyzer(){
    deletePixelMatrix();
 }

 void ImageAnalyzer::createPixelMatrix(){
    if (NULL == m_pixelMatrix){
        m_pixelMatrix = new PixelUnit*[m_nRows];
        for (int i=0;i<m_nRows;i++){
            *(m_pixelMatrix+i) = new PixelUnit[m_nCols];
        }

    }
    else{
        assert(1);

    }
 }

void ImageAnalyzer::deletePixelMatrix(){
   if (NULL != m_pixelMatrix){
        for(int i=0; i<m_nRows; i++) delete[] *(m_pixelMatrix+i);
        delete[] m_pixelMatrix;
        m_pixelMatrix = NULL;
   }

}

int ImageAnalyzer::readImageInialize(){
    m_originalMat = cv::imread(m_originalFileName);   // Read the file
    if(! m_originalMat.data )                  // Check for invalid input
    {
        std::cout << "Error: Could not open or find the image." << std::endl ;
        return -1;
    }
    m_WndOriginal = "Original: "+ m_originalFileName;
    m_WndAfterFilter = "After Filter: " + m_originalFileName;
    m_WndClustring = "Auto Clustering Result: " + m_originalFileName;
    m_WndKmeans = "Kmeans: " + m_originalFileName;


    cv::namedWindow( m_WndOriginal, cv::WINDOW_AUTOSIZE );// Create a window for display.
    cv::imshow( m_WndOriginal, m_originalMat);                   // Show our image inside it.
    cv::waitKey(20);
    m_Mat = m_originalMat.clone();
    m_nRows = m_Mat.rows;
    m_nCols = m_Mat.cols;
    createPixelMatrix();
    m_labelMgr.linkMat(m_Mat);// final color uses original Mat

    if (!m_preProcess)
    {
        upateMeanStdDev();
        updateNormalizeBGR();
        calculateSimilarity();
        calculateMinMaxSimilarity();
        updateSimilarityThreshold();
    }

    return 0;
}

void ImageAnalyzer::upateMeanStdDev(){
    cv::meanStdDev(m_Mat,m_mean,m_std);
    return;
 }

void ImageAnalyzer::updateNormalizeBGR()
{
    for (int i=0; i<m_nRows; i++)
    {
        for (int j=0; j<m_nCols; j++)
        {
            for (int k=0; k<3; k++) //BGR channels
            {
                if (0== m_std.val[k])
                {
                    m_pixelMatrix[i][j].m_normBGR.val[k] = 0;
                }
                else{
                   m_pixelMatrix[i][j].m_normBGR.val[k] = (m_Mat.at<cv::Vec3b>(i,j).val[k]-m_mean[k])/m_std.val[k];
                }

            }
        }

    }

}

//also set label = -1;
void ImageAnalyzer::calculateSimilarity()
{
    //set all label = -1 and all similarity = -1
    for (int i=0; i<m_nRows; i++)
    {
        for (int j=0; j<m_nCols; j++)
        {
            m_pixelMatrix[i][j].m_label = -1;
            for (int d=0;d<8;d++) m_pixelMatrix[i][j].m_similarity[d] = -1;
        }
    }

    // calculate similarity
    for (int i=1; i<m_nRows-1; i++)
    {
        for (int j=1; j<m_nCols-1; j++)
        {
            for(int k=0; k<8; k++)
            {
                if (-1 != m_pixelMatrix[i][j].m_similarity[k]) continue;

                int peeri = -1;
                int peerj = -1;
                int peerk = -1;
                switch (k)
                {
                case 0: peeri = i-1; peerj = j;   peerk = 4; break;
                case 1: peeri = i-1; peerj = j+1; peerk = 5; break;
                case 2: peeri = i;   peerj = j+1; peerk = 6; break;
                case 3: peeri = i+1; peerj = j+1; peerk = 7; break;
                case 4: peeri = i+1; peerj = j;   peerk = 0; break;
                case 5: peeri = i+1; peerj = j-1; peerk = 1; break;
                case 6: peeri = i;   peerj = j-1; peerk = 2; break;
                case 7: peeri = i-1; peerj = j-1; peerk = 3; break;
                default:
                    assert(1);
                }

                float similarity = projectionSimilarity(m_pixelMatrix[i][j].m_normBGR, m_pixelMatrix[peeri][peerj].m_normBGR);
                m_pixelMatrix[i][j].m_similarity[k] = similarity;
                m_pixelMatrix[peeri][peerj].m_similarity[peerk] = similarity;

            }
        }
    }
}

Coordinates ImageAnalyzer::getPeerPos(const Coordinates pos,const int k){
    Coordinates peerPos;
    switch (k)
    {
     case 0: peerPos.i = pos.i-1; peerPos.j = pos.j;   break;
     case 1: peerPos.i = pos.i-1; peerPos.j = pos.j+1; break;
     case 2: peerPos.i = pos.i;   peerPos.j = pos.j+1; break;
     case 3: peerPos.i = pos.i+1; peerPos.j = pos.j+1; break;
     case 4: peerPos.i = pos.i+1; peerPos.j = pos.j;   break;
     case 5: peerPos.i = pos.i+1; peerPos.j = pos.j-1; break;
     case 6: peerPos.i = pos.i;   peerPos.j = pos.j-1; break;
     case 7: peerPos.i = pos.i-1; peerPos.j = pos.j-1; break;
     default:
          assert(1);
    }
    return peerPos;


}

Coordinates ImageAnalyzer::getPeerPos(const int i,const int j,const int k){
    Coordinates pos;
    pos.i = i;
    pos.j = j;
    return getPeerPos(pos,k);
}

float ImageAnalyzer::projectionSimilarity(const cv::Vec3f& a,const cv::Vec3f& b)
{
    float norm1 = cv::norm(a);
    float norm2 = cv::norm(b);
    float maxNorm = (norm1> norm2)? norm1: norm2;
    float dotProduct = 0;
    for (int i =0;i<3;i++){
        dotProduct +=a.val[i]*b.val[i];
    }
    if (0 != maxNorm) return dotProduct/(maxNorm*maxNorm);
    else return 1; // a, b are zeros vector, they are same mean in RGB
}


//get general min similarity among all local max
void ImageAnalyzer::calculateMinMaxSimilarity()
{
    float generalMin = 2;
    float sumSimilar = 0;
    long n= 0;
    for (int i=1; i<m_nRows-1; i++)
    {
        for (int j=1; j<m_nCols-1; j++)
        {
            float localMax = getLocalMaxSimilarity(m_pixelMatrix[i][j].m_similarity);
            sumSimilar += localMax;
            n++;

            //if (localMax < 0.2) printf("i =%d, j =%d, localMax = %f\n", i,j,localMax);

            if (localMax < generalMin) generalMin = localMax;
        }
    }
    m_generalAvgMaxSimilarity = sumSimilar/n;
    m_generalMinMaxSimilarity = generalMin;
    //printf("General average similarity = %f ; General Min Max Similarity = %f \n", m_generalAvgMaxSimilarity, m_generalMinMaxSimilarity);

}


float ImageAnalyzer::getLocalMaxSimilarity(const float* similarityArray){
    float localMax = -1;
    for (int i=0; i<8; i++){
        if (similarityArray[i] > localMax)  localMax = similarityArray[i];
    }

    return localMax;
}

float ImageAnalyzer::getLocalMaxSimilarity(const int i, const int j){
    return getLocalMaxSimilarity(m_pixelMatrix[i][j].m_similarity);
}

int ImageAnalyzer::getLocalMaxDirection(const float* similarityArray){
    float localMax = -1;
    int maxk = 0;
    for (int i=0; i<8; i++){
        if (similarityArray[i] > localMax){
            localMax = similarityArray[i];
            maxk = i;
         }
    }
    return maxk;
}

void ImageAnalyzer::preprocessMat()
{
    int nIter = 0;
    std::string filterWnd;
    const float epsilon = 1e-4;

    float m_OldMinMaxSimilarity = 1+ m_generalMinMaxSimilarity;
    float m_OldAvgMaxSimilarity = 1+ m_generalAvgMaxSimilarity;

    while (fabs(m_OldMinMaxSimilarity - m_generalMinMaxSimilarity) > epsilon ||
           fabs(m_OldAvgMaxSimilarity - m_generalAvgMaxSimilarity) > epsilon ||
           m_generalMinMaxSimilarity < epsilon)
    {
        m_OldMinMaxSimilarity = m_generalMinMaxSimilarity;
        m_OldAvgMaxSimilarity = m_generalAvgMaxSimilarity;

        cv::medianBlur(m_Mat,m_Mat,3);
        upateMeanStdDev();
        updateNormalizeBGR();
        calculateSimilarity();
        calculateMinMaxSimilarity();
        nIter++;
    }
    printf("Progress: Automatically MeanFiler Iterate %d times.\n", nIter);
//    cv::namedWindow( m_WndAfterFilter, cv::WINDOW_AUTOSIZE );
//    cv::imshow( m_WndAfterFilter, m_Mat);
//    cv::waitKey(4);
    updateSimilarityThreshold();

}

void ImageAnalyzer::updateSimilarityThreshold(){
    if (-1 == m_similarThreshold){
        m_similarThreshold = int(m_generalAvgMaxSimilarity *97)/100.0;
        printf("Progress: the automatic similarity threshold = %.4f\n", m_similarThreshold);
    }
    else{
        printf("Progress: the user specified similarity threshold = %.2f\n", m_similarThreshold);
    }
}


void ImageAnalyzer::clusterMat()
{
    printf("Hint: A big-size image needs more computation time. please wait......\n");

    for(int i =1; i<m_nRows-1; i++)
    {
        for(int j=1; j<m_nCols-1; j++)
        {
            if (-1 == m_pixelMatrix[i][j].m_label)
            {
                int label = m_labelMgr.getLeastVacancyLabel();
                float localMaxSimilarity = getLocalMaxSimilarity(i,j);
                m_pixelMatrix[i][j].m_label = label;
                m_labelMgr.updateLabel(label,i,j,localMaxSimilarity);
                BFSExpandLabel(label,i,j);

            }
        }
    }
    //mergeSingularity();
    //mergeSmallCluster(m_minClusterSize);
    m_labelMgr.computerAvgRGB();
    showResultImage();
    saveClusterImage();
    statisticsCluster();

    long K = m_K;
    if (-1 == K) K = m_labelMgr.getNumCluster()-1;
    Kmeans(K);

}


void ImageAnalyzer::BFSExpandLabel(int label,const int i,const int j)
{
    Coordinates pos;
    pos.i = i;
    pos.j = j;
    std::list<Coordinates> heap;
    heap.push_back(pos);

    while (0 != heap.size())
    {
        pos = heap.front();
        heap.pop_front();

        float similarThreshold = m_labelMgr.getMinSimilarity(label);
        //printf("i=%d,j=%d,similar = %.4f\t",pos.i,pos.j,similarThreshold);

        for(int k=0; k<8; k++)
        {

            float similarity = m_pixelMatrix[pos.i][pos.j].m_similarity[k];
            if ( similarity >= similarThreshold)
            {
                Coordinates peerPos = getPeerPos(pos,k);
                int peerLabel = m_pixelMatrix[peerPos.i][peerPos.j].m_label;
                if (-1 == peerLabel)
                {
                    //BFS Expand Label
                    m_pixelMatrix[peerPos.i][peerPos.j].m_label = label;
                    m_labelMgr.updateLabel(label,peerPos.i,peerPos.j,similarity);
                    heap.push_back(peerPos);
                }
                else if ( label != peerLabel)
                {
                    //Merge label
                    label = mergeLabel(label,peerLabel);
                    m_labelMgr.updateLabel(label,similarity);
                    float const newSimlarThreshold = m_labelMgr.getMinSimilarity(label);
                    if ( newSimlarThreshold < similarThreshold){
                        similarThreshold = newSimlarThreshold;
                        k=0;
                    } ;

                }
                else
                {
                    continue;
                }

            }
        }
    }
}

int ImageAnalyzer::mergeLabel(const int label1,const int label2){
    assert (label1 != label2);
    int minLabel = 0;
    int maxLabel = 0;
    if (label1 < label2){
       minLabel = label1;
       maxLabel = label2;
    }
    else{
       minLabel = label2;
       maxLabel = label1;
    }

    for(int i =0; i<m_nRows; i++)
    {
        for(int j=0;j<m_nCols; j++)
        {
            if (maxLabel == m_pixelMatrix[i][j].m_label){

                m_pixelMatrix[i][j].m_label = minLabel;

            }

        }

    }
    m_labelMgr.mergeLabel(minLabel,maxLabel);

    return minLabel;
}

void ImageAnalyzer::showResultImage(){

    m_clusterMat.create(cv::Size(m_nCols,m_nRows),m_Mat.type());

    for(int i =0; i<m_nRows; i++)
    {
        for(int j=0;j<m_nCols; j++)
        {
            int label = m_pixelMatrix[i][j].m_label;
            if (label >=1) m_clusterMat.at<cv::Vec3b>(i,j) = m_labelMgr.getColor(label);
            else {
                //printf(" i=%d,j=%d \t",i,j);
                m_clusterMat.at<cv::Vec3b>(i,j) = cv::Vec3b(0,0,0);
            }
         }
    }

    cv::namedWindow( m_WndClustring, cv::WINDOW_AUTOSIZE );
    cv::imshow( m_WndClustring, m_clusterMat);
    cv::waitKey(20);

}

void ImageAnalyzer::setInvertColor(const int invertColor){
    m_labelMgr.m_invertColor = invertColor;
}

void ImageAnalyzer::statisticsCluster()
{
    //construct file name
    int length;
    char threshold[20];
    length = sprintf(threshold,"_threshold_%d",int(m_similarThreshold*100+0.5));
    threshold[length] = '\0';

    length = m_originalFileName.size();
    std::string filename = m_originalFileName.substr(0,length-4)+threshold+"%.csv";

    m_labelMgr.statisticsCluster(filename);
}

void ImageAnalyzer::saveClusterImage()
{

    int length;
    char threshold[20];
    length = sprintf(threshold,"_threshold_%d%%",int(m_similarThreshold*100+0.5));
    threshold[length] = '\0';

    length = m_originalFileName.size();
    std::string name1 = m_originalFileName.substr(0,length-4);
    std::string name2 = m_originalFileName.substr(length-4,4);

    std::string filename = name1+threshold + name2;
    cv::imwrite(filename,m_clusterMat);

    printf("Output: %s\n", filename.c_str());

}

void ImageAnalyzer::saveKmeansImage(const int K)
{
    int length = 0;
    char Kmeans_K[30];
    length = sprintf(Kmeans_K,"_Kmeans_%d",K);
    Kmeans_K[length] = '\0';

    length = m_originalFileName.size();
    std::string name1 = m_originalFileName.substr(0,length-4);
    std::string name2 = m_originalFileName.substr(length-4,4);

    std::string filename = name1+Kmeans_K + name2;
    cv::imwrite(filename,m_kmeansMat);

    printf("Output: %s\n", filename.c_str());

}

void ImageAnalyzer::mergeSingularity()
{
    for(int i =0; i<m_nRows; i++)
    {
        for(int j=0; j<m_nCols; j++)
        {
            int myLabel = m_pixelMatrix[i][j].m_label;
            if (-1 == myLabel) continue;
            int maxk = getLocalMaxDirection(m_pixelMatrix[i][j].m_similarity);
            float similarity = getLocalMaxSimilarity(m_pixelMatrix[i][j].m_similarity);
            Coordinates peerPos = getPeerPos(i,j,maxk);
            int peerLabel = m_pixelMatrix[peerPos.i][peerPos.j].m_label;
            if (-1 == peerLabel) continue;
            if (myLabel != peerLabel)
            {
                myLabel = mergeLabel(myLabel,peerLabel);
                m_labelMgr.updateLabel(myLabel,similarity);
            }

        }
    }

}


PossibleRelative ImageAnalyzer::getPossibleRelative(const int label, const int i, const int j)
{
    PossibleRelative possiRelative;
    possiRelative.m_label = -1;
    possiRelative.m_similarity = -1;

    for (int k=0; k<8; k++)
    {
        Coordinates peerPos = getPeerPos(i,j,k);
        PixelUnit& peerPixel =  m_pixelMatrix[peerPos.i][peerPos.j];
        if (label != peerPixel.m_label)
        {
            if (m_pixelMatrix[i][j].m_similarity[k] > possiRelative.m_similarity)
            {
                possiRelative.m_label = peerPixel.m_label;
                possiRelative.m_similarity = m_pixelMatrix[i][j].m_similarity[k];

            }
        }

    }
    return possiRelative;
}

PossibleRelative ImageAnalyzer::getMaxPossibleRelative(const int label, const int range)
{
    PossibleRelative maxPossiRelative;
    maxPossiRelative.m_label = -1;
    maxPossiRelative.m_similarity = -1;

    for(int i =1; i<m_nRows-1; i++)
    {
        for(int j=1; j<m_nCols-1; j++)
        {
            if(label == m_pixelMatrix[i][j].m_label)
            {
                int ilow = i;
                int ihigh = (i+range < m_nRows-1)? (i+range): (m_nRows-1);
                int jlow = (j-range < 0)? 0:(j-range);
                int jhigh = (j+range < m_nCols-1)? (j+range): (m_nCols-1);

                for(int ii= ilow; ii<ihigh; ii++)
                {
                    for(int jj=jlow; jj<jhigh; jj++)
                    {
                        PossibleRelative temp =  getPossibleRelative(label,ii,jj);
                        if (temp.m_similarity > maxPossiRelative.m_similarity)
                        {
                            maxPossiRelative.m_similarity = temp.m_similarity;
                            maxPossiRelative.m_label = temp.m_label;
                        }
                    }
                }
                return maxPossiRelative;
            }


        }
    }
    return maxPossiRelative;
}

void ImageAnalyzer::mergeSmallCluster(int clusterSize){
    if (clusterSize < 1) return;

    //delete zero-size cluster
    m_labelMgr.deleteZerosPixelLabel();

    //merge small cluster
    for(int s=1; s<=clusterSize;s++){
        int label = m_labelMgr.getOneLabel(s);
        while (label > 0){
            PossibleRelative maxRelative = getMaxPossibleRelative(label,s);
            if (-1 != maxRelative.m_label && label != maxRelative.m_label) {
                label = mergeLabel(label,maxRelative.m_label);
                m_labelMgr.updateLabel(label,maxRelative.m_similarity);

            }
            label =  m_labelMgr.getOneLabel(s);

        }

    }

}


void ImageAnalyzer::Kmeans(const int K)
{
    if (K<2 || K > m_nCols*m_nRows/10)
    {
        printf("Error: parameter K for Kmeans has incorrect range.\n");
        return;
     }

    // fill mat result
    int labelFactor  = 1;
    if (K+1 <= m_labelMgr.getNumCluster()) {
       labelFactor = m_labelMgr.getNumCluster()/(K+1);
    }
    else{
       printf("Error: parameter K > possible automatic clustering number.\n");
       return;
    }

    m_originalMat.convertTo(m_originalMat,CV_32F);
    cv::Mat srcMat(m_nCols*m_nRows,1,CV_32F);
    long index = 0;
    for(int i =0; i<m_nRows; i++)
    {
        for(int j=0;j<m_nCols; j++)
        {
            srcMat.at<cv::Vec3f>(index++,0) =  m_originalMat.at<cv::Vec3f>(i,j);
        }
    }

    m_kmeansMat.create(cv::Size(m_nCols,m_nRows),m_Mat.type());
    cv::Mat clusterMatrix(m_nCols*m_nRows,1,CV_8UC1);

    double compactness = cv::kmeans (srcMat, K, clusterMatrix, cv::TermCriteria(cv::TermCriteria::EPS+cv::TermCriteria::COUNT, 10, 1.0),
                3, cv::KMEANS_PP_CENTERS);
    printf("Kmeans result compactness = %f\n",compactness);



    index = 0;
    for(int i =0; i<m_nRows; i++)
    {
        for(int j=0;j<m_nCols; j++)
        {
            int label = (1+clusterMatrix.at<int>(index++))*labelFactor; //kmeans cluster index from 0;
            m_kmeansMat.at<cv::Vec3b>(i,j) = m_labelMgr.getColor(label);

         }
    }

    cv::namedWindow( m_WndKmeans, cv::WINDOW_AUTOSIZE );
    cv::imshow( m_WndKmeans, m_kmeansMat);
    cv::waitKey(20);
    saveKmeansImage(K);

}








