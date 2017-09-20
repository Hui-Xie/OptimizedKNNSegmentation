#include <opencv2/opencv.hpp>
#include <iostream>
#include "ImageAnalyzer.h"


void printUsage()
{
    std::cout << "==============Projection Similarity Segmentation============\n"
         "Version: April 30th, 2016\n"
         "Description: Automatically segment an RGB image with its projection similarity.\n"
         "Usage: ./ProjectSegment [Option Option_parameter] ...\n"
         "Example: ./ProjectSegment -F clown.jpg\n"
         "Example: ./ProjectSegment -F clown.jpg -S 9 -T 0.95 -I 3 -P 0\n\n"
         "Options:\n"
         "-F image filename\n"
         "-S minimum cluster size(1-100), default=9;\n"
         "-T (0.8-1) similarity threshold, otherwise automatically compute a default value;\n"
         "-I 0: (default) without inverting color; 1: invert color; 3: random clustering color;\n"
         "-P 1: (default) median filter preprocess; 0: no preprocess;\n"
         "-K num: specify kmeans's K; e.g -K 30;default use project clustering number;\n"
         "=====================================================================\n"
         <<std::endl;
}

int main( int argc, char** argv )
{
    printUsage();

    ImageAnalyzer analyzer;

    if (1 != argc%2 || argc < 3) {
         printf("Error: input parameter error. \n");
         return -1;
    }
    //Check input arguments
    for(int i = 1; i<argc-1; i= i+2 ){
        std::string argLeader(*(argv+i));

        if ( 0 == argLeader.compare("-F") || 0 == argLeader.compare("-f")){
            analyzer.m_originalFileName = std::string (*(argv+i+1));
        }
        else if ( 0 == argLeader.compare("-I") || 0 == argLeader.compare("-i")){
            int invertColor = atoi (*(argv+i+1));
            analyzer.setInvertColor(invertColor);
        }
        else if ( 0 == argLeader.compare("-T") || 0 == argLeader.compare("-t")){
            analyzer.m_similarThreshold = (float) atof (*(argv+i+1));
        }
        else if ( 0 == argLeader.compare("-P") || 0 == argLeader.compare("-p")){
            int temp = atoi (*(argv+i+1));
            if (0 == temp ) analyzer.m_preProcess = false;
            else analyzer.m_preProcess = true;
        }
        else if ( 0 == argLeader.compare("-S") || 0 == argLeader.compare("-s")){
            analyzer.m_minClusterSize = atoi (*(argv+i+1));
         }
        else if ( 0 == argLeader.compare("-K") || 0 == argLeader.compare("-k")){
            analyzer.m_K = atoi (*(argv+i+1));
         }
        else {
            printf("Error: input parameter error. \n");
            return -2;
        }

    }


    if ((analyzer.m_similarThreshold <=0 && -1 != analyzer.m_similarThreshold)
         || analyzer.m_similarThreshold >1)
   {
        printf("Error: Similarity parameter error. Quit.\n");
        return -4;
    }

    if (0 != analyzer.readImageInialize()) return -3;
    if (analyzer.m_preProcess) analyzer.preprocessMat();

    analyzer.clusterMat();

    printf("Please press any key to exit.\n");
    getchar();
    return 0;
}
