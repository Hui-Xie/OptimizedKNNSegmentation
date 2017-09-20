
Development Environment:
1  OpenCV 3.1;
2  Code::Blocks 16.01
3  C++


Test:
1  in dir: Dropbox\ProjectSegment\bin\Release, you can directly run the program and test it.
2  current compiled version is Windows version;
3  in dir: bin\release dir, there are some test image file for test;

later, I will write a detailed document.


use example:
E:\VC Projects\ProjectSegment\bin\Release>projectsegment -F clown.jpg
==============Projection Similarity Segmentation============
Version: April 23th, 2016
Description: Automatically segment an RGB image with its projection similarity.
Usage: ./ProjectSegment [Option Option_parameter] ...
Example: ./ProjectSegment -F clown.jpg
Example: ./ProjectSegment -F clown.jpg -S 9 -T 0.95 -I 3 -P 0

Options:
-F image filename
-S minimum cluster size(1-100), default=9;
-T (0.8-1) similarity threshold, otherwise automatically compute a default value;
-I 0: (default) without inverting color; 1: invert color; 3: random clustering color;
-P 1: (default) median filter preprocess; 0: no preprocess;

Progress: Automatically MeanFiler Iterate 63 times.
Progress: the automatic similarity threshold = 0.9600
Hint: A big-size image needs more computation time. please wait....
Output: clown_threshold_96%.jpg
Output: clown_threshold_96%.csv
Please press any key to exit.