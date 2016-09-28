/*RealTime Face Recognization
  Project Mentor : Mr. Manoj wariya
  Himani Raghav
  Shivang Gupta
  Rohit Sharma
  Niraj Kumar
  Harshita Bunas	

*/
#define FALSE 0
#define TRUE 1 
#include <stdio.h>
#include <cstdio>
#include <sys/ioctl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/select.h>
#include <termios.h>
#include <stropts.h>
#include <vector>
#include <string>
#include "cv.h"
#include "cvaux.h"
#include "highgui.h"

using namespace std;

//#define RWRWRW (S_IRUSR|S_IWUSR|S_IRGRP|S_IWGRP|S_IROTH|S_IWOTH)
// Haar Cascade file, used for Face Detection.
const char *faceCascadeFilename = "haarcascade_frontalface_alt.xml";

int SAVE_EIGENFACE_IMAGES = 1;		
// Global variables
IplImage ** faceImgArr        = 0; // array of face images
CvMat    *  personNumTruthMat = 0; // array of person numbers
vector<string> personNames;			// array of person names (indexed by the person number)
int faceWidth = 120;	// Default dimensions for faces in the face recognition database
int faceHeight = 90;	
int nPersons                  = 0; // the number of people in the training set
int nTrainFaces               = 0; // the number of training images
int nEigens                   = 0; // the number of eigenvalues
IplImage * pAvgTrainImg       = 0; // the average image
IplImage ** eigenVectArr      = 0; // eigenvectors
CvMat * eigenValMat           = 0; // eigenvalues
CvMat * projectedTrainFaceMat = 0; // projected training faces

CvCapture* camera = 0;	// camera object


// Functions 
void learn(char *szFileTrain);
void doPCA();
void storeTrainingData();
int  loadTrainingData(CvMat ** pTrainPersonNumMat);
int  findNearestNeighbor(float * projectedTestFace);
int findNearestNeighbor(float * projectedTestFace, float *pConfidence);
int  loadFaceImgArray(char * filename);
void recognizeFileList(char *szFileTest);
void recognizeFromCam(void);
IplImage* getCameraFrame(void);
IplImage* convertImageToGreyscale(const IplImage *imageSrc);
IplImage* cropImage(const IplImage *img, const CvRect region);
IplImage* resizeImage(const IplImage *origImg, int newWidth, int newHeight);
IplImage* convertFloatImageToUcharImage(const IplImage *srcImg);
void saveFloatImage(const char *filename, const IplImage *srcImg);
CvRect detectFaceInImage(const IplImage *inputImg, const CvHaarClassifierCascade* cascade );
CvMat* retrainOnline(void);

int _kbhit() {
    static const int STDIN = 0;
    static bool initialized = false;

    if (! initialized) {
        // Use termios to turn off line buffering
        termios term;
        tcgetattr(STDIN, &term);
        term.c_lflag &= ~ICANON;
        tcsetattr(STDIN, TCSANOW, &term);
        setbuf(stdin, NULL);
        initialized = true;
    }

    int bytesWaiting;
    ioctl(STDIN, FIONREAD, &bytesWaiting);
    return bytesWaiting;
}


// main function
int main( int argc, char** argv )
{
	recognizeFromCam();
	return 0;
}

// Save all the eigenvectors as images
void storeEigenfaceImages()
{
	// Store the average image
	printf("Saving the image of the average face as 'out_averageImage.bmp'.\n");
	cvSaveImage("out_averageImage.bmp", pAvgTrainImg);
	// Create a large image made of many eigenface images.
	printf("Saving the %d eigenvector images as 'out_eigenfaces.bmp'\n", nEigens);
	if (nEigens > 0) {
		// Put all the eigenfaces adjacent to each other
		int COLUMNS = 8;	// Put upto 8 images on a row.
		int nCols = min(nEigens, COLUMNS);
		int nRows = 1 + (nEigens / COLUMNS);	// Put the rest on new rows.
		int w = eigenVectArr[0]->width;
		int h = eigenVectArr[0]->height;
		CvSize size;
		size = cvSize(nCols * w, nRows * h);
		IplImage *bigImg = cvCreateImage(size, IPL_DEPTH_8U, 1);	// 8-bit Greyscale UCHAR image
		for (int i=0; i<nEigens; i++) {
			// Get the eigenface image.
			IplImage *byteImg = convertFloatImageToUcharImage(eigenVectArr[i]); //convert each eigenface image to a normal 8-bit UCHAR image 													from a 32-bit float image
			// Paste it into the correct position.
			int x = w * (i % COLUMNS);
			int y = h * (i / COLUMNS);	
			CvRect ROI = cvRect(x, y, w, h);		//region of interest, offset
			cvSetImageROI(bigImg, ROI);
			cvCopyImage(byteImg, bigImg);
			cvResetImageROI(bigImg);
			cvReleaseImage(&byteImg);
		}
		cvSaveImage("out_eigenfaces.bmp", bigImg);
		cvReleaseImage(&bigImg);
	}
}

// train from the data in the given text file, and store the trained data into the file 'facedata.xml'.
void learn(char *szFileTrain)
{
	int i, offset;
	// load training data
	printf("Loading the training images in '%s'\n", szFileTrain);
	nTrainFaces = loadFaceImgArray(szFileTrain);
	printf("Got %d training images.\n", nTrainFaces);
	if( nTrainFaces < 2 )
	{
		fprintf(stderr,
		        "Need 2 or more training faces\n"
		        "Input file contains only %d\n", nTrainFaces);
		return;
	}

	// applying PCA algorithm
	doPCA();

	// project the training images onto the PCA subspace
	projectedTrainFaceMat = cvCreateMat( nTrainFaces, nEigens, CV_32FC1 );
	offset = projectedTrainFaceMat->step / sizeof(float);
	for(i=0; i<nTrainFaces; i++)
	{
		cvEigenDecomposite(
			faceImgArr[i],
			nEigens,
			eigenVectArr,
			0, 0,
			pAvgTrainImg,
			projectedTrainFaceMat->data.fl + i*offset);
	}

	// store the recognition data as an xml file
	storeTrainingData();

	// Save all the eigenvectors as images
	if (SAVE_EIGENFACE_IMAGES) {
		storeEigenfaceImages();
	}

}

// open the training data from the file 'facedata.xml'.
int loadTrainingData(CvMat ** pTrainPersonNumMat)
{
	CvFileStorage * fileStorage;
	int i;
	// create a file-storage
	fileStorage = cvOpenFileStorage( "facedata.xml", 0, CV_STORAGE_READ );
	if( !fileStorage ) {
		printf("Can't open training database file 'facedata.xml'.\n");
		return 0;
	}
	// load the person names
	personNames.clear();	// starts as empty.
	nPersons = cvReadIntByName( fileStorage, 0, "nPersons", 0 );
	if (nPersons == 0) {
		printf("no person found in the training database 'facedata.xml'.\n");
		return 0;
	}
	// Load each person's name.
	for (i=0; i<nPersons; i++) {
		string sPersonName;
		char varname[200];
		sprintf( varname, "personName_%d", (i+1) );
		sPersonName = cvReadStringByName(fileStorage, 0, varname );
		personNames.push_back( sPersonName );
	}

	// Load the image information
	nEigens = cvReadIntByName(fileStorage, 0, "nEigens", 0);
	nTrainFaces = cvReadIntByName(fileStorage, 0, "nTrainFaces", 0);
	*pTrainPersonNumMat = (CvMat *)cvReadByName(fileStorage, 0, "trainPersonNumMat", 0);
	eigenValMat  = (CvMat *)cvReadByName(fileStorage, 0, "eigenValMat", 0);
	projectedTrainFaceMat = (CvMat *)cvReadByName(fileStorage, 0, "projectedTrainFaceMat", 0);
	pAvgTrainImg = (IplImage *)cvReadByName(fileStorage, 0, "avgTrainImg", 0);
	eigenVectArr = (IplImage **)cvAlloc(nTrainFaces*sizeof(IplImage *));
	for(i=0; i<nEigens; i++)
	{
		char varname[200];
		sprintf( varname, "eigenVect_%d", i );
		eigenVectArr[i] = (IplImage *)cvReadByName(fileStorage, 0, varname, 0);
	}

	// release the file-storage pointer
	cvReleaseFileStorage( &fileStorage );

	printf("Training data loaded (%d training images of %d people):\n", nTrainFaces, nPersons);
	printf("People: ");
	if (nPersons > 0)
		printf("<%s>", personNames[0].c_str());
	for (i=1; i<nPersons; i++) {
		printf(", <%s>", personNames[i].c_str());
	}
	return 1;
}


// Save the training data to the file 'facedata.xml'.
void storeTrainingData()
{
	CvFileStorage * fileStorage;
	int i;

	// create a file-storage
	fileStorage = cvOpenFileStorage( "facedata.xml", 0, CV_STORAGE_WRITE );

	// store the person names
	cvWriteInt( fileStorage, "nPersons", nPersons );
	for (i=0; i<nPersons; i++) {
		char varname[200];
		sprintf( varname, "personName_%d", (i+1) );
		cvWriteString(fileStorage, varname, personNames[i].c_str(), 0);
	}

	// store all the image information
	cvWriteInt( fileStorage, "nEigens", nEigens );
	cvWriteInt( fileStorage, "nTrainFaces", nTrainFaces );
	cvWrite(fileStorage, "trainPersonNumMat", personNumTruthMat, cvAttrList(0,0));
	cvWrite(fileStorage, "eigenValMat", eigenValMat, cvAttrList(0,0));
	cvWrite(fileStorage, "projectedTrainFaceMat", projectedTrainFaceMat, cvAttrList(0,0));
	cvWrite(fileStorage, "avgTrainImg", pAvgTrainImg, cvAttrList(0,0));
	for(i=0; i<nEigens; i++)
	{
		char varname[200];
		sprintf( varname, "eigenVect_%d", i );
		cvWrite(fileStorage, varname, eigenVectArr[i], cvAttrList(0,0));
	}

	// release the file-storage
	cvReleaseFileStorage( &fileStorage );
}

// find the most likely person based on a detection
int findNearestNeighbor(float * projectedTestFace, float *pConfidence)
{
	double leastDistSq = DBL_MAX;
	int i, iTrain, iNearest = 0;
	for(iTrain=0; iTrain<nTrainFaces; iTrain++)
	{
		double distSq=0;
		for(i=0; i<nEigens; i++)
		{
			float d_i = projectedTestFace[i] - projectedTrainFaceMat->data.fl[iTrain*nEigens + i];
			distSq += d_i*d_i; // Euclidean distance
		}

		if(distSq < leastDistSq)
		{
			leastDistSq = distSq;
			iNearest = iTrain;
		}
	}

	// Return the confidence level based on the Euclidean distance
	*pConfidence = 1.0f - sqrt( leastDistSq / (float)(nTrainFaces * nEigens) ) / 255.0f;

	// Return the found index of detected image
	return iNearest;
}

// do the Principal Component Analysis, finding the average image
// and the eigenfaces that represent any image in the given database
void doPCA()
{
	int i;
	CvTermCriteria calcLimit;
	CvSize faceImgSize;
	// set the number of eigenvalues to be used
	nEigens = nTrainFaces-1;

	// allocate the eigenvector images
	faceImgSize.width  = faceImgArr[0]->width;
	faceImgSize.height = faceImgArr[0]->height;
	eigenVectArr = (IplImage**)cvAlloc(sizeof(IplImage*) * nEigens);
	for(i=0; i<nEigens; i++)
		eigenVectArr[i] = cvCreateImage(faceImgSize, IPL_DEPTH_32F, 1);

	// allocate the eigenvalue array
	eigenValMat = cvCreateMat( 1, nEigens, CV_32FC1 );

	// allocate the averaged image
	pAvgTrainImg = cvCreateImage(faceImgSize, IPL_DEPTH_32F, 1);

	// set the PCA termination criteria
	calcLimit = cvTermCriteria( CV_TERMCRIT_ITER, nEigens, 1);

	// compute average image, eigenvalues, and eigenvectors
	cvCalcEigenObjects(
		nTrainFaces,
		(void*)faceImgArr,
		(void*)eigenVectArr,
		CV_EIGOBJ_NO_CALLBACK,
		0,
		0,
		&calcLimit,
		pAvgTrainImg,
		eigenValMat->data.fl);

	cvNormalize(eigenValMat, eigenValMat, 1, 0, CV_L1, 0);
}

// read the names & image filenames of student from a text file and load all those images
int loadFaceImgArray(char * filename)
{
	FILE * imgListFile = 0;
	char imgFilename[512];
	int iFace, nFaces=0;
	int i;

	// open the input file train.txt
	if( !(imgListFile = fopen(filename, "r")) )
	{
		fprintf(stderr, "Can\'t open file %s\n", filename);
		return 0;
	}

	// count the number of faces
	while( fgets(imgFilename, 512, imgListFile) ) ++nFaces;
	rewind(imgListFile);

	// allocate the face-image array and person number matrix
	faceImgArr        = (IplImage **)cvAlloc( nFaces*sizeof(IplImage *) );
	personNumTruthMat = cvCreateMat( 1, nFaces, CV_32SC1 );

	personNames.clear();	// start as empty.
	nPersons = 0;

	// store the face images in an array
	for(iFace=0; iFace<nFaces; iFace++)
	{
		char personName[256];
		string sPersonName;
		int personNumber;

		// read person number starting from 1, their name and the image filename.
		fscanf(imgListFile, "%d %s %s", &personNumber, personName, imgFilename);
		sPersonName = personName;
		printf("Got %d: %d, <%s>, <%s>.\n", iFace, personNumber, personName, imgFilename);

		// check if a new person is being loaded.
		if (personNumber > nPersons) {
			// allocate memory for the extra person
			for (i=nPersons; i < personNumber; i++) {
				personNames.push_back( sPersonName );
			}
			nPersons = personNumber;
			printf("Got new person <%s> -> nPersons = %d [%d]\n", sPersonName.c_str(), nPersons, personNames.size());
		}

		// keep the data
		personNumTruthMat->data.i[iFace] = personNumber;

		// load the face image
		faceImgArr[iFace] = cvLoadImage(imgFilename, CV_LOAD_IMAGE_GRAYSCALE);

		if( !faceImgArr[iFace] )
		{
			fprintf(stderr, "Can\'t load image from %s\n", imgFilename);
			return 0;
		}
	}
	fclose(imgListFile);
	printf("Data loaded from '%s': (%d images of %d people).\n", filename, nFaces, nPersons);
	printf("People: ");
	if (nPersons > 0)
		printf("<%s>", personNames[0].c_str());
	for (i=1; i<nPersons; i++) {
		printf(", <%s>", personNames[i].c_str());
	}
	return nFaces;
}


// recognize the face in each of the test images given, and compare the results with the original
void recognizeFileList(char *szFileTest)
{
	int i, nTestFaces  = 0;         // the number of test images
	CvMat * trainPersonNumMat = 0;  // the person numbers during training
	float * projectedTestFace = 0;
	char *answer;
	int nCorrect = 0;
	int nWrong = 0;
	double timeFaceRecognizeStart;
	double tallyFaceRecognizeTime;
	float confidence;

	// load test images
	nTestFaces = loadFaceImgArray(szFileTest);
	printf("%d test faces loaded\n", nTestFaces);

	// load the saved training data
	if( !loadTrainingData( &trainPersonNumMat ) ) return;

	// project the test images onto the PCA subspace
	projectedTestFace = (float *)cvAlloc( nEigens*sizeof(float) );
	timeFaceRecognizeStart = (double)cvGetTickCount();	// record the timing.
	for(i=0; i<nTestFaces; i++)
	{
		int iNearest, nearest, truth;

		// project the test image onto the PCA subspace
		cvEigenDecomposite(
			faceImgArr[i],
			nEigens,
			eigenVectArr,
			0, 0,
			pAvgTrainImg,
			projectedTestFace);

		iNearest = findNearestNeighbor(projectedTestFace, &confidence);
		truth    = personNumTruthMat->data.i[i];
		nearest  = trainPersonNumMat->data.i[iNearest];

		if (nearest == truth) {
			answer = "Correct";
			nCorrect++;
		}
		else {
			answer = "WRONG!";
			nWrong++;
		}
		printf("nearest = %d, Truth = %d (%s). Confidence = %f\n", nearest, truth, answer, confidence);
	}
	tallyFaceRecognizeTime = (double)cvGetTickCount() - timeFaceRecognizeStart;
	if (nCorrect+nWrong > 0) {
		printf("TOTAL ACCURACY: %d%% out of %d tests.\n", nCorrect * 100/(nCorrect+nWrong), (nCorrect+nWrong));
		printf("TOTAL TIME: %.1fms average.\n", tallyFaceRecognizeTime/((double)cvGetTickFrequency() * 1000.0 * (nCorrect+nWrong) ) );
	}

}

// grab the next camera frame, waits until the next frame is ready,
// automatically initialize the camera on the first frame.
IplImage* getCameraFrame(void)
{
	IplImage *frame;

	// if the camera hasn't been initialized, then open it.
	if (!camera) {
		printf("Acessing the camera ...\n");
		camera = cvCaptureFromCAM( -1 );
		if (!camera) {
			printf("ERROR in getCameraFrame(): Couldn't access the camera.\n");
			exit(1);
		}
		// try to set the camera resolution
		cvSetCaptureProperty( camera, CV_CAP_PROP_FRAME_WIDTH, 320 );
		cvSetCaptureProperty( camera, CV_CAP_PROP_FRAME_HEIGHT, 240 );
		// wait a little so that camera can auto-adjust itself
		sleep(10);	// (in seconds)
		frame = cvQueryFrame( camera );	// get the first frame, to make sure the camera is initialized.
		if (frame) {
			printf("Got a camera using a resolution of %dx%d.\n", (int)cvGetCaptureProperty( camera, CV_CAP_PROP_FRAME_WIDTH), (int)cvGetCaptureProperty( camera, CV_CAP_PROP_FRAME_HEIGHT) );
		}
	}

	frame = cvQueryFrame( camera );
	if (!frame) {
		fprintf(stderr, "ERROR in recognizeFromCam(): Could not access the camera\n");
		exit(1);
	}
	return frame;
}

// return a new image that is always greyscale
IplImage* convertImageToGreyscale(const IplImage *imageSrc)
{
	IplImage *imageGrey;
	// either convert the image to greyscale, or make a copy of the existing greyscale image.
	if (imageSrc->nChannels == 3) {
		imageGrey = cvCreateImage( cvGetSize(imageSrc), IPL_DEPTH_8U, 1 );
		cvCvtColor( imageSrc, imageGrey, CV_BGR2GRAY );
	}
	else {
		imageGrey = cvCloneImage(imageSrc);
	}
	return imageGrey;
}

// creates a new image copy that is of a desired size.
IplImage* resizeImage(const IplImage *origImg, int newWidth, int newHeight)
{
	IplImage *outImg = 0;
	int origWidth;
	int origHeight;
	if (origImg) {
		origWidth = origImg->width;
		origHeight = origImg->height;
	}
	if (newWidth <= 0 || newHeight <= 0 || origImg == 0 || origWidth <= 0 || origHeight <= 0) {
		printf("ERROR in resizeImage: Bad desired image size of %dx%d\n.", newWidth, newHeight);
		exit(1);
	}

	// Scale the image to the new dimensions, even if the aspect ratio will be changed.
	outImg = cvCreateImage(cvSize(newWidth, newHeight), origImg->depth, origImg->nChannels);
	if (newWidth > origImg->width && newHeight > origImg->height) {
		// make the image larger
		cvResetImageROI((IplImage*)origImg);
		cvResize(origImg, outImg, CV_INTER_LINEAR);	// CV_INTER_LINEAR is good for enlarging
	}
	else {
		// make the image smaller
		cvResetImageROI((IplImage*)origImg);
		cvResize(origImg, outImg, CV_INTER_AREA);	// CV_INTER_AREA is good for shrinking / decimation, but bad at enlarging.
	}
	return outImg;
}

// returns a new image that is a cropped version of the original image. 
IplImage* cropImage(const IplImage *img, const CvRect region)
{
	IplImage *imageTmp;
	IplImage *imageRGB;
	CvSize size;
	size.height = img->height;
	size.width = img->width;

	if (img->depth != IPL_DEPTH_8U) {
		printf("ERROR in cropImage: Unknown image depth of %d given in cropImage() instead of 8 bits per pixel.\n", img->depth);
		exit(1);
	}

	// first create a new (color or greyscale) IPL Image and copy contents of img into it.
	imageTmp = cvCreateImage(size, IPL_DEPTH_8U, img->nChannels);
	cvCopy(img, imageTmp, NULL);

	// create a new image of the detected region
	// set region of interest to that surrounding the face
	cvSetImageROI(imageTmp, region);
	// copy region of interest into a new iplImage (in imageRGB format) and return it
	size.width = region.width;
	size.height = region.height;
	imageRGB = cvCreateImage(size, IPL_DEPTH_8U, img->nChannels);
	cvCopy(imageTmp, imageRGB, NULL);	// copy the region only

    cvReleaseImage( &imageTmp );
	return imageRGB;		
}

// get an 8-bit equivalent of the 32-bit Float image.
IplImage* convertFloatImageToUcharImage(const IplImage *srcImg)
{
	IplImage *dstImg = 0;
	if ((srcImg) && (srcImg->width > 0 && srcImg->height > 0)) {

		// spread the 32bit floating point pixels to fit within 8bit pixel range.
		double minVal, maxVal;
		cvMinMaxLoc(srcImg, &minVal, &maxVal);
		// check NaN and extreme values, as DFT seems to give some NaN results.
		if (cvIsNaN(minVal) || minVal < -1e30)
			minVal = -1e30;
		if (cvIsNaN(maxVal) || maxVal > 1e30)
			maxVal = 1e30;
		if (maxVal-minVal == 0.0f)
			maxVal = minVal + 0.001;	// remove divide by zero errors.

		// convert the format
		dstImg = cvCreateImage(cvSize(srcImg->width, srcImg->height), 8, 1);
		cvConvertScale(srcImg, dstImg, 255.0 / (maxVal - minVal), - minVal * 255.0 / (maxVal-minVal));
	}
	return dstImg;
}

// store a greyscale floating-point CvMat image into a BMP/JPG image,
void saveFloatImage(const char *filename, const IplImage *srcImg)
{
	IplImage *byteImg = convertFloatImageToUcharImage(srcImg);
	cvSaveImage(filename, byteImg);        //cvSaveImage() can only handle 8bit images (not 32bit float images)
	cvReleaseImage(&byteImg);
}

// perform face detection on the input image, using the given Haar cascade classifier.
// returns a rectangle for the detected region in the given image.
CvRect detectFaceInImage(const IplImage *inputImg, const CvHaarClassifierCascade* cascade )
{
	const CvSize minFeatureSize = cvSize(20, 20);
	const int flags = CV_HAAR_FIND_BIGGEST_OBJECT | CV_HAAR_DO_ROUGH_SEARCH;	// only search for 1 face at a time
	const float search_scale_factor = 1.1f;
	IplImage *detectImg;
	IplImage *greyImg = 0;
	CvMemStorage* storage;
	CvRect rc;
	double t;
	CvSeq* rects;
	int i;

	storage = cvCreateMemStorage(0);
	cvClearMemStorage( storage );

	// if the image is color, use a greyscale copy of the image.
	detectImg = (IplImage*)inputImg;	
	if (inputImg->nChannels > 1) 
	{
		greyImg = cvCreateImage(cvSize(inputImg->width, inputImg->height), IPL_DEPTH_8U, 1 );
		cvCvtColor( inputImg, greyImg, CV_BGR2GRAY );
		detectImg = greyImg;	// use the greyscale image as the input.
	}

	// detect all the faces.
	t = (double)cvGetTickCount();
	rects = cvHaarDetectObjects( detectImg, (CvHaarClassifierCascade*)cascade, storage,
				search_scale_factor, 3, flags, minFeatureSize );
	t = (double)cvGetTickCount() - t;

	printf("[Face Detection took %d ms and found %d objects]\n", cvRound( t/((double)cvGetTickFrequency()*1000.0) ), rects->total );

	// get the first detected face, largest one
	if (rects->total > 0) {
        rc = *(CvRect*)cvGetSeqElem( rects, 0 );
    }
	else
		rc = cvRect(-1,-1,-1,-1);	// couldn't find the face.

	if (greyImg)
		cvReleaseImage( &greyImg );
	cvReleaseMemStorage( &storage );

	return rc;	// return the largest face found, or (-1,-1,-1,-1).
}

// re-train the new face database
CvMat* retrainOnline(void)
{
	CvMat *trainPersonNumMat;
	int i;

	// free & re-initialize the global variables.
	if (faceImgArr) {
		for (i=0; i<nTrainFaces; i++) {
			if (faceImgArr[i])
				cvReleaseImage( &faceImgArr[i] );
		}
	}
	cvFree( &faceImgArr ); // array of face images
	cvFree( &personNumTruthMat ); // array of person numbers
	personNames.clear();			// array of person names (indexed by the person number). 
	nPersons = 0; // the number of people in the training set.
	nTrainFaces = 0; // the number of training images
	nEigens = 0; // the number of eigenvalues
	cvReleaseImage( &pAvgTrainImg ); // the average image
	for (i=0; i<nTrainFaces; i++) {
		if (eigenVectArr[i])
			cvReleaseImage( &eigenVectArr[i] );
	}
	cvFree( &eigenVectArr ); // eigenvectors
	cvFree( &eigenValMat ); // eigenvalues
	cvFree( &projectedTrainFaceMat ); // projected training faces

	// retrain from the image data in the files
	printf("Retraining with the new person ...\n");
	learn("train.txt");
	printf("Done retraining.\n");

	// load the previously saved training data
	if( !loadTrainingData( &trainPersonNumMat ) ) {
		printf("ERROR in recognizeFromCam(): Couldn't load the training data!\n");
		exit(1);
	}

	return trainPersonNumMat;
}

// continuously recognize the person in the camera
void recognizeFromCam(void)
{
	int i;
	CvMat * trainPersonNumMat;  // the person numbers during training
	float * projectedTestFace;
	double timeFaceRecognizeStart;
	double tallyFaceRecognizeTime;
	CvHaarClassifierCascade* faceCascade;
	char cstr[256];
	char pstr[256];
	bool saveNextFaces = FALSE;
	char newPersonName[256];
	int newPersonFaces;

	trainPersonNumMat = 0;  // the person numbers during training
	projectedTestFace = 0;
	saveNextFaces = FALSE;
	newPersonFaces = 0;

	printf("Recognizing person in the camera ...\n");

	// load the previously saved training data
	if( loadTrainingData( &trainPersonNumMat ) ) {
		faceWidth = pAvgTrainImg->width;
		faceHeight = pAvgTrainImg->height;
	}

	// project the test images onto the PCA subspace
	projectedTestFace = (float *)cvAlloc( nEigens*sizeof(float) );

	// create a GUI window for the user to see the camera image
	cvNamedWindow("Input", CV_WINDOW_AUTOSIZE);

	// create a data folder if not exist for storing
	mkdir("data", 0777);
	mkdir("unknown",0777);

	// load the HaarCascade classifier for face detection
	faceCascade = (CvHaarClassifierCascade*)cvLoad(faceCascadeFilename, 0, 0, 0 );
	if( !faceCascade ) {
		printf("ERROR in recognizeFromCam(): Could not load Haar cascade Face detection classifier in '%s'.\n", faceCascadeFilename);
		exit(1);
	}
	timeFaceRecognizeStart = (double)cvGetTickCount();	// record the timing.

	while (1)
	{
		int iNearest, nearest, truth;
		IplImage *camImg;
		IplImage *greyImg;
		IplImage *faceImg;
		IplImage *sizedImg;
		IplImage *equalizedImg;
//		IplImage *edgedetectImg;
		IplImage *processedFaceImg;
		IplImage *img_b;
		CvRect faceRect;
		IplImage *shownImg;
		int keyPressed = 0;
		FILE *trainFile;
		float confidence;
		
		int N = 7;
/*		// Edge Detection Variables
		int aperature_size = N;
		double lowThresh = 20;
		double highThresh = 40;
*/
		// check keyboard input
		if (_kbhit())
			keyPressed = getchar();

		switch (keyPressed) {
			case 'n':	// add a new person to the training set.


				printf("Enter your name: ");
				strcpy(newPersonName, "newPerson");
				gets(newPersonName);
				printf("Collecting all images until you hit 't', to start Training the images as '%s' ...\n", newPersonName);
				newPersonFaces = 0;	// restart training a new person
				saveNextFaces = TRUE;
				break;
			case 't':	// start training
				saveNextFaces = FALSE;	// stop saving next faces.
				// store the saved data into the training file.
				printf("Storing the training data for new person '%s'.\n", newPersonName);
				// append the new person to the end of the training data.
				trainFile = fopen("train.txt", "a");
				for (i=0; i<newPersonFaces; i++) {
					sprintf(cstr, "data/%d_%s%d.jpg", nPersons+1, newPersonName, i+1);
					fprintf(trainFile, "%d %s %s\n", nPersons+1, newPersonName, cstr);
				}
				fclose(trainFile);
				// re-initialize
				projectedTestFace = 0;
				saveNextFaces = FALSE;
				newPersonFaces = 0;
				cvFree( &trainPersonNumMat );	// free the previous data before getting new data
				trainPersonNumMat = retrainOnline();
				cvFree(&projectedTestFace);	// free the previous data before getting new data
				projectedTestFace = (float *)cvAlloc( nEigens*sizeof(float) );

				printf("Recognizing person in the camera ...\n");
				continue;	// begin with the next frame.
				break;
		}

		// get the camera frame
		camImg = getCameraFrame();
		if (!camImg) {
			printf("ERROR in recognizeFromCam(): Bad input image!\n");
			exit(1);
		}
		// make sure the image is greyscale, as the Eigenfaces is only done on greyscale image.
		greyImg = convertImageToGreyscale(camImg);

		// perform face detection on the input image
		faceRect = detectFaceInImage(greyImg, faceCascade );
		// make sure a valid face was detected.
		if (faceRect.width > 0) {
			faceImg = cropImage(greyImg, faceRect);          // get the detected face image.
			// setting image to the same dimensions as the training image
			sizedImg = resizeImage(faceImg, faceWidth, faceHeight);
			// give the image a standard brightness and contrast, in case it was too dark or low contrast.
			equalizedImg = cvCreateImage(cvGetSize(sizedImg), 8, 1);	// create an empty greyscale image
			cvEqualizeHist(sizedImg, equalizedImg);
			
			img_b = cvCreateImage(cvGetSize(equalizedImg), 8, 1);
			// Add convolution boarders
			CvPoint offset = cvPoint(0, 0);
			cvCopyMakeBorder(equalizedImg, img_b, offset, IPL_BORDER_REPLICATE, cvScalarAll(0));

			
//			edgedetectImg = cvCreateImage(cvGetSize(equalizedImg), IPL_DEPTH_8U, 1);
//			cvCanny( img_b, edgedetectImg, lowThresh*N*N, highThresh*N*N, aperature_size );
//			processedFaceImg = edgedetectImg;
			processedFaceImg = img_b;
//			processedFaceImg = equalizedImg;
			if (!processedFaceImg) {
				printf("ERROR in recognizeFromCam(): Don't have input image!\n");
				exit(1);
			}

			// If the face database has been loaded, then try to recognize the person currently detected.
			if (nEigens > 0) {
				// project the test image onto the PCA subspace
				cvEigenDecomposite(
					processedFaceImg,
					nEigens,
					eigenVectArr,
					0, 0,
					pAvgTrainImg,
					projectedTestFace);

				// recognize which person is most likely
				iNearest = findNearestNeighbor(projectedTestFace, &confidence);
				nearest  = trainPersonNumMat->data.i[iNearest];
				if(confidence<.90)
				{
				printf("Most likely person in came: Unknown \n" );
				sprintf(pstr, "unknown/%d_%s%d.jpg", nPersons+1, "unknown", newPersonFaces+1);
				printf("Storing the current face of '%s' into image '%s'.\n", "unknown", pstr);
				cvSaveImage(pstr, processedFaceImg);
				newPersonFaces++;
				}
				else
				printf("Recognised '%s' .... \n", personNames[nearest-1].c_str());

			}//end  of if nEigens

			// save the processed face to the training database
			if (saveNextFaces) {
				sprintf(cstr, "data/%d_%s%d.jpg", nPersons+1, newPersonName, newPersonFaces+1);
				printf("Storing the current face of '%s' into image '%s'.\n", newPersonName, cstr);
				cvSaveImage(cstr, processedFaceImg);
				newPersonFaces++;
			}

			// free the resources used for this frame
			cvReleaseImage( &greyImg );
			cvReleaseImage( &faceImg );
			cvReleaseImage( &sizedImg );
			cvReleaseImage( &equalizedImg );
			cvReleaseImage( &img_b );
//			cvReleaseImage( &edgedetectImg );
		}

		// show the data on the screen
		shownImg = cvCloneImage(camImg);
		if (faceRect.width > 0) {	// check if a face was detected
			// show the detected face region
			cvRectangle(shownImg, cvPoint(faceRect.x, faceRect.y), cvPoint(faceRect.x + faceRect.width-1, faceRect.y + faceRect.height-1), CV_RGB(0,255,0), 1, 8, 0);

			if (nEigens > 0) {	// check if the face database is loaded and a person was recognized
				CvFont font;
				cvInitFont(&font,CV_FONT_HERSHEY_PLAIN, 1.0, 1.0, 0,1,CV_AA);
				CvScalar textColor = CV_RGB(0,255,255);	// light blue text colour
				char text[256];
				if(confidence<.70)
				{
				snprintf(text, sizeof(text)-1, "Unknown");
				cvPutText(shownImg, text, cvPoint(faceRect.x, faceRect.y + faceRect.height + 15), &font, textColor);
				}
				else			// show the name of the recognized person, overlayed on the image below their face
				{
				snprintf(text, sizeof(text)-1, "Name: '%s'", personNames[nearest-1].c_str());
				cvPutText(shownImg, text, cvPoint(faceRect.x, faceRect.y + faceRect.height + 15), &font, textColor);
				//snprintf(text, sizeof(text)-1, "Confidence: %f", confidence);
				//cvPutText(shownImg, text, cvPoint(faceRect.x, faceRect.y + faceRect.height + 30), &font, textColor);
				}
			}
		}

		// display the image.
		cvShowImage("Input", shownImg);

		// give some time for OpenCV to draw the GUI and check if the user has pressed something in the GUI window.
		keyPressed = cvWaitKey(200);
		cvReleaseImage( &shownImg );
	}//end of while(1)
	tallyFaceRecognizeTime = (double)cvGetTickCount() - timeFaceRecognizeStart;

	// free the camera and resources used.
	cvReleaseCapture( &camera );
	cvReleaseHaarClassifierCascade( &faceCascade );
} // end of function recognizeFromCam()
