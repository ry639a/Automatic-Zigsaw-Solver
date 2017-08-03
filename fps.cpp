#include<sys/types.h>
#include<sys/stat.h>
#include<dirent.h>
#include<string>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

using namespace std;
using namespace cv;

Mat canvas = Mat::zeros(Size(3000,3000), CV_8UC3 );
Mat prevcanvas,original;
int lastrow=0,lastcol=300;
int prevlastrow=0,prevlastcol=0;
Mat input_Array[100][4];
int labels[100][2];
int total=0,pos=0;
RNG rng(12345);
float angle=0;
vector< Point2f > corners[100];
char edge_type[100][4];
int cornerpieces[5][2],edgepiece[50][2],innerpieces[50][2];
int cornerindex=0,edgepieceindex=0,innerpieceindex=0;
int prevtype =0;
int shapematch =0;
int colormatch =0;
int attachedcorners=0,temptype=0;
int previndex =0;
int canvasindex=1;
Point2f midpoints[100][4];
Rect getBoundRect(int);
void removeChip()
{
	cout<<"entred remove code";
	prevcanvas.copyTo(canvas);
	lastrow = prevlastrow;
	lastcol = prevlastcol;
	imwrite("canvas.jpg",canvas);	
}
void attachToCanvas(int index)
{
	//Attaches each image to canvas
	//Mat half;
	//scaling both images to be of same size
	cout<<"entered attach";
	Mat image = input_Array[index][0];
	canvas.copyTo(prevcanvas);
	prevlastrow = lastrow;
	prevlastcol = lastcol;
	resize(image,image,Size(300,300));
	if(prevtype ==1 || prevtype ==2)
	{	
		Mat half(canvas,Rect(lastcol,lastrow,300,300));
		image.copyTo(half);
		lastcol+=300;
	}
	else 
	{
		lastcol = 0;
		lastrow+=300;
		Mat half(canvas,Rect(lastcol,lastrow,300,300));
		image.copyTo(half);
		lastcol+=300;
	}
	string kpfname = "canvas";
	stringstream ss;
	ss<<canvasindex;
	kpfname = kpfname+ss.str()+".jpg";
	ss.clear();
	imwrite(kpfname,canvas);
	canvasindex++;
}
void assignLabels(int index)
{
	//Detects shape and assign labels to each piece
	Mat image;
	input_Array[index][1].copyTo(image);
	//labels[total][0] = 1;
	int nlines=0;
	
	int mid=0,inside=0,outside=0;
	int ncorners = corners[index].size();
	int mindpointindex=1;
	

	if(ncorners>1)
	{
		midpoints[index][0].x = (corners[index].at(0).x+corners[index].at(1).x)*0.5;
		midpoints[index][0].y = (corners[index].at(0).y+corners[index].at(1).y)*0.5;
		
		mid = image.at<uchar>(midpoints[index][0].y, midpoints[index][0].x);
		inside = image.at<uchar>(midpoints[index][0].y+10, midpoints[index][0].x);
		outside = image.at<uchar>(midpoints[index][0].y-10, midpoints[index][0].x);
		if((outside==255) && (inside==255))
		{
			edge_type[index][0]='T';
		}
		else if((outside==0) && (inside==0))
		{
			edge_type[index][0]='H';
		}
		else{
			edge_type[index][0]='L';
			nlines++;
		}
	
		if(ncorners>2)
		{
			midpoints[index][1].x = (corners[index].at(1).x+corners[index].at(2).x)*0.5;
			midpoints[index][1].y = (corners[index].at(1).y+corners[index].at(2).y)*0.5;
		
			mid = image.at<uchar>(midpoints[index][1].y, midpoints[index][1].x);
			inside = image.at<uchar>(midpoints[index][1].y, midpoints[index][1].x-10);
			outside = image.at<uchar>(midpoints[index][1].y, midpoints[index][1].x+10);
			if((outside==255) && (inside==255))
			{
				edge_type[index][1]='T';
			}
			else if((outside==0) && (inside==0))
			{
				edge_type[index][1]='H';
			}
			else{
				edge_type[index][1]='L';
				nlines++;
			}
			if(ncorners>3)
			{
				midpoints[index][2].x = (corners[index].at(2).x+corners[index].at(3).x)*0.5;
				midpoints[index][2].y = (corners[index].at(2).y+corners[index].at(3).y)*0.5;
				
				mid = image.at<uchar>(midpoints[index][2].y, midpoints[index][2].x);
				inside = image.at<uchar>(midpoints[index][2].y-10, midpoints[index][2].x);
				outside = image.at<uchar>(midpoints[index][2].y+10, midpoints[index][2].x);
				if((outside==255) && (inside==255))
				{
					edge_type[index][2]='T';
				}
				else if((outside==0) && (inside==0))
				{
					edge_type[index][2]='H';
				}
				else{
					edge_type[index][2]='L';
					nlines++;
				}
				
				midpoints[index][3].x = (corners[index].at(3).x+corners[index].at(0).x)*0.5;
				midpoints[index][3].y = (corners[index].at(3).y+corners[index].at(0).y)*0.5;
				
				mid = image.at<uchar>(midpoints[index][3].y, midpoints[index][3].x);
				inside = image.at<uchar>(midpoints[index][3].y, midpoints[index][3].x+10);
				outside = image.at<uchar>(midpoints[index][3].y, midpoints[index][3].x-10);
				if((outside==255) && (inside==255))
				{
					edge_type[index][3]='T';
				}
				else if((outside==0) && (inside==0))
				{
					edge_type[index][3]='H';
				}
				else{
					edge_type[index][3]='L';
					nlines++;
				}
			}
		}
	cout<<"index:"<<index;
	circle( image, midpoints[index][0], 3, cv::Scalar(120,120,120), -1 );
	
	imwrite("midpoint.jpg",image);
	if(nlines==2)
	{
		cornerpieces[cornerindex][0]=index;
		cornerpieces[cornerindex][1]=0;//notused
		cornerindex++;
		cout<<"\n cornerindex:"<<cornerindex;
	}
	else if(nlines==1)
	{
		cout<<"index:"<<index;
		edgepiece[edgepieceindex][0] = index;
		edgepiece[edgepieceindex][1]=0;
		edgepieceindex++;
		cout<<"\n edgepieceindex:"<<edgepiece[edgepieceindex-1][0];
	}
	else
	{
		innerpieces[innerpieceindex][0]=index;
		innerpieces[innerpieceindex][1]=0;
		innerpieceindex++;
		cout<<"\n innerpieceindex:"<<innerpieceindex;
	}
	cout<<edge_type[index];
}
}
void fixOrientation(int index,float angle,int upd)
{
	//Align each piece 
	//If upd flag is sent, it updates labels in edge_type
	Mat image = input_Array[index][0];
	Mat rotated;
	Point2f oldcenter(image.cols/2., image.rows/2.); 
		
	Mat r = getRotationMatrix2D(oldcenter,angle, 1.0);
	Rect bbox = RotatedRect(oldcenter,image.size(), angle).boundingRect();
	
	r.at<double>(0,2) += bbox.width/2.0 - oldcenter.x;
	r.at<double>(1,2) += bbox.height/2.0 - oldcenter.y;
	
	warpAffine(image,input_Array[index][0], r, bbox.size());
	warpAffine(input_Array[index][1], input_Array[index][1], r, bbox.size());
	warpAffine(input_Array[index][2], input_Array[index][2], r, bbox.size());
	warpAffine(input_Array[index][3], input_Array[index][3], r, bbox.size());
	
	
	int a = angle;
	if(upd)
	{
		//update edgematrix
		int nrotations = a/90;
		while(nrotations>0)
		{
			char temp = edge_type[index][3];
			edge_type[index][3] = edge_type[index][2];
			edge_type[index][2] = edge_type[index][1];
			edge_type[index][1] = edge_type[index][0];
			edge_type[index][0] = temp;
			
			nrotations--;
		}
	}
	
	/*namedWindow("color.jpg",CV_WINDOW_NORMAL);
	imshow("color.jpg",input_Array[index][0]);
	namedWindow("thres.jpg",CV_WINDOW_NORMAL);
	imshow("thres.jpg",input_Array[index][1]);
	namedWindow("approx.jpg",CV_WINDOW_NORMAL);
	imshow("approx.jpg",input_Array[index][3]);
	namedWindow("contour.jpg",CV_WINDOW_NORMAL);
	imshow("contour.jpg",input_Array[index][2]);
	waitKey(0);*/
}

void extractPieces(Mat image)
{
	//Obtains contours, approximates polygons, and draws lines using hough transform and calls fixOrientation
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	Mat image2;
	original.copyTo(image2);
	Rect boundRect;
	Mat threshImage=Mat::zeros(image.size(), CV_8UC1);
	Mat drawing = Mat::zeros(image.size(), CV_8UC1 );
	Mat approx = Mat::zeros(image.size(), CV_8UC1 );
	
	
	findContours( image, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE,Point(0,0) );

	Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
	for( int i = 0; i< contours.size(); i++ )
	{
		drawContours( drawing, contours, i, color, 1, 1, hierarchy, 0, Point() );
		
		drawContours( image2, contours, i, color, 1, 1, hierarchy, 0, Point() );
	}
	imwrite("contour.jpg",image2);
	imwrite("drwaing.jpg",drawing);
	cout<<"\n contours.size():"<<contours.size();
	double maxArea=0.0;
	vector<vector<Point> > contours_poly( contours.size() );
	for( int i = 0; i < contours.size(); i++ )
	{ 
		double area = contourArea(contours[i]);
		//
		if(area>=40000 && area<=360000)
		{
			maxArea = area;
			drawContours( threshImage, contours, i, Scalar(255,255,255), CV_FILLED, 1, hierarchy, 0, Point() );
			medianBlur(threshImage,threshImage,3);
			approxPolyDP( Mat(contours[i]), contours_poly[i], 10, true );
			Mat approxcolor;
			original.copyTo(approxcolor);
			drawContours( approx, contours_poly, i, Scalar(255,255,255), 1, 1, hierarchy, 0, Point() );
			drawContours( approxcolor, contours_poly, i, Scalar(255,255,255), 1, 1, hierarchy, 0, Point() );
			boundRect = boundingRect(Mat(contours_poly[i]));
			//namedWindow("approxcolor.jpg", CV_WINDOW_NORMAL);
			//imshow("approxcolor.jpg", approxcolor(boundRect));
			//waitKey(0);
			//boundRect.width+=5;
			//boundRect.height+=5;
			input_Array[pos][0] = original(boundRect);
			input_Array[pos][1] = threshImage(boundRect);
			
			Mat pieceapprox = approx(boundRect);
			input_Array[pos][3] = pieceapprox;
			input_Array[pos][2] = drawing(boundRect);
			Mat lineimage = Mat::zeros(pieceapprox.size(), CV_8UC1 );
			vector<Vec4i> lines;
			HoughLinesP(pieceapprox, lines, 1, CV_PI/180, 40, 40, 30);
			//cout<<"lines sze:"<<lines.size();
			int linenum=0;
			float maxlength =0.0,distance =0.0;
			
			
			for( size_t i = 0; i < lines.size(); i++ )
			{
				Vec4i l = lines[i];
				line(lineimage , Point(l[0], l[1]), Point(l[2], l[3]), color, 2,8,0);
				distance = sqrt(pow(l[3]-l[1],2)+ pow(l[2]-l[0],2));
				//cout<<"distance:"<<distance;
				if(distance>maxlength)
				{
					maxlength=distance;
					linenum=i;
					float a = (float)(l[3]-l[1]);
					float slope = a/(float)(l[2]-l[0]);
					angle = atan(slope);
					angle = (angle*180*7/22);
				}
			}
			Vec4i l = lines[linenum];
			line(lineimage , Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,0), 2,8,0);
			//namedWindow("lines.jpg", CV_WINDOW_NORMAL);
			//imshow("lines.jpg", lineimage);
			//waitKey(0);
			cout<<"angle:"<<angle;
			if(angle!=0 && angle!=90)
			{
			fixOrientation(pos,angle,0);	
			//Rect crop = getBoundRect(pos);
			}
			//namedWindow("pieces.jpg", CV_WINDOW_NORMAL);
			//imshow("pieces.jpg",input_Array[pos][0]);
			//namedWindow("thresh.jpg", CV_WINDOW_NORMAL);
			//imshow("thresh.jpg",input_Array[pos][1]);
			
			//waitKey(10);
			
			string kpfname = "piece";
			stringstream ss;
			ss<<pos;
			kpfname = kpfname+ss.str()+".jpg";
			ss.clear();
			imwrite(kpfname,input_Array[pos][0]);
			pos++;
			//destroyAllWindows();
		}
	}
	cout<<"pos:"<<pos;
	//imwrite("boundrect.jpg",drawing);
	imwrite("approx.jpg",approx);
	
}
void detectCorners(int index)
{
	//Detects corners
	Mat imagecorner;
	cout<<"index:"<<index;
	input_Array[index][1].copyTo(imagecorner);
	
	int maxcorners = 4;
	float mindistance = 200;
	float qualityLevel  = 0.1;
	Mat mask;
	int blocksize = 3;
	int useHarrisDetector=1;
	double k=0.04;
//cvtColor(imagecorner,imagecorner,CV_BGR2GRAY);

	int y=((imagecorner.cols/2)-imagecorner.cols/4);
	for(int x=0;x<imagecorner.rows;x++)
	{
		int intensity =imagecorner.at<uchar>(x,y);
		//imageintensity is white move left to fin corner
		if(intensity==255)
		{
			for(;y>=3;y--)
			{
				intensity =imagecorner.at<uchar>(x,y-3);
				if(intensity==0 || y==0)
				{
						Point2f c1;
					c1.x=y;
					c1.y=x; 
					
					corners[index].push_back(c1);
					//circle( imagecorner, c1, 8, cv::Scalar(120,120,120), -1 );
					x=imagecorner.rows;
					y=0;
				}	
			}
		}
	}
	
	y=((imagecorner.cols/2)+imagecorner.cols/4);
	for(int x=0;x<imagecorner.rows;x++)
	{
		int intensity =imagecorner.at<uchar>(x,y);
		if(intensity==255)
		{
			for(;y<imagecorner.cols-2;y++)
			{
			intensity =imagecorner.at<uchar>(x,y+3);
				if(intensity==0 || y==imagecorner.cols)
				{
					Point2f c1;
					c1.x=y;
					c1.y=x;
					corners[index].push_back(c1);
					x=imagecorner.rows;
					y=imagecorner.cols;
					//circle( imagecorner, c1, 8, cv::Scalar(120,120,120), -1 );
				}
			}
		}
	}
	
	y=((imagecorner.cols/2)-imagecorner.cols/4);
	for(int x=imagecorner.rows;x>0;x--)
	{
		int intensity =imagecorner.at<uchar>(x,y);
		if(intensity==255)
		{
			for(;y>=3;y--)
			{
			intensity =imagecorner.at<uchar>(x,y-3);
				if(intensity==0 ||  y==0)
				{
					Point2f c1;
					c1.x=y;
					c1.y=x; 
					corners[index].push_back(c1);
					y=0;
					//circle( imagecorner, c1, 8, cv::Scalar(120,120,120), -1 );
					x=0;
				}
			}
		}
	}
	
	y=((imagecorner.cols/2)+imagecorner.cols/4);
	for(int x=imagecorner.rows;x>0;x--)
	{
		int intensity =imagecorner.at<uchar>(x,y);
		if(intensity==255)
		{
			for(;y<imagecorner.cols-2;y++)
			{
			intensity =imagecorner.at<uchar>(x,y+3);
				if(intensity==0 ||  y==imagecorner.cols)
				{
					Point2f c1;
					c1.x=y;
					c1.y=x; 
					corners[index].push_back(c1);
					//circle( imagecorner, c1, 8, cv::Scalar(120,120,120), -1 );
					y=imagecorner.cols;
					x=0;
				}
			}
		}
	}
	
	
	/*while(corners[index].size()!=4 && qualityLevel<=0.5)
	{
	vector< Point2f > c;
	goodFeaturesToTrack(imagecorner,corners[index],maxcorners,qualityLevel,mindistance,mask,blocksize, useHarrisDetector,k);
	qualityLevel+=0.1;
	}*/
	cout<<"\n csize:"<<corners[index].size();
	imagecorner = input_Array[index][3];
	if(corners[index].size()>1)
	{
	
	for( size_t i = 0; i < corners[index].size(); i++ )
    {
		circle( imagecorner, corners[index][i], 8, cv::Scalar(120,120,120), -1 );
    }

	//namedWindow("imagecorner.jpg",CV_WINDOW_NORMAL);
	//imshow("imagecorner.jpg",imagecorner);
	//waitKey(0);
	Point2f temp;	
	for( size_t i = 0; i < corners[index].size(); i++ )
	{
		for(size_t j = i+1; j < corners[index].size(); j++)
		{
			if(corners[index].at(i).y>corners[index].at(j).y)
			{
				temp = corners[index].at(j);
				corners[index].at(j) = corners[index].at(i);
				corners[index].at(i) = temp;
			}
		}
	}

	if(corners[index].at(0).x>corners[index].at(1).x)
	{
		temp = corners[index].at(0);
		corners[index].at(0) = corners[index].at(1);
		corners[index].at(1) = temp;
	}
	if(corners[index].size()>3)
	{
	if(corners[index].at(2).x<corners[index].at(3).x)
	{
		temp = corners[index].at(2);
		corners[index].at(2) = corners[index].at(3);
		corners[index].at(3) = temp;
	}	
	}
	}
	namedWindow("corners.jpg", CV_WINDOW_NORMAL);
	imshow("corners.jpg", imagecorner);
	waitKey(0);
}
Rect getBoundRect(int index)
{
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	Rect boundRect;
	Mat image = input_Array[index][1];
	Mat image2 = input_Array[index][0];
	Mat image3 = input_Array[index][2];
	Mat image4 = input_Array[index][3];
	findContours( image, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE,Point(0,0) );
	int maxindex = 0;
	float maxArea=0.0,area=0.0;
	for( int i = 0; i< contours.size(); i++ )
	{
		area = contourArea(contours[i]); 
		if(area>maxArea)
		{
			maxArea = area;
			maxindex =i;
		}
	}
	vector<vector<Point> > contours_poly( contours.size() );
	approxPolyDP( Mat(contours[maxindex]), contours_poly[maxindex], 3, true);
	boundRect = boundingRect(Mat(contours_poly[maxindex]));
	input_Array[index][1] = image(boundRect);
	input_Array[index][0] = image2(boundRect);
	input_Array[index][2] = image3(boundRect);
	input_Array[index][3] = image4(boundRect);
	return boundRect;
}
void ComputeGeometry(int index)
{
	Mat image = input_Array[index][1];
	detectCorners(index);
	//if()//{
	//}
}
void compareShape(int index1, int index2)
{
	Mat image1 = input_Array[index1][1];
	Mat image2 = input_Array[index2][1];
	Mat image = image1;
	int depth[2],j=0,arclength[2];
	depth[1]=0;
	depth[0]=0;
	arclength[0]=0;
	arclength[1]=0;
	int point = index1;
	
	int x1,y1,x2,y2;
	x1= corners[index1].at(0).x;
	y1 = corners[index1].at(0).y;
	x2= corners[index1].at(3).x;
	y2 = corners[index1].at(3).y;
	arclength[0] = pow(x1-x2,2)+pow(y1-y2,2);
	
	x1= corners[index2].at(0).x;
	y1 = corners[index2].at(0).y;
	x2= corners[index2].at(3).x;
	y2 = corners[index2].at(3).y;
	arclength[1] = pow(x1-x2,2)+pow(y1-y2,2);
	
	while(j<2)
	{
		int x= midpoints[point][3].x;
		int y= midpoints[point][3].y;
		int intensity = image.at<uchar>(x,y);
		
		if(intensity==0)
		{
			for(int i=0;i<image.cols-2;i++)
			{
				depth[j]++;
				intensity = image.at<uchar>(i+2,y);
				if(intensity ==255)
				{
					break;
				}
			}
		}
		else
		{
			for(int i=0;i>2;i--)
			{
				depth[j]++;
				intensity = image.at<uchar>(i-2,y);
				if(intensity ==0)
				{
					break;
				}
			}
		}
		point=index2;
		image= image2;
		j++;
	}
	
	cout<<"depth:"<<depth[0]<<" "<<depth[1];
	cout<<"arclength:"<<arclength[0]<<" "<<arclength[1];
	
	if(abs(depth[0]-depth[1])<5 && abs(arclength[1]-arclength[0])<2000)
	{
		cout<<"matched shape";
		shapematch=1;
	}
	else
		shapematch=0;
	
	//arcLength(corner[1],false);
}
void compareColor(int index1, int index2)
{
	
	colormatch=1;
}
void prepareCanvas()
{
	
	canvas.copyTo(prevcanvas);
	int index = cornerpieces[0][0];
	previndex = index;
	int i=0;
	for(i=0;i<4;i++)
	{
		if(edge_type[index][i] == 'L' && edge_type[index][(i+1)%4] =='L')
		{	
			break;
		}
	}
	if(i==0)
	{
		fixOrientation(index,270,1);
	}
	else if(i==1)
		fixOrientation(index,180,1);
	else if(i==2)
		fixOrientation(index,90,1);

	Rect bound = getBoundRect(index);
	Mat firstcorner = input_Array[index][0];

	
	namedWindow("firstcorner.jpg",CV_WINDOW_NORMAL);
	imshow("firstcorner.jpg",firstcorner);
	Mat half(canvas,Rect(0,0,firstcorner.cols,firstcorner.rows));
	firstcorner.copyTo(half);
	cornerpieces[0][1]=1;
	attachedcorners++;
	imwrite("canvas.jpg",canvas);
	namedWindow("canvas.jpg",CV_WINDOW_NORMAL);
	imshow("canvas.jpg",canvas);
	//waitKey(10);
	prevtype =1;
	ComputeGeometry(index);
}
int getCandidatePiece(int previndex, int prevtype)
{
	int index;
	char edge2;
	char edge1 = edge_type[previndex][3];
	if(edge1=='H')
		edge2='T';
	else if (edge1 == 'T')
		edge2 = 'H';
	else if (edge1 =='L')
	{
		index = cornerpieces[cornerindex][0];
				cornerindex++;
				
		temptype=1;
			compareColor(previndex,index);
			compareShape(previndex,index);
		return index;
	}
		
	if(prevtype ==1 || prevtype ==2)
	{
		for(int i=0;i<edgepieceindex;i++)
		{
			if(edgepiece[i][1]==0 && (edge_type[i][0] == edge2 || edge_type[i][1] == edge2 || edge_type[i][2] == edge2 || edge_type[i][3] == edge2))
			{
				index = edgepiece[i][0];
				temptype=2;
				compareColor(previndex,index);
				compareShape(previndex,index);
			 
			}
		}
	}
	else
	{
		for(int i=0;i<innerpieceindex;i++)
		{
			if(innerpieces[i][1]==0 && (edge_type[i][0] == edge2 || edge_type[i][1] == edge2 || edge_type[i][2] == edge2 || edge_type[i][3] == edge2))
			{
			index = innerpieces[i][0];
			temptype=3;
			compareColor(previndex,index);
			compareShape(previndex,index);
			}
		}

	}
	cout<<"index:"<<index;
	return index;
}

int main(int argc,char* argv[])
{
	Mat image,diff;
	if(argc>1)
	{
		image = imread(argv[1]);
		original = imread(argv[1]);
		
		/*cvtColor( image, image, CV_BGR2GRAY );
	
	
	int scale = 1;
	int delta = 0;
	int ddepth = CV_16S;
	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y,grad;

	normalize(image, image, 0, 255, NORM_MINMAX, CV_8UC1);
	
	// Gradient X
	Sobel( image, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_CONSTANT );
	convertScaleAbs( grad_x, abs_grad_x );
	// Gradient Y
	Sobel( image, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_CONSTANT );
	convertScaleAbs( grad_y, abs_grad_y );
	// Total Gradient (approximate)
	addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, image );
	imwrite("sobel.jpg", image);
	
	int bgintensity = image.at<Vec3b>(10,10)[0];
	Scalar meanIntensity=mean(image,noArray());
	
	cout<<"bgintensity:"<<bgintensity;
	cout<<"\n meanIntensity:"<<meanIntensity.val[0];
	
	medianBlur(image,image,5);
	imwrite("blur1.jpg",image);
	
	for(int x=0;x<image.cols;x++)
	{
		for(int y=0;y<image.rows;y++)
		{
			//Setting the intesity to 255
			
			if(image.at<uchar>(y,x)<=bgintensity-12)
				image.at<uchar>(y,x) = 255;
			else
				image.at<uchar>(y,x) = 0;
		}
	}*/
	
		
		cvtColor( image, image, CV_BGR2GRAY );
		int bgintensity = image.at<uchar>(10,10);

		cout<<"bgintensity:"<<bgintensity;
		Mat background(image.rows,image.cols, CV_8UC1,bgintensity);

		imwrite("diff.jpg",image);
		//adaptiveThreshold(image,image,255, ADAPTIVE_THRESH_MEAN_C,THRESH_BINARY,199,2);
		for(int x=0;x<image.cols;x++)
		{
			for(int y=0;y<image.rows;y++)
			{				
				if(400<=x && (x<=image.cols-400) && 400<=y && (y<=image.rows-400))
					bgintensity=bgintensity*2;
				if(abs(image.at<uchar>(y,x)- bgintensity)>30)// 
					image.at<uchar>(y,x) = 255;
				else
					image.at<uchar>(y,x) = 0;
			}
		}
		imwrite("threshold.jpg",image);
		medianBlur(image,image,7);
		extractPieces(image);
		for(int i=0;i<pos;i++)
		{
			detectCorners(i);
			assignLabels(i);
		}
		prepareCanvas();
		for(int i=0;i<pos;i++)
		{
			int candidateIndex = getCandidatePiece(previndex,prevtype);
			cout<<"\n candidateIndex:"<<candidateIndex;
			if(shapematch && colormatch)
			{
				previndex = candidateIndex;
				prevtype=temptype;
				attachToCanvas(candidateIndex);
				shapematch=0;
				colormatch=0;
				edgepiece[previndex][1]=1;	
			}
		}
		namedWindow("canvas.jpg", CV_WINDOW_NORMAL);
		imshow("canvas.jpg", canvas);
		waitKey(0);	
	}
	return 0;
}
	