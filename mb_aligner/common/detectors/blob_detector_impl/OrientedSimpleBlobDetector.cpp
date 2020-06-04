/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

//#include "precomp.hpp"
#include "OrientedSimpleBlobDetector.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/imgproc/imgproc.hpp"

//#include "opencv2/core/utility.hpp"
//#include "opencv2/core/private.hpp"
//#include "opencv2/core/ocl.hpp"
//#include "opencv2/core/hal/hal.hpp"

#include <algorithm>

#include <iterator>
#include <limits>

#include <math.h> /* atan2 */
//#define PI 3.14159265

//#define DEBUG_BLOB_DETECTOR

#ifdef DEBUG_BLOB_DETECTOR
#  include "opencv2/opencv_modules.hpp"
#  ifdef HAVE_OPENCV_HIGHGUI
#    include "opencv2/highgui.hpp"
#  else
#    undef DEBUG_BLOB_DETECTOR
#  endif
#endif

#define sign(a) a > 0 ? 1 : a == 0 ? 0 : -1

using namespace cv;
//namespace cv
//{

/*
*  OrientedSimpleBlobDetector
*/
OrientedSimpleBlobDetector::Params::Params()
{
    thresholdStep = 10;
    minThreshold = 50;
    maxThreshold = 220;
    minRepeatability = 2;
    minDistBetweenBlobs = 10;

    filterByColor = true;
    blobColor = 0;

    filterByArea = true;
    minArea = 25;
    maxArea = 5000;

    filterByCircularity = false;
    minCircularity = 0.8f;
    maxCircularity = std::numeric_limits<float>::max();

    filterByInertia = true;
    //minInertiaRatio = 0.6;
    minInertiaRatio = 0.1f;
    maxInertiaRatio = std::numeric_limits<float>::max();

    filterByConvexity = true;
    //minConvexity = 0.8;
    minConvexity = 0.95f;
    maxConvexity = std::numeric_limits<float>::max();
}

/*
void OrientedSimpleBlobDetector::Params::read(const cv::FileNode& fn )
{
    thresholdStep = fn["thresholdStep"];
    minThreshold = fn["minThreshold"];
    maxThreshold = fn["maxThreshold"];

    minRepeatability = (size_t)(int)fn["minRepeatability"];
    minDistBetweenBlobs = fn["minDistBetweenBlobs"];

    filterByColor = (int)fn["filterByColor"] != 0 ? true : false;
    blobColor = (uchar)(int)fn["blobColor"];

    filterByArea = (int)fn["filterByArea"] != 0 ? true : false;
    minArea = fn["minArea"];
    maxArea = fn["maxArea"];

    filterByCircularity = (int)fn["filterByCircularity"] != 0 ? true : false;
    minCircularity = fn["minCircularity"];
    maxCircularity = fn["maxCircularity"];

    filterByInertia = (int)fn["filterByInertia"] != 0 ? true : false;
    minInertiaRatio = fn["minInertiaRatio"];
    maxInertiaRatio = fn["maxInertiaRatio"];

    filterByConvexity = (int)fn["filterByConvexity"] != 0 ? true : false;
    minConvexity = fn["minConvexity"];
    maxConvexity = fn["maxConvexity"];
}

void OrientedSimpleBlobDetector::Params::write(cv::FileStorage& fs) const
{
    fs << "thresholdStep" << thresholdStep;
    fs << "minThreshold" << minThreshold;
    fs << "maxThreshold" << maxThreshold;

    fs << "minRepeatability" << (int)minRepeatability;
    fs << "minDistBetweenBlobs" << minDistBetweenBlobs;

    fs << "filterByColor" << (int)filterByColor;
    fs << "blobColor" << (int)blobColor;

    fs << "filterByArea" << (int)filterByArea;
    fs << "minArea" << minArea;
    fs << "maxArea" << maxArea;

    fs << "filterByCircularity" << (int)filterByCircularity;
    fs << "minCircularity" << minCircularity;
    fs << "maxCircularity" << maxCircularity;

    fs << "filterByInertia" << (int)filterByInertia;
    fs << "minInertiaRatio" << minInertiaRatio;
    fs << "maxInertiaRatio" << maxInertiaRatio;

    fs << "filterByConvexity" << (int)filterByConvexity;
    fs << "minConvexity" << minConvexity;
    fs << "maxConvexity" << maxConvexity;
}
*/

//OrientedSimpleBlobDetectorImpl::OrientedSimpleBlobDetectorImpl(const OrientedSimpleBlobDetector::Params &parameters) :
//params(parameters)
OrientedSimpleBlobDetector::OrientedSimpleBlobDetector(const OrientedSimpleBlobDetector::Params &parameters) :
params(parameters)
{
}

/*
void OrientedSimpleBlobDetectorImpl::read( const cv::FileNode& fn )
{
    params.read(fn);
}

void OrientedSimpleBlobDetectorImpl::write( cv::FileStorage& fs ) const
{
    writeFormat(fs);
    params.write(fs);
}
*/

void OrientedSimpleBlobDetector::findBlobs(InputArray _image, InputArray _binaryImage, std::vector<Center> &centers) const
{
    Mat image = _image.getMat(), binaryImage = _binaryImage.getMat();
    (void)image;
    centers.clear();

    std::vector < std::vector<Point> > contours;
    Mat tmpBinaryImage = binaryImage.clone();
    findContours(tmpBinaryImage, contours, RETR_LIST, CHAIN_APPROX_NONE);

#ifdef DEBUG_BLOB_DETECTOR
    //  Mat keypointsImage;
    //  cvtColor( binaryImage, keypointsImage, CV_GRAY2RGB );
    //
    //  Mat contoursImage;
    //  cvtColor( binaryImage, contoursImage, CV_GRAY2RGB );
    //  drawContours( contoursImage, contours, -1, Scalar(0,255,0) );
    //  imshow("contours", contoursImage );
#endif

    for (size_t contourIdx = 0; contourIdx < contours.size(); contourIdx++)
    {
        Center center;
        center.confidence = 1;
        Moments moms = moments(Mat(contours[contourIdx]));
        if (params.filterByArea)
        {
            double area = moms.m00;
            if (area < params.minArea || area >= params.maxArea)
                continue;
        }

        if (params.filterByCircularity)
        {
            double area = moms.m00;
            double perimeter = arcLength(Mat(contours[contourIdx]), true);
            double ratio = 4 * CV_PI * area / (perimeter * perimeter);
            if (ratio < params.minCircularity || ratio >= params.maxCircularity)
                continue;
        }

        if (params.filterByInertia)
        {
            double denominator = std::sqrt(std::pow(2 * moms.mu11, 2) + std::pow(moms.mu20 - moms.mu02, 2));
            const double eps = 1e-2;
            double ratio;
            if (denominator > eps)
            {
                double cosmin = (moms.mu20 - moms.mu02) / denominator;
                double sinmin = 2 * moms.mu11 / denominator;
                double cosmax = -cosmin;
                double sinmax = -sinmin;

                double imin = 0.5 * (moms.mu20 + moms.mu02) - 0.5 * (moms.mu20 - moms.mu02) * cosmin - moms.mu11 * sinmin;
                double imax = 0.5 * (moms.mu20 + moms.mu02) - 0.5 * (moms.mu20 - moms.mu02) * cosmax - moms.mu11 * sinmax;
                ratio = imin / imax;
            }
            else
            {
                ratio = 1;
            }

            if (ratio < params.minInertiaRatio || ratio >= params.maxInertiaRatio)
                continue;

            center.confidence = ratio * ratio;
        }

        if (params.filterByConvexity)
        {
            std::vector < Point > hull;
            convexHull(Mat(contours[contourIdx]), hull);
            double area = contourArea(Mat(contours[contourIdx]));
            double hullArea = contourArea(Mat(hull));
            double ratio = area / hullArea;
            if (ratio < params.minConvexity || ratio >= params.maxConvexity)
                continue;
        }

        if(moms.m00 == 0.0)
            continue;
        center.location = Point2d(moms.m10 / moms.m00, moms.m01 / moms.m00);

        if (params.filterByColor)
        {
            if (binaryImage.at<uchar> (cvRound(center.location.y), cvRound(center.location.x)) != params.blobColor)
                continue;
        }

        //compute blob radius
        {
            std::vector<double> dists;
            double minDist = 10000000;
            size_t minDistPointIdx = 0;
            for (size_t pointIdx = 0; pointIdx < contours[contourIdx].size(); pointIdx++)
            {
                Point2d pt = contours[contourIdx][pointIdx];
                double dist = norm(center.location - pt);
                dists.push_back(dist);
                if (dist < minDist) {
                    minDist = dist;
                    minDistPointIdx = pointIdx;
                }
            }
            std::sort(dists.begin(), dists.end());
            center.radius = (dists[(dists.size() - 1) / 2] + dists[dists.size() / 2]) / 2.;
            // Compute the angle by using major and minor axes of the image intensity
            // (https://en.wikipedia.org/wiki/Image_moment)
            center.angle = 0.f;
#if 0
            double x_bar = moms.m10 / moms.m00;
            double y_bar = moms.m01 / moms.m00;
            double u20_prime = moms.m20 / moms.m00 - x_bar * x_bar;
            double u02_prime = moms.m02 / moms.m00 - y_bar * y_bar;
            
            if(std::abs(u20_prime - u02_prime) < FLT_EPSILON)
                center.angle = 0.f;
            else {
                double u11_prime = moms.m11 / moms.m00 - x_bar * y_bar;
                double arctan_res = atan2(2.f * u11_prime , (u20_prime - u02_prime));
                /*
                double arctan_res = atan2(2 * u11_prime, u20_prime - u02_prime);
                // Adjust the atan2 to arctan (https://en.wikipedia.org/wiki/Atan2)
                if (u20_prime < u02_prime) { // atan2(y, x) and x < 0
                    if (u11_prime >= 0)
                        arctan_res -= CV_PI;
                    else
                        arctan_res += CV_PI;
                }
                */
                center.angle = (0.5 * arctan_res) / CV_PI * 180.f;
            }
            /*
            // Compute the angle by using the farthest point of the contour, relative to the horizontal x-axis of the image
            Point2d minDistPt = contours[contourIdx][minDistPointIdx];
            Point2d diff = minDistPt - center.location;
            center.angle = atan2(diff.y, diff.x) / CV_PI * 180.0;
            */
            if (center.angle < 0)
                center.angle += 360.f;
#endif

            double mu20_prime = moms.mu20 / moms.m00;
            double mu02_prime = moms.mu02 / moms.m00;

            if(std::abs(mu20_prime - mu02_prime) < FLT_EPSILON)
                center.angle = 0.f;
            else {
                double mu11_prime = moms.mu11 / moms.m00;
                double delta = sqrt(4 * mu11_prime * mu11_prime + (mu20_prime - mu02_prime) * (mu20_prime - mu02_prime));
                center.angle = atan2(4.f * mu11_prime, mu20_prime - mu02_prime + delta);

                /* fix orientation axis angle according to the 3rd moment and the angle
                 * (beacuse the orientation axis is between -pi/2 to pi/2 (-90 to 90 degrees)) */
                if (std::abs(center.angle) > CV_PI / 4) {
                    /* Look at the Y component */
                    if ((sign(moms.mu03)) != (sign(center.angle)))
                        center.angle += CV_PI;
                }
                else {
                    /* Look at the X component */
                    if (moms.mu30 < 0)
                        center.angle += CV_PI;
                }

                /*center.angle = (0.5 * arctan_res) / CV_PI * 180.f; */
                /* Convert to degrees */
                center.angle = center.angle / CV_PI * 180.f;
            }
            if (center.angle < 0)
                center.angle += 360.f;

        }

        centers.push_back(center);


#ifdef DEBUG_BLOB_DETECTOR
        //    circle( keypointsImage, center.location, 1, Scalar(0,0,255), 1 );
#endif
    }
#ifdef DEBUG_BLOB_DETECTOR
    //  imshow("bk", keypointsImage );
    //  waitKey();
#endif
}

void OrientedSimpleBlobDetector::detect(InputArray image, std::vector<cv::KeyPoint>& keypoints, InputArray)
{
    //TODO: support mask
    keypoints.clear();
    Mat grayscaleImage;
    if (image.channels() == 3) {
        cvtColor(image, grayscaleImage, COLOR_BGR2GRAY);
    } else {
        grayscaleImage = image.getMat();
    }

    if (grayscaleImage.type() != CV_8UC1) {
        CV_Error(CV_StsUnsupportedFormat, "Blob detector only supports 8-bit images!");
    }

    std::vector < std::vector<Center> > centers;
    for (double thresh = params.minThreshold; thresh < params.maxThreshold; thresh += params.thresholdStep)
    {
        Mat binarizedImage;
        threshold(grayscaleImage, binarizedImage, thresh, 255, THRESH_BINARY);

        std::vector < Center > curCenters;
        findBlobs(grayscaleImage, binarizedImage, curCenters);
        std::vector < std::vector<Center> > newCenters;
        for (size_t i = 0; i < curCenters.size(); i++)
        {
            bool isNew = true;
            for (size_t j = 0; j < centers.size(); j++)
            {
                double dist = norm(centers[j][ centers[j].size() / 2 ].location - curCenters[i].location);
                isNew = dist >= params.minDistBetweenBlobs && dist >= centers[j][ centers[j].size() / 2 ].radius && dist >= curCenters[i].radius;
                if (!isNew)
                {
                    centers[j].push_back(curCenters[i]);

                    size_t k = centers[j].size() - 1;
                    while( k > 0 && centers[j][k].radius < centers[j][k-1].radius )
                    {
                        centers[j][k] = centers[j][k-1];
                        k--;
                    }
                    centers[j][k] = curCenters[i];

                    break;
                }
            }
            if (isNew)
                newCenters.push_back(std::vector<Center> (1, curCenters[i]));
        }
        std::copy(newCenters.begin(), newCenters.end(), std::back_inserter(centers));
    }

    for (size_t i = 0; i < centers.size(); i++)
    {
        if (centers[i].size() < params.minRepeatability)
            continue;
        Point2d sumPoint(0, 0);
        double normalizer = 0;
        for (size_t j = 0; j < centers[i].size(); j++)
        {
            sumPoint += centers[i][j].confidence * centers[i][j].location;
            normalizer += centers[i][j].confidence;
        }
        sumPoint *= (1. / normalizer);
        KeyPoint kpt(sumPoint, (float)(centers[i][centers[i].size() / 2].radius) * 2.0f, (float)centers[i][centers[i].size() / 2].angle);
        keypoints.push_back(kpt);
    }
}

//}
