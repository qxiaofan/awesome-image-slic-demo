#ifndef SLIC_H
#define SLIC_H

/* slic.h.
 *
 * Written by: Pascal Mettes.
 *
 * This file contains the class elements of the class Slic. This class is an
 * implementation of the SLIC Superpixel algorithm by Achanta et al. [PAMI'12,
 * vol. 34, num. 11, pp. 2274-2282].
 *
 * This implementation is created for the specific purpose of creating
 * over-segmentations in an OpenCV-based environment.
 */

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <float.h>

/* 2d matrices are handled by 2d vectors. */
#define vec2dd std::vector<std::vector<double> >
#define vec2di std::vector<std::vector<int> >
#define vec2db std::vector<std::vector<bool> >
/* The number of iterations run by the clustering algorithm. */
#define NR_ITERATIONS 4

namespace image_split {

/*
 * class Slic.
 *
 * In this class, an over-segmentation is created of an image, provided by the
 * step-size (distance between initial cluster locations) and the colour
 * distance parameter.
 */
class Slic {
    private:
        /* The cluster assignments and distance values for each pixel. */
        vec2dd distances;
        
        /* The LAB and xy values of the centers. */
        vec2dd centers;
        vec2di clusters;
        /* The number of occurences of each center. */
        std::vector<int> center_counts;
        
        /* The step size per cluster, and the colour (nc) and distance (ns)
         * parameters. */
        int step, nc, ns;
        
        /* Compute the distance between a center and an individual pixel. */
        double compute_dist(int ci, CvPoint pixel, CvScalar colour);
        /* Find the pixel with the lowest gradient in a 3x3 surrounding. */
        CvPoint find_local_minimum(IplImage *image, CvPoint center);
        
        /* Remove and initialize the 2d vectors. */
        void clear_data();
        void init_data(IplImage *image);

    public:
        /* Class constructors and deconstructors. */
        Slic();
        ~Slic();

        /* Generate an over-segmentation for an image. */
        void generate_superpixels(IplImage *image, int step, int nc);
        /* Enforce connectivity for an image. */
        void create_connectivity(IplImage *image);
        
        /* Draw functions. Resp. displayal of the centers and the contours. */
        void display_center_grid(IplImage *image, CvScalar colour);
        void display_contours(IplImage *image, CvScalar colour, std::vector<cv::Point> &contours);
        void colour_with_cluster_means(IplImage *image);

        /// get all points with same cluster
        void getClusterPoints(std::vector<std::vector<cv::Point > > &spPoints);
        /// get all centers
        vec2dd getClusterCenters() { return centers; }
};

} // namespace image_split

#endif // SLIC_H
