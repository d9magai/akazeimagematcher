/*
 * MatSerialization.h
 *
 *  Created on: 2015/10/12
 *      Author: d9magai
 */

#ifndef MATSERIALIZATION_H_
#define MATSERIALIZATION_H_

#include <opencv2/opencv.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/vector.hpp>

BOOST_SERIALIZATION_SPLIT_FREE(cv::Mat)
namespace boost {
namespace serialization {

/*** Mat ***/
template<class Archive>
void save(Archive & ar, const cv::Mat& m, const unsigned int version) {
    size_t elemSize = m.elemSize(), elemType = m.type();
    ar & m.cols;
    ar & m.rows;
    ar & elemSize;
    ar & elemType; // element type.
    size_t dataSize = m.cols * m.rows * m.elemSize();
    for (size_t dc = 0; dc < dataSize; ++dc) {
        ar & m.data[dc];
    }
}

template<class Archive>
void load(Archive & ar, cv::Mat& m, const unsigned int version) {
    int cols, rows;
    size_t elemSize, elemType;

    ar & cols;
    ar & rows;
    ar & elemSize;
    ar & elemType;

    m.create(rows, cols, elemType);
    size_t dataSize = m.cols * m.rows * elemSize;
    for (size_t dc = 0; dc < dataSize; ++dc) {
        ar & m.data[dc];
    }
}
}
}

#endif /* MATSERIALIZATION_H_ */
