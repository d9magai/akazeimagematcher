#include <sstream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include "matSerialization.h"
#include "commons.h"
#include "s3Utils.h"

int main(int argc, char** argv) {

    Aws::S3::S3Client s3clinet = d9magai::s3utils::getS3client();

    try {
        cv::Ptr<cv::FeatureDetector> d = cv::AKAZE::create();

        std::vector<Aws::String> s = { "path/to/f_lena.jpg", "path/to/img.jpg", "path/to/graf1.png", "path/to/graf3.png" };
        int i = 0;
        std::vector<cv::Mat> descriptors;
        for (auto itr = s.begin(); itr != s.end(); ++itr) {
            std::cout << i << ":" << (*itr) << std::endl;
            cv::Mat image = d9magai::s3utils::getImage(s3clinet, d9magai::commons::BUCKET, (*itr));
            std::vector<cv::KeyPoint> kp;
            d->detect(image, kp);
            std::cout << "detected: " << kp.size() << std::endl;
            cv::Mat descriptor;
            d->compute(image, kp, descriptor);
            std::cout << "descriptor size: " << descriptor.size() << std::endl;
            descriptors.push_back(descriptor);
            i++;
        }

        d9magai::s3utils::putDescriptor(s3clinet, descriptors, d9magai::commons::BUCKET, "path/to/matcher1");

    } catch (std::string str) {
        std::cerr << str << std::endl;
    }

    return 0;
}
