#include <sstream>
#include <fstream>
#include <pqxx/pqxx>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include "matSerialization.h"
#include "commons.h"
#include "s3Utils.h"

int main(int argc, char** argv) {

    Aws::S3::S3Client s3clinet = d9magai::s3utils::getS3client();

    try {
        pqxx::connection conn("dbname=mtg user=d9magai");
        pqxx::work T(conn);
        pqxx::result R(T.exec("SELECT imagename FROM cards ORDER BY id"));

        std::vector<Aws::String> cards;
        for (pqxx::result::const_iterator c = R.begin(); c != R.end(); ++c) {
            std::stringstream ss;
            ss << c[0].as(std::string());
            std::string str = ss.str();
            cards.push_back("LEA/" + Aws::Utils::StringUtils::URLEncode(str.c_str()) + ".jpg");
        }
        T.commit();
        conn.disconnect();

        cv::Ptr<cv::FeatureDetector> d = cv::ORB::create();
        int i = 0;
        std::vector<cv::Mat> descriptors;
        for (auto itr = cards.begin(); itr != cards.end(); ++itr) {
            std::cout << i << ":" << (*itr) << std::endl;
            cv::Mat image = d9magai::s3utils::getImage(s3clinet, "mtg.d9magai.jp", (*itr));
            std::vector<cv::KeyPoint> kp;
            d->detect(image, kp);
            std::cout << "detected: " << kp.size() << std::endl;
            cv::Mat descriptor;
            d->compute(image, kp, descriptor);
            std::cout << "descriptor size: " << descriptor.size() << std::endl;
            descriptors.push_back(descriptor);
            i++;
        }

        d9magai::s3utils::putDescriptor(s3clinet, descriptors, "mtg.d9magai.jp", "LEA/descriptor");

    } catch (std::string str) {
        std::cerr << str << std::endl;
    }

    return 0;
}
