#include <sstream>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <aws/s3/model/ListObjectsRequest.h>
#include "matSerialization.h"
#include "commons.h"
#include "s3Utils.h"

int main(int argc, char** argv) {

    Aws::S3::S3Client s3client = d9magai::s3utils::getS3client();
    std::vector<Aws::String> s = { "path/to/f_lena.jpg", "path/to/img.jpg", "path/to/graf1.png", "path/to/graf3.png" };

    try {
        Aws::S3::Model::ListObjectsRequest listObjectsRequest;
        listObjectsRequest.SetBucket(d9magai::commons::BUCKET);
        listObjectsRequest.SetPrefix("prefix/");
        Aws::S3::Model::ListObjectsOutcome listObjectsOutcome = s3client.ListObjects(listObjectsRequest);

        if (!listObjectsOutcome.IsSuccess()) {
            std::stringstream ss;
            ss << "get listObjects error :" << listObjectsOutcome.GetError().GetMessage() << std::endl;
            throw ss.str();
        }
        for (const auto& object : listObjectsOutcome.GetResult().GetContents()) {
            std::cout << object.GetKey() << std::endl;
        }

        cv::Ptr<cv::FeatureDetector> d = cv::ORB::create();
        std::vector<cv::KeyPoint> kp;
        cv::Mat image = d9magai::s3utils::getImage(s3client, d9magai::commons::BUCKET, "path/to/img.jpg");
        d->detect(image, kp);
        cv::Mat queryDesc;
        d->compute(image, kp, queryDesc);

        cv::Ptr<cv::flann::IndexParams> indexParams = new cv::flann::LshIndexParams(12, 20, 2);
        cv::Ptr<cv::FlannBasedMatcher> m = cv::Ptr<cv::FlannBasedMatcher>(new cv::FlannBasedMatcher(indexParams));
        std::vector<cv::Mat> desc = d9magai::s3utils::getDescriptor(s3client, d9magai::commons::BUCKET, "path/to/matcher1");
        std::vector<cv::DMatch> matches;
        m->add(desc);
        m->train();
        m->match(queryDesc, matches);

        int votes[s.size()] = { }; // 学習画像の投票箱
        // 投票数の多い画像のIDと特徴点の数を調査
        int maxImageId = -1;
        int maxVotes = 0;
        for (const cv::DMatch& match : matches) {
            if (match.distance < 45.0) {
                votes[match.imgIdx]++;
                if (votes[match.imgIdx] > maxVotes) {
                    maxImageId = match.imgIdx;        //マッチした特徴点を一番多く持つ学習画像のIDを記憶
                    maxVotes = votes[match.imgIdx];        //マッチした特徴点の数
                }
            }
        }

        if (static_cast<double>(maxVotes) / m->getTrainDescriptors()[maxImageId].rows < 0.05) {
            maxImageId = -1; // マッチした特徴点の数が全体の5%より少なければ、未検出とする
        }
        std::cout << maxImageId << std::endl;
        std::cout << s[maxImageId] << std::endl;

    } catch (std::string str) {
        std::cerr << str << std::endl;
    }

    return 0;
}
