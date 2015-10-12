#include <sstream>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include "matSerialization.h"
#include "commons.h"
#include "s3Utils.h"

int main(int argc, char** argv) {

    Aws::S3::S3Client s3client = d9magai::s3utils::getS3client();
    std::vector<Aws::String> s = { "path/to/f_lena.jpg", "path/to/img.jpg", "path/to/graf1.png", "path/to/graf3.png" };

    try {
        cv::Ptr<cv::FeatureDetector> d = cv::AKAZE::create();
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

        int length = s.size();
        int votes[length]; // 学習画像の投票箱
        for (int i = 0; i < length; i++)
            votes[i] = 0;

        for (unsigned int i = 0; i < matches.size(); i++) {
            float d = matches[i].distance;
            if (d < 45.0) {
                votes[matches[i].imgIdx]++;
            }
        }
        // 投票数の多い画像のIDを調査
        int maxImageId = -1;
        int maxVotes = 0;
        for (int i = 0; i < length; i++) {
            if (votes[i] > maxVotes) {
                maxImageId = i;  //マッチした特徴点を一番多く持つ学習画像のID
                maxVotes = votes[i]; //マッチした特徴点の数
            }
        }
        std::vector<cv::Mat> trainDescs = m->getTrainDescriptors();
        float similarity = (float) maxVotes / trainDescs[maxImageId].rows * 100;
        if (similarity < 5) {
            maxImageId = -1; // マッチした特徴点の数が全体の5%より少なければ、未検出とする
        }
        std::cout << maxImageId << std::endl;
        std::cout << s[maxImageId] << std::endl;

    } catch (std::string str) {
        std::cerr << str << std::endl;
    }

    return 0;
}
