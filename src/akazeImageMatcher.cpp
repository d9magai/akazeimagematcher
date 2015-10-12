#include <sstream>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/features2d.hpp>
#include <aws/core/utils/StringUtils.h>
#include <aws/s3/model/GetObjectRequest.h>
#include "matSerialization.h"
#include "commons.h"
#include "s3Utils.h"

int main(int argc, char** argv) {

    Aws::S3::S3Client s3client = d9magai::s3utils::getS3client();
    std::vector<Aws::String> s = { "path/to/f_lena.jpg", "path/to/img.jpg", "path/to/graf1.png", "path/to/graf3.png" };

    try {
        cv::Ptr<cv::FeatureDetector> d = cv::AKAZE::create();
        cv::Ptr<cv::flann::IndexParams> indexParams = new cv::flann::LshIndexParams(12, 20, 2);
        cv::Ptr<cv::FlannBasedMatcher> m = cv::Ptr<cv::FlannBasedMatcher>(new cv::FlannBasedMatcher(indexParams));

        std::vector<cv::Mat> desc;
        Aws::S3::Model::GetObjectRequest getObjectRequest;
        getObjectRequest.SetBucket("d9magai.mybucket");
        getObjectRequest.SetKey("path/to/matcher1");
        auto getObjectOutcome = s3client.GetObject(getObjectRequest);
        if (!getObjectOutcome.IsSuccess()) {
            std::stringstream ss;
            ss << "File download failed from s3 with error " << getObjectOutcome.GetError().GetMessage() << std::endl;
            throw ss.str();
        }
        std::stringstream ss;
        ss << getObjectOutcome.GetResult().GetBody().rdbuf();
        boost::archive::binary_iarchive ar(ss);
        ar >> desc;
        cv::Ptr<cv::FlannBasedMatcher> m2 = cv::Ptr<cv::FlannBasedMatcher>(new cv::FlannBasedMatcher(indexParams));
        m2->add(desc);
        m2->train();

        cv::Mat image = d9magai::s3utils::getImage(s3client, d9magai::commons::BUCKET, "path/to/img.jpg");
        std::vector<cv::KeyPoint> kp;
        d->detect(image, kp);
        cv::Mat queryDesc;
        d->compute(image, kp, queryDesc);

        std::vector<cv::DMatch> matches;
        m2->match(queryDesc, matches);

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
        std::vector<cv::Mat> trainDescs = m2->getTrainDescriptors();
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
