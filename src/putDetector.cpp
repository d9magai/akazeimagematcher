#include <sstream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <aws/s3/model/GetObjectRequest.h>
#include <aws/s3/model/PutObjectRequest.h>
#include <aws/core/utils/HashingUtils.h>
#include "matSerialization.h"
#include "commons.h"
#include "s3Utils.h"

int main(int argc, char** argv) {

    Aws::S3::S3Client s3Client = d9magai::s3utils::getS3client();

    try {
        cv::Ptr<cv::FeatureDetector> d = cv::AKAZE::create();
        cv::Ptr<cv::flann::IndexParams> indexParams = new cv::flann::LshIndexParams(12, 20, 2);
        cv::Ptr<cv::FlannBasedMatcher> m = cv::Ptr<cv::FlannBasedMatcher>(new cv::FlannBasedMatcher(indexParams));

        std::vector<Aws::String> s = { "path/to/f_lena.jpg", "path/to/img.jpg", "path/to/graf1.png", "path/to/graf3.png" };
        int i = 0;
        std::vector<cv::Mat> addDesc;
        for (auto itr = s.begin(); itr != s.end(); ++itr) {
            std::cout << i << ":" << (*itr) << std::endl;
            cv::Mat image = d9magai::s3utils::getImageFromS3(s3Client, d9magai::commons::BUCKET, (*itr));
            std::vector<cv::KeyPoint> kp;
            d->detect(image, kp);
            std::cout << "detected: " << kp.size() << std::endl;
            cv::Mat desc;
            d->compute(image, kp, desc);
            std::cout << "descriptor size: " << desc.size() << std::endl;
            addDesc.push_back(desc);
            i++;
        }
        m->add(addDesc);
        m->train();

        std::string result;
        std::stringstream ss;
        boost::archive::binary_oarchive oa(ss);
        oa << addDesc;
        result = ss.str();

        Aws::S3::Model::PutObjectRequest putObjectRequest;
        putObjectRequest.SetBucket("d9magai.mybucket");
        std::shared_ptr<Aws::IOStream> objectStream = Aws::MakeShared<Aws::StringStream>("PutDetector");
        *objectStream << result;
        objectStream->flush();
        putObjectRequest.SetBody(objectStream);
        putObjectRequest.SetContentLength(static_cast<long>(putObjectRequest.GetBody()->tellp()));
        putObjectRequest.SetContentMD5(Aws::Utils::HashingUtils::Base64Encode(Aws::Utils::HashingUtils::CalculateMD5(*putObjectRequest.GetBody())));
        putObjectRequest.SetContentType("application/octet-stream");
        putObjectRequest.SetKey("path/to/matcher1");
        Aws::S3::Model::PutObjectOutcome putObjectOutcome = s3Client.PutObject(putObjectRequest);
        if (!putObjectOutcome.IsSuccess()) {
            std::stringstream ss;
            ss << "Object put failed tos3 with error :" << putObjectOutcome.GetError().GetMessage() << std::endl;
            throw ss.str();
        }

    } catch (std::string str) {
        std::cerr << str << std::endl;
    }

    return 0;
}
