#include <sstream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <aws/core/auth/AWSCredentialsProvider.h>
#include <aws/core/utils/StringUtils.h>
#include <aws/s3/S3Client.h>
#include <aws/s3/model/GetObjectRequest.h>
#include <aws/s3/model/PutObjectRequest.h>
#include <aws/core/utils/HashingUtils.h>
#include <aws/core/utils/StringUtils.h>
#include <boost/shared_ptr.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/vector.hpp>

const Aws::String AWS_ACCESS_KEY_ID = std::getenv("AWS_ACCESS_KEY_ID");
const Aws::String AWS_SECRET_ACCESS_KEY = std::getenv("AWS_SECRET_ACCESS_KEY");

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

cv::Mat getImage(Aws::S3::S3Client s3Client, Aws::String key) {

    Aws::S3::Model::GetObjectRequest getObjectRequest;
    getObjectRequest.SetBucket("d9magai.mybucket");
    getObjectRequest.SetKey(key);

    auto getObjectOutcome = s3Client.GetObject(getObjectRequest);
    if (!getObjectOutcome.IsSuccess()) {
        std::stringstream ss;
        ss << "File download failed from s3 with error " << getObjectOutcome.GetError().GetMessage() << std::endl;
        throw ss.str();
    }

    std::stringstream ss;
    ss << getObjectOutcome.GetResult().GetBody().rdbuf();
    std::string str = ss.str();
    std::vector<char> v(str.begin(), str.end());
    return cv::imdecode(cv::Mat(v), CV_LOAD_IMAGE_COLOR);
}

int main(int argc, char** argv) {

    Aws::Client::ClientConfiguration config;
    config.scheme = Aws::Http::Scheme::HTTPS;
    config.connectTimeoutMs = 30000;
    config.requestTimeoutMs = 30000;
    config.region = Aws::Region::AP_NORTHEAST_1;
    Aws::S3::S3Client s3Client(Aws::Auth::AWSCredentials(AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY), config);

    try {
        cv::Ptr<cv::FeatureDetector> d = cv::AKAZE::create();
        cv::Ptr<cv::flann::IndexParams> indexParams = new cv::flann::LshIndexParams(12, 20, 2);
        cv::Ptr<cv::FlannBasedMatcher> m = cv::Ptr<cv::FlannBasedMatcher>(new cv::FlannBasedMatcher(indexParams));

        std::vector<Aws::String> s = { "path/to/f_lena.jpg", "path/to/img.jpg", "path/to/graf1.png", "path/to/graf3.png" };
        int i = 0;
        std::vector<cv::Mat> addDesc;
        for (auto itr = s.begin(); itr != s.end(); ++itr) {
            std::cout << i << ":" << (*itr) << std::endl;
            cv::Mat image = getImage(s3Client, (*itr));
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
