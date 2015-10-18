#include <sstream>
#include <cstdlib>
#include <chrono>
#include <pqxx/pqxx>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include "matSerialization.h"
#include "commons.h"
#include "s3Utils.h"

int main(int argc, char** argv) {

    auto start = std::chrono::system_clock::now();

    Aws::S3::S3Client s3client = d9magai::s3utils::getS3client();
    std::vector<Aws::String> s = { "path/to/f_lena.jpg", "path/to/img.jpg", "path/to/graf1.png", "path/to/graf3.png", "path/to/rotate.jpg", "path/to/f_lena.jpg", "path/to/img.jpg", "path/to/graf1.png", "path/to/graf3.png", "path/to/rotate.jpg", "path/to/f_lena.jpg", "path/to/img.jpg", "path/to/graf1.png", "path/to/graf3.png", "path/to/rotate.jpg", "path/to/f_lena.jpg", "path/to/img.jpg", "path/to/graf1.png", "path/to/graf3.png", "path/to/rotate.jpg", "path/to/f_lena.jpg", "path/to/img.jpg", "path/to/graf1.png", "path/to/graf3.png", "path/to/rotate.jpg", "path/to/f_lena.jpg", "path/to/img.jpg", "path/to/graf1.png", "path/to/graf3.png", "path/to/rotate.jpg", "path/to/f_lena.jpg", "path/to/img.jpg", "path/to/graf1.png", "path/to/graf3.png", "path/to/rotate.jpg", "path/to/f_lena.jpg", "path/to/img.jpg", "path/to/graf1.png", "path/to/graf3.png", "path/to/rotate.jpg", "path/to/f_lena.jpg", "path/to/img.jpg", "path/to/graf1.png", "path/to/graf3.png", "path/to/rotate.jpg",
            "path/to/f_lena.jpg", "path/to/img.jpg", "path/to/graf1.png", "path/to/graf3.png", "path/to/rotate.jpg", "path/to/f_lena.jpg", "path/to/img.jpg", "path/to/graf1.png", "path/to/graf3.png", "path/to/rotate.jpg", "path/to/f_lena.jpg", "path/to/img.jpg", "path/to/graf1.png", "path/to/graf3.png", "path/to/rotate.jpg", "path/to/f_lena.jpg", "path/to/img.jpg", "path/to/graf1.png", "path/to/graf3.png", "path/to/rotate.jpg", "path/to/f_lena.jpg", "path/to/img.jpg", "path/to/graf1.png", "path/to/graf3.png", "path/to/rotate.jpg", "path/to/f_lena.jpg", "path/to/img.jpg", "path/to/graf1.png", "path/to/graf3.png", "path/to/rotate.jpg", "path/to/f_lena.jpg", "path/to/img.jpg", "path/to/graf1.png", "path/to/graf3.png", "path/to/rotate.jpg", "path/to/f_lena.jpg", "path/to/img.jpg", "path/to/graf1.png", "path/to/graf3.png", "path/to/rotate.jpg", "path/to/f_lena.jpg", "path/to/img.jpg", "path/to/graf1.png", "path/to/graf3.png", "path/to/rotate.jpg", "path/to/f_lena.jpg", "path/to/img.jpg",
            "path/to/graf1.png", "path/to/graf3.png", "path/to/rotate.jpg", "path/to/f_lena.jpg", "path/to/img.jpg", "path/to/graf1.png", "path/to/graf3.png", "path/to/rotate.jpg", "path/to/f_lena.jpg", "path/to/img.jpg", "path/to/graf1.png", "path/to/graf3.png", "path/to/rotate.jpg", "path/to/f_lena.jpg", "path/to/img.jpg", "path/to/graf1.png", "path/to/graf3.png", "path/to/rotate.jpg", "path/to/f_lena.jpg", "path/to/img.jpg", "path/to/graf1.png", "path/to/graf3.png", "path/to/rotate.jpg", "path/to/f_lena.jpg", "path/to/img.jpg", "path/to/graf1.png", "path/to/graf3.png", "path/to/rotate.jpg", "path/to/f_lena.jpg", "path/to/img.jpg", "path/to/graf1.png", "path/to/graf3.png", "path/to/rotate.jpg", "path/to/f_lena.jpg", "path/to/img.jpg", "path/to/graf1.png", "path/to/graf3.png", "path/to/rotate.jpg", "path/to/f_lena.jpg", "path/to/img.jpg", "path/to/graf1.png", "path/to/graf3.png", "path/to/rotate.jpg" };

    try {
        cv::Ptr<cv::FeatureDetector> d = cv::ORB::create();
        std::vector<cv::KeyPoint> kp;
        //cv::Mat image = d9magai::s3utils::getImage(s3client, d9magai::commons::BUCKET, "path/to/rotate.jpg");
        cv::Mat image = cv::imread("13929.jpg", CV_LOAD_IMAGE_GRAYSCALE);
        d->detect(image, kp);
        cv::Mat queryDesc;
        d->compute(image, kp, queryDesc);
        auto end = std::chrono::system_clock::now();       // 計測終了時刻を保存
        auto dur = end - start;        // 要した時間を計算
        auto msec = std::chrono::duration_cast < std::chrono::milliseconds > (dur).count();
        std::cout << "get image, and compute: " << msec << " milli sec \n";

        start = std::chrono::system_clock::now();
        cv::Ptr<cv::flann::IndexParams> indexParams = new cv::flann::LshIndexParams(13, 20, 2);
        cv::Ptr<cv::FlannBasedMatcher> m = cv::Ptr<cv::FlannBasedMatcher>(new cv::FlannBasedMatcher(indexParams));
        std::vector<cv::Mat> desc = d9magai::s3utils::getDescriptor(s3client, "mtg.d9magai.jp", "LEA/descriptor");
        end = std::chrono::system_clock::now();       // 計測終了時刻を保存
        dur = end - start;        // 要した時間を計算
        msec = std::chrono::duration_cast < std::chrono::milliseconds > (dur).count();
        std::cout << "download descriptor: " << msec << " milli sec \n";

        start = std::chrono::system_clock::now();
        std::vector<cv::DMatch> matches;
        m->add(desc);
        end = std::chrono::system_clock::now();       // 計測終了時刻を保存
        dur = end - start;        // 要した時間を計算
        msec = std::chrono::duration_cast < std::chrono::milliseconds > (dur).count();
        std::cout << "add desc: " << msec << " milli sec \n";
        start = std::chrono::system_clock::now();
        m->train();
        end = std::chrono::system_clock::now();       // 計測終了時刻を保存
        dur = end - start;        // 要した時間を計算
        msec = std::chrono::duration_cast < std::chrono::milliseconds > (dur).count();
        std::cout << "train: " << msec << " milli sec \n";
        start = std::chrono::system_clock::now();
        m->match(queryDesc, matches);
        end = std::chrono::system_clock::now();       // 計測終了時刻を保存
        dur = end - start;        // 要した時間を計算
        msec = std::chrono::duration_cast < std::chrono::milliseconds > (dur).count();
        std::cout << "matching: " << msec << " milli sec \n";

        start = std::chrono::system_clock::now();
        int votes[295] = { }; // 学習画像の投票箱
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
        end = std::chrono::system_clock::now();       // 計測終了時刻を保存
        dur = end - start;        // 要した時間を計算
        msec = std::chrono::duration_cast < std::chrono::milliseconds > (dur).count();
        std::cout << "get once:" << msec << " milli sec \n";

        pqxx::connection conn("dbname=mtg user=d9magai");
        pqxx::work T(conn);
        std::string sql = "SELECT name FROM cards  WHERE id = " + std::to_string(maxImageId + 1);
        pqxx::result R(T.exec(sql));
        for (pqxx::result::const_iterator c = R.begin(); c != R.end(); ++c) {
            std::cout << c[0].as(std::string()) << std::endl;
        }
        T.commit();
        conn.disconnect();
        //std::cout << (maxImageId + 1) << std::endl;
        //std::cout << s[maxImageId] << std::endl;
    } catch (std::string str) {
        std::cerr << str << std::endl;
    }

    return 0;
}
