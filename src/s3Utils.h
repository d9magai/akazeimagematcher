/*
 * s3Utils.h
 *
 *  Created on: 2015/10/12
 *      Author: d9magai
 */

#ifndef S3UTILS_H_
#define S3UTILS_H_

#include <aws/s3/S3Client.h>
#include <aws/s3/model/GetObjectRequest.h>
#include <aws/core/auth/AWSCredentialsProvider.h>
#include <aws/core/utils/StringUtils.h>
#include <opencv2/opencv.hpp>
#include "commons.h"

namespace d9magai {
namespace s3utils {

Aws::S3::S3Client getS3client() {

    Aws::Client::ClientConfiguration config;
    config.scheme = Aws::Http::Scheme::HTTPS;
    config.connectTimeoutMs = 30000;
    config.requestTimeoutMs = 30000;
    config.region = Aws::Region::AP_NORTHEAST_1;
    return Aws::S3::S3Client(Aws::Auth::AWSCredentials(d9magai::commons::AWS_ACCESS_KEY_ID, d9magai::commons::AWS_SECRET_ACCESS_KEY), config);
}

cv::Mat getImageFromS3(Aws::S3::S3Client s3client, Aws::String bucket, Aws::String key) {

    Aws::S3::Model::GetObjectRequest getObjectRequest;
    getObjectRequest.SetBucket(bucket);
    getObjectRequest.SetKey(key);
    auto getObjectOutcome = s3client.GetObject(getObjectRequest);
    if (!getObjectOutcome.IsSuccess()) {
        std::stringstream ss;
        ss << "File download failed from s3 with error :" << getObjectOutcome.GetError().GetMessage() << std::endl;
        throw ss.str();
    }

    std::stringstream ss;
    ss << getObjectOutcome.GetResult().GetBody().rdbuf();
    std::string str = ss.str();
    std::vector<char> vec(str.begin(), str.end());
    return cv::imdecode(cv::Mat(vec), CV_LOAD_IMAGE_COLOR);
}

}
}

#endif /* S3UTILS_H_ */
