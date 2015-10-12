/*
 * s3Utils.h
 *
 *  Created on: 2015/10/12
 *      Author: d9magai
 */

#ifndef S3UTILS_H_
#define S3UTILS_H_

#include <aws/core/auth/AWSCredentialsProvider.h>
#include <aws/core/utils/StringUtils.h>
#include <aws/s3/S3Client.h>
#include <aws/s3/model/GetObjectRequest.h>
#include <aws/s3/model/PutObjectRequest.h>
#include <aws/core/utils/HashingUtils.h>
#include <aws/core/utils/StringUtils.h>
#include "commons.h"

Aws::S3::S3Client getS3client() {

    Aws::Client::ClientConfiguration config;
    config.scheme = Aws::Http::Scheme::HTTPS;
    config.connectTimeoutMs = 30000;
    config.requestTimeoutMs = 30000;
    config.region = Aws::Region::AP_NORTHEAST_1;
    Aws::S3::S3Client s3client(Aws::Auth::AWSCredentials(AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY), config);
    return s3client;
}

#endif /* S3UTILS_H_ */
