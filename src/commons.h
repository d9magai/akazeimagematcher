/*
 * commons.h
 *
 *  Created on: 2015/10/12
 *      Author: d9magai
 */

#ifndef COMMONS_H_
#define COMMONS_H_

#include <aws/core/utils/StringUtils.h>

namespace d9magai {
namespace commons {

const Aws::String AWS_ACCESS_KEY_ID = std::getenv("AWS_ACCESS_KEY_ID");
const Aws::String AWS_SECRET_ACCESS_KEY = std::getenv("AWS_SECRET_ACCESS_KEY");
const Aws::String BUCKET = "d9magai.mybudket";

}
}

#endif /* COMMONS_H_ */
