/*
Copyright 2022 NVIDIA CORPORATION

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include "nvblox/datasets/augiclnuim.h"

#include "nvblox/utils/logging.h"

#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <unordered_map>

#include "nvblox/utils/timing.h"

namespace nvblox {
namespace datasets {
namespace augiclnuim {
namespace internal {

void parsePoseFromStream(std::ifstream* trajectory_file_stream_ptr,
                         Transform* transform_ptr) {
  CHECK_NOTNULL(transform_ptr);
  constexpr int kDimension = 4;
  for (int row = 0; row < kDimension; row++) {
    for (int col = 0; col < kDimension; col++) {
      float item = 0.0;
      (*trajectory_file_stream_ptr) >> item;
      (*transform_ptr)(row, col) = item;
    }
  }
  // NOTE(alexmillane): There's some weird trailing characters... just reading
  // one extra char solves things
  char extra_char;
  (*trajectory_file_stream_ptr) >> extra_char;
}

std::string getPathForTrajectory(const std::string& base_path,
                                 const std::string& dataset_name) {
  return base_path + "/" + dataset_name + "-traj.log";
}

std::string getPathForDepthImage(const std::string& base_path,
                                 const std::string& dataset_name,
                                 const int frame_id) {
  std::stringstream ss;
  if (dataset_name == "livingroom1" || dataset_name == "livingroom2" || dataset_name == "office1" || dataset_name == "office2")
  {
    ss << base_path << "/" << dataset_name << "-depth/" << std::setfill('0') << std::setw(5) << frame_id << ".png";
  }
  else
  {
    ss << base_path << "/" << dataset_name << "-depth/" << std::setfill('0') << std::setw(6) << frame_id << ".png";
  }
  return ss.str();
}

std::string getPathForColorImage(const std::string& base_path,
                                 const std::string& dataset_name,
                                 const int frame_id) {
  std::stringstream ss;
  if (dataset_name == "livingroom1" || dataset_name == "livingroom2" || dataset_name == "office1" || dataset_name == "office2")
  {
    ss << base_path << "/" << dataset_name << "-color/" << std::setfill('0') << std::setw(5) << frame_id << ".jpg";
  }
  else
  {
    ss << base_path << "/" << dataset_name << "-color/" << std::setfill('0') << std::setw(6) << frame_id << ".png";
  }
  return ss.str();
}

std::unique_ptr<ImageLoader<DepthImage>> createDepthImageLoader(
    const std::string& base_path, const std::string& dataset_name,
    const float depth_scaling_factor, const bool multithreaded) {
  return createImageLoader<DepthImage>(
      std::bind(getPathForDepthImage, base_path, dataset_name,
                std::placeholders::_1),
      multithreaded, depth_scaling_factor);
}

std::unique_ptr<ImageLoader<ColorImage>> createColorImageLoader(
    const std::string& base_path, const std::string& dataset_name,
    const bool multithreaded) {
  return createImageLoader<ColorImage>(
      std::bind(getPathForColorImage, base_path, dataset_name,
                std::placeholders::_1),
      multithreaded);
}

}  // namespace internal

std::unique_ptr<Fuser> createFuser(const std::string base_path,
                                   const std::string dataset_name) {
  auto data_loader = DataLoader::create(base_path, dataset_name);
  if (!data_loader) {
    return std::unique_ptr<Fuser>();
  }
  return std::make_unique<Fuser>(std::move(data_loader));
}

std::unique_ptr<DataLoader> DataLoader::create(const std::string& base_path,
                                               const std::string& dataset_name,
                                               bool multithreaded) {
  // Construct a dataset loader but only return it if everything worked.
  auto dataset_loader = std::make_unique<DataLoader>(base_path, dataset_name, multithreaded);
  if (dataset_loader->setup_success_) {
    return dataset_loader;
  } else {
    return std::unique_ptr<DataLoader>();
  }
}

DataLoader::DataLoader(const std::string& base_path, const std::string& dataset_name, bool multithreaded)
    : RgbdDataLoaderInterface(
          internal::createDepthImageLoader(
              base_path, dataset_name,
              datasets::kDefaultUintDepthScaleFactor, multithreaded),
          internal::createColorImageLoader(
              base_path, dataset_name,
              multithreaded)),
      base_path_(base_path),
      dataset_name_(dataset_name),
      trajectory_file_(std::ifstream(internal::getPathForTrajectory(
          base_path, dataset_name))) {
  constexpr float fu = 525;
  constexpr float fv = 525;
  constexpr float cu = 319.5;
  constexpr float cv = 239.5;
  constexpr int width = 640;
  constexpr int height = 480;
  camera_ = Camera(fu, fv, cu, cv, width, height);
}

/// Interface for a function that loads the next frames in a dataset
///@param[out] depth_frame_ptr The loaded depth frame.
///@param[out] T_L_C_ptr Transform from Camera to the Layer frame.
///@param[out] camera_ptr The intrinsic camera model.
///@param[out] color_frame_ptr Optional, load color frame.
///@return Whether loading succeeded.
DataLoadResult DataLoader::loadNext(DepthImage* depth_frame_ptr,
                                    Transform* T_L_C_ptr, Camera* camera_ptr,
                                    ColorImage* color_frame_ptr) {
  CHECK(setup_success_);
  CHECK_NOTNULL(depth_frame_ptr);
  CHECK_NOTNULL(T_L_C_ptr);
  CHECK_NOTNULL(camera_ptr);
  // CHECK_NOTNULL(color_frame_ptr); // Can be null

  // Because we might fail along the way, increment the frame number before we
  // start.
  const int frame_number = frame_number_;
  ++frame_number_;

  // Load the image into a Depth Frame.
  CHECK(depth_image_loader_);
  timing::Timer timer_file_depth("file_loading/depth_image");
  if (!depth_image_loader_->getNextImage(depth_frame_ptr)) {
    LOG(INFO) << "Couldn't find depth image";
    return DataLoadResult::kNoMoreData;
  }
  timer_file_depth.Stop();

  // Load the color image into a ColorImage
  if (color_frame_ptr) {
    CHECK(color_image_loader_);
    timing::Timer timer_file_color("file_loading/color_image");
    if (!color_image_loader_->getNextImage(color_frame_ptr)) {
      LOG(INFO) << "Couldn't find color image";
      return DataLoadResult::kNoMoreData;
    }
    timer_file_color.Stop();
  }

  float scale;

  // Get the camera for this frame.
  timing::Timer timer_file_intrinsics("file_loading/camera");
  *camera_ptr = camera_;
  timer_file_intrinsics.Stop();

  // Get the next pose
  timing::Timer timer_file_pose("file_loading/pose");
  CHECK(trajectory_file_.is_open());
  std::string line;
  if (std::getline(trajectory_file_, line)) {
    augiclnuim::internal::parsePoseFromStream(&trajectory_file_, T_L_C_ptr);
  } else {
    LOG(INFO) << "Couldn't find pose";
    return DataLoadResult::kNoMoreData;
  }

  // Check that the loaded data doesn't contain NaNs or a faulty rotation
  // matrix. This does occur. If we find one, skip that frame and move to the
  // next.
  constexpr float kRotationMatrixDetEpsilon = 1e-4;
  if (!T_L_C_ptr->matrix().allFinite() ||
      std::abs(T_L_C_ptr->matrix().block<3, 3>(0, 0).determinant() - 1.0f) >
          kRotationMatrixDetEpsilon) {
    LOG(WARNING) << "Bad CSV data.";
    return DataLoadResult::kBadFrame;  // Bad data, but keep going.
  }

  timer_file_pose.Stop();
  return DataLoadResult::kSuccess;
}

}  // namespace augiclnuim
}  // namespace datasets
}  // namespace nvblox
