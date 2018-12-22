#ifndef CAFFE_VIDEO_DATA_LAYER_HPP_
#define CAFFE_VIDEO_DATA_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Provides data to the Net from video files.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class VideoDataLayer : public BasePrefetchingDataLayer<Dtype> {
public:
  explicit VideoDataLayer(const LayerParameter& param)
  : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~VideoDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "VideoData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 2; }

protected:
  shared_ptr<Caffe::RNG> prefetch_rng_;
  shared_ptr<Caffe::RNG> prefetch_rng_2_;
  shared_ptr<Caffe::RNG> prefetch_rng_1_;
  shared_ptr<Caffe::RNG> frame_prefetch_rng_;
  virtual void ShuffleVideos();
  virtual void load_batch(Batch<Dtype>* batch);

#ifdef USE_MPI
  inline virtual void advance_cursor(){
    lines_id_++;
    if (lines_id_ >= lines_.size()) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
      if (this->layer_param_.video_data_param().shuffle()) {
        ShuffleVideos();
      }
    }
  }
#endif

  vector<std::pair<std::string, int> > lines_;
  vector<int> lines_duration_;
  int lines_id_;
  string name_pattern_;
};


}  // namespace caffe

#endif  // CAFFE_IMAGE_DATA_LAYER_HPP_
