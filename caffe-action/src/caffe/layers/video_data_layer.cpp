#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/video_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
VideoDataLayer<Dtype>::~VideoDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void VideoDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int new_height = this->layer_param_.video_data_param().new_height();
  const int new_width  = this->layer_param_.video_data_param().new_width();

  const int new_length  = this->layer_param_.video_data_param().new_length();
  const int num_segments = this->layer_param_.video_data_param().num_segments();
  //const string& source = this->layer_param_.video_data_param().source();


  //const bool is_color  = this->layer_param_.video_data_param().is_color();
  //string root_folder = this->layer_param_.video_data_param().root_folder();

  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";

  // Read the file with filenames and labels
  const string& source = this->layer_param_.video_data_param().source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());

  string filename;
  int label;
  int length;
  while (infile >> filename >> length >> label){
    lines_.push_back(std::make_pair(filename,label));
    lines_duration_.push_back(length);
  }


  if (this->layer_param_.video_data_param().shuffle()){
    const unsigned int prefectch_rng_seed = caffe_rng_rand();
    prefetch_rng_1_.reset(new Caffe::RNG(prefectch_rng_seed));
    prefetch_rng_2_.reset(new Caffe::RNG(prefectch_rng_seed));
    ShuffleVideos();
  }

  LOG(INFO) << "A total of " << lines_.size() << " videos.";
  lines_id_ = 0;

  Datum datum;

  const unsigned int frame_prefectch_rng_seed = caffe_rng_rand();
  frame_prefetch_rng_.reset(new Caffe::RNG(frame_prefectch_rng_seed));
  int average_duration = (int) lines_duration_[lines_id_]/num_segments;
  LOG(INFO)  << "avarge_duration" << average_duration ; 
  vector<int> offsets;
  for (int i = 0; i < num_segments; ++i){
    caffe::rng_t* frame_rng = static_cast<caffe::rng_t*>(frame_prefetch_rng_->generator());
    int offset = (*frame_rng)() % (average_duration - new_length + 1);
    LOG(INFO) << "offset" << offset ;
    offsets.push_back(offset+i*average_duration);
  }

  CHECK(ReadSegmentFLOW_RGBToDatum(lines_[lines_id_].first, lines_[lines_id_].second,
                  offsets, new_height, new_width, new_length, &datum, false));

  const int crop_size = this->layer_param_.transform_param().crop_size();
  const int batch_size = this->layer_param_.video_data_param().batch_size();

  for (int i = 0; i < this->prefetch_.size(); ++i) {
  if (crop_size > 0){
    top[0]->Reshape(batch_size, datum.channels(), crop_size, crop_size);
    this->prefetch_[i]->data_.Reshape(batch_size, datum.channels(), crop_size, crop_size);
  } else {
    top[0]->Reshape(batch_size, datum.channels(), datum.height(), datum.width());
    this->prefetch_[i]->data_.Reshape(batch_size, datum.channels(), datum.height(), datum.width());
  }
}

  LOG(INFO) << "output data size: " << top[0]->num() << "," << top[0]->channels() << "," << top[0]->height() << "," << top[0]->width();

  top[1]->Reshape(batch_size, 1, 1, 1);
  //this->prefetch_label_.Reshape(batch_size, 1, 1, 1);

  for (int i = 0; i < this->prefetch_.size(); ++i) {
    this->prefetch_[i]->label_.Reshape(batch_size, 1, 1, 1);
  }


  vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
  this->transformed_data_.Reshape(top_shape);


 /* // Read an image, and use it to initialize the top blob.
  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
                                    new_height, new_width, is_color);
  CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
  // Use data_transformer to infer the expected blob shape from a cv_image.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  this->transformed_data_.Reshape(top_shape);
  // Reshape prefetch_data and top[0] according to the batch_size.
  const int batch_size = this->layer_param_.video_data_param().batch_size();
  CHECK_GT(batch_size, 0) << "Positive batch size required";
  top_shape[0] = batch_size;
  for (int i = 0; i < this->prefetch_.size(); ++i) {
    this->prefetch_[i]->data_.Reshape(top_shape);
  }
  top[0]->Reshape(top_shape);

  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  vector<int> label_shape(1, batch_size);
  top[1]->Reshape(label_shape);
  for (int i = 0; i < this->prefetch_.size(); ++i) {
    this->prefetch_[i]->label_.Reshape(label_shape);
  }*/
}

template <typename Dtype>
void VideoDataLayer<Dtype>::ShuffleVideos(){
  caffe::rng_t* prefetch_rng1 = static_cast<caffe::rng_t*>(prefetch_rng_1_->generator());
  caffe::rng_t* prefetch_rng2 = static_cast<caffe::rng_t*>(prefetch_rng_2_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng1);
  shuffle(lines_duration_.begin(), lines_duration_.end(),prefetch_rng2);
}

// This function is called on prefetch thread
template <typename Dtype>
void VideoDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
    Datum datum;

  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;

  Dtype* top_data = batch->data_.mutable_cpu_data();
  Dtype* top_label = batch->label_.mutable_cpu_data();

  VideoDataParameter video_data_param = this->layer_param_.video_data_param();
  const int batch_size = video_data_param.batch_size();
  const int new_height = video_data_param.new_height();
  const int new_width = video_data_param.new_width();
  const int new_length = video_data_param.new_length();
  const int num_segments = video_data_param.num_segments();
  const int lines_size = lines_.size();


  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
    timer.Start();
    CHECK_GT(lines_size, lines_id_);

    vector<int> offsets;    
    int average_duration = (int) lines_duration_[lines_id_] / num_segments;
    float average_duration_2 = (float) lines_duration_[lines_id_] / num_segments;
    for (int i = 0; i < num_segments; ++i){
      if (this->phase_==TRAIN){
        if (average_duration >= new_length){
          caffe::rng_t* frame_rng = static_cast<caffe::rng_t*>(frame_prefetch_rng_->generator());
          int offset = (*frame_rng)() % (average_duration - new_length + 1);
          offsets.push_back(offset+i*average_duration);
        } else {
          caffe::rng_t* frame_rng = static_cast<caffe::rng_t*>(frame_prefetch_rng_->generator());
          int offset = (*frame_rng)() % (average_duration + 1);
          int temp = int(offset + i*average_duration_2);
          if (temp > lines_duration_[lines_id_] - new_length){
            temp = lines_duration_[lines_id_] - new_length;
          }
          offsets.push_back(temp);
        }
      } else{
        if (average_duration_2 >= new_length){
          offsets.push_back(int((average_duration_2-new_length+1)/2 + i*average_duration_2));
        }
        else {
          average_duration_2 = (float) (lines_duration_[lines_id_] - new_length) / (num_segments-1);
            offsets.push_back(int(i * average_duration_2));
        }
      }      
    }
  //if (this->phase_==TEST && lines_[lines_id_].second == 34){
  //  LOG(INFO) << lines_[lines_id_].first <<"__";
 /// }

    if(!ReadSegmentFLOW_RGBToDatum(lines_[lines_id_].first, lines_[lines_id_].second,
                offsets, new_height, new_width, new_length, &datum, false)){
      continue;
    }

    read_time += timer.MicroSeconds();
    timer.Start();

    int offset1 = batch->data_.offset(item_id);
      this->transformed_data_.set_cpu_data(top_data + offset1);
    this->data_transformer_->Transform(datum, &(this->transformed_data_));
    top_label[item_id] = lines_[lines_id_].second;

    trans_time += timer.MicroSeconds();

    // go to the next iter
    lines_id_++;
    if (lines_id_ >= lines_size) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
      if (this->layer_param_.video_data_param().shuffle()) {
        ShuffleVideos();
      }
    }
  }
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(VideoDataLayer);
REGISTER_LAYER_CLASS(VideoData);

}  // namespace caffe
#endif  // USE_OPENCV
