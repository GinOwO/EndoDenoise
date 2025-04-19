#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/all_layers.hpp>

#include <string>

class DeepDenoiser {
    public:
	DeepDenoiser(const std::string &model_path, bool use_cuda = true);
	~DeepDenoiser();

	cv::Mat processFrame(const cv::Mat &input_frame);
	void reset();

    private:
	cv::dnn::Net network;
	bool model_loaded;
	bool use_cuda;
	cv::Size input_size;
	float scale;
	cv::Scalar mean;

	void createDummyModel();
};
