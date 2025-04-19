#include "filters/deep_denoiser.cuh"

#include <iostream>
#include <fstream>
#include <stdint.h>

DeepDenoiser::DeepDenoiser(const std::string& model_path, bool use_cuda)
	: model_loaded(false)
	, use_cuda(use_cuda)
	, scale(1.0f / 255.0f)
	, mean(0, 0, 0)
{
	std::ifstream model_file(model_path);
	if (model_file.good()) {
		try {
			network = cv::dnn::readNetFromONNX(model_path);

			if (use_cuda) {
				network.setPreferableBackend(
					cv::dnn::DNN_BACKEND_CUDA);
				network.setPreferableTarget(
					cv::dnn::DNN_TARGET_CUDA);
			} else {
				network.setPreferableBackend(
					cv::dnn::DNN_BACKEND_DEFAULT);
				network.setPreferableTarget(
					cv::dnn::DNN_TARGET_CPU);
			}

			model_loaded = true;
			std::cout << "Successfully loaded DNN model from: "
				  << model_path << '\n' << "Model Backend: "
				  << (use_cuda ? "CUDA" : "CPU") << '\n';
			
		} catch (const cv::Exception& e) {
			std::cerr << "Error loading DNN model: " << e.what()
				  << '\n';
			createDummyModel();
		}
	} else {
		std::cerr << "DNN model file not found: " << model_path << '\n';
		std::cerr << "Using fallback processing instead..." << '\n';
		createDummyModel();
	}
}

DeepDenoiser::~DeepDenoiser()
{
}

void DeepDenoiser::createDummyModel()
{
	model_loaded = false;
}

void DeepDenoiser::reset()
{
}

cv::Mat DeepDenoiser::processFrame(const cv::Mat& input_frame)
{
	if (!model_loaded) {
		cv::Mat result;
		cv::GaussianBlur(input_frame, result, cv::Size(5, 5), 1.5);
		return result;
	}

	try {
		cv::Mat blob = cv::dnn::blobFromImage(
			input_frame, scale, cv::Size(), mean, false, false);
		network.setInput(blob);
		cv::Mat output = network.forward();
		cv::Mat result(input_frame.size(), CV_8UC1);
		cv::Mat outputReshaped = output.reshape(1, input_frame.rows);
		outputReshaped.convertTo(result, CV_8UC1, 255.0);

		return result;
	} catch (const cv::Exception& e) {
		std::cerr << "Error during DNN inference: " << e.what() << '\n';
		cv::Mat result;
		cv::GaussianBlur(input_frame, result, cv::Size(5, 5), 1.5);
		return result;
	}
}