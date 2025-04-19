#pragma diag_suppress 611

#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include "filters/temporal_median_filter.cuh"
#include "filters/spatiotemporal_bilateral_filter.cuh"
#include "filters/optical_flow_blending.cuh"
#include "filters/deep_denoiser.cuh"

cv::VideoCapture openVideo(const std::string& file_path)
{
	cv::VideoCapture cap(file_path);
	if (!cap.isOpened()) {
		throw std::runtime_error("Error opening video file: " +
					 file_path);
	}
	return cap;
}

cv::Mat safeConcatFrames(const std::vector<cv::Mat>& frames)
{
	int total_frames = frames.size();

	int cols = std::ceil(std::sqrt(total_frames));
	int rows = std::ceil(static_cast<float>(total_frames) / cols);

	int target_width = 640;
	int target_height = 480;

	std::vector<cv::Mat> resized_frames;
	for (const auto& frame : frames) {
		cv::Mat resized_frame;
		cv::resize(frame, resized_frame,
			   cv::Size(target_width, target_height));
		resized_frames.push_back(resized_frame);
	}

	cv::Mat result(rows * target_height, cols * target_width,
		       resized_frames[0].type(), cv::Scalar(0));

	for (int i = 0; i < total_frames; ++i) {
		int row = i / cols;
		int col = i % cols;

		resized_frames[i].copyTo(
			result(cv::Rect(col * target_width, row * target_height,
					target_width, target_height)));
	}

	return result;
}

enum DenoiseMode {
	MODE_TMF = 0x01,
	MODE_STBF = 0x02,
	MODE_OFTB = 0x04,
	MODE_DNN = 0x08,
};

int32_t main(int32_t argc, char** argv)
{
#ifndef DEBUG
	try {
#endif
		std::string video_path;
		int32_t mode = 0xF;

		if (argc < 2) {
			std::cout << "Error: No video file provided.\n"
				  << "Enter path: ";
			std::cin >> video_path;
			std::cout << "Enter denoising mode (0-4): ";
			std::cin >> mode;

			if (video_path.empty()) {
				std::cerr
					<< "Usage: " << argv[0]
					<< " <path-to-video-file> [denoising-mode: 0x01-TMF, 0x02-STBF, 0x04-OFTB, 0x08-DNN, 0x0F-ALL]"
					<< '\n';
				return 1;
			}

		} else {
			video_path = argv[1];
			mode = (argc > 2) ? std::stoi(argv[2]) : 0x0F;
		}

		std::cout << "Processing video: " << video_path << '\n';
		std::cout << "Denoising mode: " << mode << '\n';

		cv::VideoCapture cap = openVideo(video_path);

		int32_t target_width = 640;
		int32_t target_height = 480;
		int32_t width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
		int32_t height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
		double fps = cap.get(cv::CAP_PROP_FPS);
		int32_t total_frames = cap.get(cv::CAP_PROP_FRAME_COUNT);

		std::cout << "Video properties:" << '\n';
		std::cout << "  Width: " << width << '\n';
		std::cout << "  Height: " << height << '\n';
		std::cout << "  Original FPS: " << fps << '\n';
		std::cout << "  Render FPS: " << fps << '\n';
		std::cout << "  Total frames: " << total_frames << '\n';

		TemporalMedianFilter tmf(width, height, 5);
		SpatioTemporalBilateralFilter stbf(width, height, 6, 3.0f,
						   18.0f, 1.0f, 3);
		OpticalFlowTemporalBlending oftb(width, height, 0.6f, 1);
		DeepDenoiser dnn("models/model.onnx", true);

		cv::Mat frame, processed_frame;
		int32_t frame_count = 0;
		int32_t wait_time = 1000 / fps;

		while (cap.read(frame)) {
			cv::Mat input_frame;
			if (frame.channels() > 1) {
				cv::cvtColor(frame, input_frame,
					     cv::COLOR_BGR2GRAY);
			} else {
				input_frame = frame.clone();
			}

			std::vector<cv::Mat> frames_to_concat;
			processed_frame = input_frame.clone();
			cv::Mat noise(processed_frame.size(),
				      processed_frame.type());
			cv::randn(noise, 0, 25);
			cv::Mat noisy_frame;
			cv::add(processed_frame, noise, noisy_frame,
				cv::noArray(), processed_frame.type());

			// frames_to_concat.push_back(input_frame);
			frames_to_concat.push_back(noisy_frame);

			constexpr float dnn_strength = 0.7f;

#ifdef DEBUG

			processed_frame = tmf.processFrame(input_frame);
			frames_to_concat.push_back(processed_frame);

			processed_frame = stbf.processFrame(input_frame);
			frames_to_concat.push_back(processed_frame);

			processed_frame = oftb.processFrame(input_frame);
			frames_to_concat.push_back(processed_frame);

			cv::Mat tmp = dnn.processFrame(input_frame);
			cv::Mat contrast_frame;
			cv::addWeighted(input_frame, 1.0f - dnn_strength, tmp,
					dnn_strength, 0.0f, contrast_frame);

			cv::convertScaleAbs(contrast_frame, processed_frame,
					    1.5, 0);
			frames_to_concat.push_back(processed_frame);

#else
		cv::Mat contrast_frame = input_frame.clone();
		if (mode & MODE_TMF) {
			contrast_frame = tmf.processFrame(contrast_frame);
		}
		if (mode & MODE_STBF) {
			contrast_frame = stbf.processFrame(contrast_frame);
		}
		if (mode & MODE_OFTB) {
			contrast_frame = oftb.processFrame(contrast_frame);
		}
		if (mode & MODE_DNN) {
			cv::Mat tmp = dnn.processFrame(contrast_frame);
			cv::addWeighted(input_frame, 1.0f - dnn_strength, tmp,
					dnn_strength, 0.0f, contrast_frame);
		}

		constexpr float sharp_strength = 0.1f;
		cv::Mat sharpened;
		cv::GaussianBlur(contrast_frame, sharpened, cv::Size(0, 0),
				 1.0);
		cv::addWeighted(contrast_frame, 1.5, sharpened, -0.5, 0,
				sharpened);
		processed_frame = sharpened;
		frames_to_concat.push_back(processed_frame);

#endif

			if (frames_to_concat.size() == 1) {
				std::cerr
					<< "Error: No filters applied for frame "
					<< frame_count << '\n';
				break;
			}

			cv::Mat display_frame =
				safeConcatFrames(frames_to_concat);
			cv::imshow("Original | Denoised", display_frame);

			if (cv::waitKey(wait_time) >= 0)
				break;

			frame_count++;
			if (frame_count % 100 == 0) {
				std::cout << "Processed " << frame_count
					  << " frames ("
					  << (frame_count * 100 / total_frames)
					  << "%)" << '\n';
			}
		}

		std::cout << "Finished processing " << frame_count << " frames"
			  << '\n';

		cap.release();
		cv::destroyAllWindows();

		return 0;
#ifndef DEBUG
	} catch (const std::exception& e) {
		std::cerr << "Error: " << e.what() << '\n';
		return 1;
	}
#endif
}
