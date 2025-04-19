#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <opencv2/opencv.hpp>

#include <stdint.h>

struct STBFParams {
	int32_t frame_width;
	int32_t frame_height;
	int32_t buffer_size;
	float sigma_space;
	float sigma_color;
	float sigma_time;
	int32_t radius;
};

class SpatioTemporalBilateralFilter {
    public:
	SpatioTemporalBilateralFilter(int32_t width, int32_t height,
				      int32_t buffer_size, float sigma_space,
				      float sigma_color, float sigma_time,
				      int32_t radius);
	~SpatioTemporalBilateralFilter();

	cv::Mat processFrame(const cv::Mat& input_frame);
	void reset();

    private:
	STBFParams params;

	int32_t current_idx;
	int32_t frames_processed;

	uint8_t* d_frame_buffer;
	uint8_t* d_output_frame;
	float* d_gaussian_space;

	void allocateMemory();
	void releaseMemory();
};

__global__ void
spatiotemporalBilateralKernel(uint8_t* frame_buffer, uint8_t* output_frame,
			      float* gaussian_space, int32_t width,
			      int32_t height, int32_t buffer_size,
			      int32_t frames_processed, int32_t radius,
			      float sigma_color, float sigma_time);