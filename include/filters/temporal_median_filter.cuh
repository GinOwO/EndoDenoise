#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <opencv2/opencv.hpp>

#include <stdint.h>

struct TMFParams {
	int32_t frame_width;
	int32_t frame_height;
	int32_t buffer_size;
};

class TemporalMedianFilter {
    public:
	TemporalMedianFilter(int32_t width, int32_t height,
			     int32_t buffer_size);
	~TemporalMedianFilter();

	cv::Mat processFrame(const cv::Mat& input_frame);

	void reset();

    private:
	TMFParams params;

	int32_t current_idx;
	int32_t frames_processed;

	uint8_t *d_frame_buffer;
	uint8_t *d_output_frame;

	void allocateMemory();
	void releaseMemory();
};

__global__ void temporalMedianKernel(uint8_t *frame_buffer,
				     uint8_t *output_frame, int32_t width,
				     int32_t height, int32_t buffer_size,
				     int32_t frames_processed);