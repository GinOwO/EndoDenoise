#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>

#include <stdint.h>

struct OFTBParams {
	int32_t frame_width;
	int32_t frame_height;
	float blending_factor;
	int32_t flow_method; // 0: Farneback, 1: TVL1
};

class OpticalFlowTemporalBlending {
    public:
	OpticalFlowTemporalBlending(int32_t width, int32_t height,
				    float blending_factor = 0.5f,
				    int32_t flow_method = 1);
	~OpticalFlowTemporalBlending();

	cv::Mat processFrame(const cv::Mat& input_frame);
	void reset();

    private:
	OFTBParams params;
	cv::Mat prev_frame;
	cv::Ptr<cv::cuda::DenseOpticalFlow> flow;

	cv::cuda::GpuMat d_prev_frame;
	cv::cuda::GpuMat d_current_frame;
	cv::cuda::GpuMat d_flow;
	cv::cuda::GpuMat d_warped_frame;
	cv::cuda::GpuMat d_blended_frame;

	cv::cuda::GpuMat d_map_x;
	cv::cuda::GpuMat d_map_y;

	bool is_initialized;

	void initializeGPUBuffers(const cv::Size& frame_size);
	void releaseGPUBuffers();
};