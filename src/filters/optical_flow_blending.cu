#include "filters/optical_flow_blending.cuh"

#include <stdint.h>

OpticalFlowTemporalBlending::OpticalFlowTemporalBlending(int32_t width, int32_t height,
							 float blending_factor,
							 int32_t flow_method)
{
	params.frame_width = width;
	params.frame_height = height;
	params.blending_factor = blending_factor;
	params.flow_method = flow_method;
	is_initialized = false;

	if (flow_method == 0) {
		flow = cv::cuda::FarnebackOpticalFlow::create();
	} else {
		flow = cv::cuda::OpticalFlowDual_TVL1::create();
	}
}

OpticalFlowTemporalBlending::~OpticalFlowTemporalBlending()
{
	releaseGPUBuffers();
}

void OpticalFlowTemporalBlending::initializeGPUBuffers(
	const cv::Size& frame_size)
{
	try {
		d_prev_frame.create(frame_size, CV_8UC1);
		d_current_frame.create(frame_size, CV_8UC1);
		d_flow.create(frame_size, CV_32FC2);
		d_warped_frame.create(frame_size, CV_8UC1);
		d_blended_frame.create(frame_size, CV_8UC1);
		d_map_x.create(frame_size, CV_32FC1);
		d_map_y.create(frame_size, CV_32FC1);
	} catch (const cv::Exception& e) {
		std::cerr << "CUDA buffer creation failed: " << e.what()
			  << '\n';
		throw;
	}
}

void OpticalFlowTemporalBlending::releaseGPUBuffers()
{
	d_prev_frame.release();
	d_current_frame.release();
	d_flow.release();
	d_warped_frame.release();
	d_blended_frame.release();
	d_map_x.release();
	d_map_y.release();
}

void OpticalFlowTemporalBlending::reset()
{
	is_initialized = false;
	prev_frame.release();
}

cv::Mat OpticalFlowTemporalBlending::processFrame(const cv::Mat& input_frame)
{
	if (input_frame.empty()) {
		std::cerr << "Input frame is empty!" << '\n';
		return cv::Mat();
	}

	cv::Mat gray_frame;
	if (input_frame.channels() > 1) {
		cv::cvtColor(input_frame, gray_frame, cv::COLOR_BGR2GRAY);
	} else {
		gray_frame = input_frame.clone();
	}

	if (!is_initialized) {
		prev_frame = gray_frame.clone();
		initializeGPUBuffers(gray_frame.size());
		d_prev_frame.upload(prev_frame);
		is_initialized = true;
		return gray_frame;
	}

	d_current_frame.upload(gray_frame);

	try {
		flow->calc(d_prev_frame, d_current_frame, d_flow);
	} catch (const cv::Exception& e) {
		std::cerr
			<< "Error during optical flow calculation: " << e.what()
			<< '\n';
		return cv::Mat();
	}

	cv::cuda::GpuMat flow_parts[2];
	cv::cuda::split(d_flow, flow_parts);

	cv::cuda::GpuMat grid_x(d_flow.size(), CV_32FC1);
	cv::cuda::GpuMat grid_y(d_flow.size(), CV_32FC1);

	if (d_flow.empty()) {
		std::cerr << "Error: d_flow is empty!" << '\n';
		return cv::Mat();
	}

	cv::Mat h_grid_x(d_flow.size(), CV_32FC1);
	cv::Mat h_grid_y(d_flow.size(), CV_32FC1);

	for (int32_t y = 0; y < h_grid_x.rows; ++y) {
		float *row_x = h_grid_x.ptr<float>(y);
		float *row_y = h_grid_y.ptr<float>(y);

		for (int32_t x = 0; x < h_grid_x.cols; ++x) {
			row_x[x] = static_cast<float>(x);
			row_y[x] = static_cast<float>(y);
		}
	}

	grid_x.upload(h_grid_x);
	grid_y.upload(h_grid_y);

	cv::cuda::add(grid_x, flow_parts[0], d_map_x);
	cv::cuda::add(grid_y, flow_parts[1], d_map_y);

	cv::cuda::remap(d_prev_frame, d_warped_frame, d_map_x, d_map_y,
			cv::INTER_LINEAR, cv::BORDER_REPLICATE);

	cv::cuda::addWeighted(d_warped_frame, 1.0 - params.blending_factor,
			      d_current_frame, params.blending_factor, 0.0,
			      d_blended_frame);

	cv::Mat output_frame;
	d_blended_frame.download(output_frame);

	d_prev_frame = d_current_frame.clone();
	prev_frame = gray_frame.clone();

	return output_frame;
}
