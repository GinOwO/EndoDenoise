#include "filters/spatiotemporal_bilateral_filter.cuh"

#include <stdio.h>
#include <stdint.h>

__global__ void spatiotemporalBilateralKernel(
	uint8_t *frame_buffer, uint8_t *output_frame,
	float *gaussian_space, int32_t width, int32_t height, int32_t buffer_size,
	int32_t frames_processed, int32_t radius, float sigma_color, float sigma_time)
{
	const int32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	const int32_t y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height)
		return;

	const int32_t pixelIdx = y * width + x;
	const int32_t actualSize = min(buffer_size, frames_processed);
	const int32_t pixelsPerFrame = width * height;

	float sum = 0.0f;
	float totalWeight = 0.0f;

	const int32_t centerFrameIdx = frames_processed % buffer_size;
	const int32_t centerframe_offset = centerFrameIdx * pixelsPerFrame;
	const uint8_t centerValue =
		frame_buffer[centerframe_offset + pixelIdx];

	for (int32_t t = 0; t < actualSize; t++) {
		int32_t frame_offset = t * pixelsPerFrame;
		float timeWeight = __expf(
			-(abs(t - centerFrameIdx) * abs(t - centerFrameIdx)) /
			(2.0f * sigma_time * sigma_time));

		for (int32_t dy = -radius; dy <= radius; dy++) {
			int32_t ny = y + dy;
			if (ny < 0 || ny >= height)
				continue;

			for (int32_t dx = -radius; dx <= radius; dx++) {
				int32_t nx = x + dx;
				if (nx < 0 || nx >= width)
					continue;

				int32_t neighborIdx = ny * width + nx;
				uint8_t neighborValue =
					frame_buffer[frame_offset + neighborIdx];

				int32_t spaceIdx =
					(dy + radius) * (2 * radius + 1) +
					(dx + radius);
				float spaceWeight = gaussian_space[spaceIdx];

				float colorWeight = __expf(
					-(neighborValue - centerValue) *
					(neighborValue - centerValue) /
					(2.0f * sigma_color * sigma_color));

				float weight =
					spaceWeight * colorWeight * timeWeight;
				sum += weight * neighborValue;
				totalWeight += weight;
			}
		}
	}

	if (totalWeight > 0.0f) {
		output_frame[pixelIdx] =
			static_cast<uint8_t>(sum / totalWeight);
	} else {
		output_frame[pixelIdx] = centerValue;
	}
}

SpatioTemporalBilateralFilter::SpatioTemporalBilateralFilter(
	int32_t width, int32_t height, int32_t buffer_size, float sigma_space,
	float sigma_color, float sigma_time, int32_t radius)
{
	params.frame_width = width;
	params.frame_height = height;
	params.buffer_size = buffer_size;
	params.sigma_space = sigma_space;
	params.sigma_color = sigma_color;
	params.sigma_time = sigma_time;
	params.radius = radius;

	current_idx = 0;
	frames_processed = 0;

	allocateMemory();
}

SpatioTemporalBilateralFilter::~SpatioTemporalBilateralFilter()
{
	releaseMemory();
}

void SpatioTemporalBilateralFilter::allocateMemory()
{
	size_t frame_size = params.frame_width * params.frame_height;
	size_t buffer_size = frame_size * params.buffer_size;

	cudaMalloc((void**)&d_frame_buffer,
		   buffer_size * sizeof(uint8_t));
	cudaMemset(d_frame_buffer, 0, buffer_size * sizeof(uint8_t));

	cudaMalloc((void**)&d_output_frame, frame_size * sizeof(uint8_t));
	cudaMemset(d_output_frame, 0, frame_size * sizeof(uint8_t));

	int32_t spaceKernelSize = (2 * params.radius + 1) * (2 * params.radius + 1);
	cudaMalloc((void**)&d_gaussian_space, spaceKernelSize * sizeof(float));

	float *h_gaussian_space = new float[spaceKernelSize];
	float sigma_space2 = 2.0f * params.sigma_space * params.sigma_space;

	for (int32_t dy = -params.radius; dy <= params.radius; dy++) {
		for (int32_t dx = -params.radius; dx <= params.radius; dx++) {
			int32_t idx =
				(dy + params.radius) * (2 * params.radius + 1) +
				(dx + params.radius);
			h_gaussian_space[idx] =
				expf(-(dx * dx + dy * dy) / sigma_space2);
		}
	}

	cudaMemcpy(d_gaussian_space, h_gaussian_space,
		   spaceKernelSize * sizeof(float), cudaMemcpyHostToDevice);
	delete[] h_gaussian_space;
}

void SpatioTemporalBilateralFilter::releaseMemory()
{
	if (d_frame_buffer)
		cudaFree(d_frame_buffer);
	if (d_output_frame)
		cudaFree(d_output_frame);
	if (d_gaussian_space)
		cudaFree(d_gaussian_space);

	d_frame_buffer = nullptr;
	d_output_frame = nullptr;
	d_gaussian_space = nullptr;
}

void SpatioTemporalBilateralFilter::reset()
{
	current_idx = 0;
	frames_processed = 0;

	size_t frame_size = params.frame_width * params.frame_height;
	size_t buffer_size = frame_size * params.buffer_size;

	cudaMemset(d_frame_buffer, 0, buffer_size * sizeof(uint8_t));
}

cv::Mat SpatioTemporalBilateralFilter::processFrame(const cv::Mat& input_frame)
{
	cv::Mat gray_frame;
	if (input_frame.channels() > 1) {
		cv::cvtColor(input_frame, gray_frame, cv::COLOR_BGR2GRAY);
	} else {
		gray_frame = input_frame.clone();
	}

	size_t frame_size = params.frame_width * params.frame_height;
	size_t frame_offset = current_idx * frame_size;

	cudaMemcpy(d_frame_buffer + frame_offset, gray_frame.data,
		   frame_size * sizeof(uint8_t), cudaMemcpyHostToDevice);

	dim3 blockDim(16, 16);
	dim3 gridDim((params.frame_width + blockDim.x - 1) / blockDim.x,
		     (params.frame_height + blockDim.y - 1) / blockDim.y);

	spatiotemporalBilateralKernel<<<gridDim, blockDim>>>(
		d_frame_buffer, d_output_frame, d_gaussian_space,
		params.frame_width, params.frame_height, params.buffer_size,
		frames_processed + 1, params.radius, params.sigma_color,
		params.sigma_time);

	cv::Mat output_frame(params.frame_height, params.frame_width, CV_8UC1);
	cudaMemcpy(output_frame.data, d_output_frame,
		   frame_size * sizeof(uint8_t), cudaMemcpyDeviceToHost);

	current_idx = (current_idx + 1) % params.buffer_size;
	frames_processed++;

	return output_frame;
}