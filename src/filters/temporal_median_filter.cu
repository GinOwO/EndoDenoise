#include "filters/temporal_median_filter.cuh"

#include <stdio.h>
#include <stdint.h>

__global__ void temporalMedianKernel(uint8_t *frame_buffer,
				     uint8_t *output_frame, int32_t width,
				     int32_t height, int32_t buffer_size,
				     int32_t frames_processed)
{
	int32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	int32_t y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height)
		return;

	int32_t pixelIdx = y * width + x;

	uint8_t values[16];
	int32_t actualSize = min(buffer_size, frames_processed);

	for (int32_t t = 0; t < actualSize; t++) {
		int32_t frame_offset = t * width * height;
		values[t] = frame_buffer[frame_offset + pixelIdx];
	}

	for (int32_t i = 0; i < actualSize - 1; i++) {
		for (int32_t j = 0; j < actualSize - i - 1; j++) {
			if (values[j] > values[j + 1]) {
				uint8_t temp = values[j];
				values[j] = values[j + 1];
				values[j + 1] = temp;
			}
		}
	}

	uint8_t medianValue;
	if (actualSize % 2 == 0) {
		medianValue =
			(values[actualSize / 2 - 1] + values[actualSize / 2]) /
			2;
	} else {
		medianValue = values[actualSize / 2];
	}

	output_frame[pixelIdx] = medianValue;
}

TemporalMedianFilter::TemporalMedianFilter(int32_t width, int32_t height,
					   int32_t buffer_size)
{
	params.frame_width = width;
	params.frame_height = height;
	params.buffer_size = buffer_size;

	current_idx = 0;
	frames_processed = 0;

	allocateMemory();
}

TemporalMedianFilter::~TemporalMedianFilter()
{
	releaseMemory();
}

void TemporalMedianFilter::allocateMemory()
{
	size_t frame_size = params.frame_width * params.frame_height;
	size_t buffer_size = frame_size * params.buffer_size;

	cudaMalloc((void**)&d_frame_buffer,
		   buffer_size * sizeof(uint8_t));
	cudaMemset(d_frame_buffer, 0, buffer_size * sizeof(uint8_t));

	cudaMalloc((void**)&d_output_frame, frame_size * sizeof(uint8_t));
	cudaMemset(d_output_frame, 0, frame_size * sizeof(uint8_t));

	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		printf("CUDA Error during allocation: %s\n",
		       cudaGetErrorString(error));
	}
}

void TemporalMedianFilter::releaseMemory()
{
	if (d_frame_buffer)
		cudaFree(d_frame_buffer);
	if (d_output_frame)
		cudaFree(d_output_frame);

	d_frame_buffer = nullptr;
	d_output_frame = nullptr;
}

void TemporalMedianFilter::reset()
{
	current_idx = 0;
	frames_processed = 0;

	size_t frame_size = params.frame_width * params.frame_height;
	size_t buffer_size = frame_size * params.buffer_size;

	cudaMemset(d_frame_buffer, 0, buffer_size * sizeof(uint8_t));
}

cv::Mat TemporalMedianFilter::processFrame(const cv::Mat& input_frame)
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

	temporalMedianKernel<<<gridDim, blockDim>>>(
		d_frame_buffer, d_output_frame, params.frame_width,
		params.frame_height, params.buffer_size, frames_processed + 1);

	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		printf("CUDA Kernel Error: %s\n", cudaGetErrorString(error));
	}

	cv::Mat output_frame(params.frame_height, params.frame_width, CV_8UC1);
	cudaMemcpy(output_frame.data, d_output_frame,
		   frame_size * sizeof(uint8_t), cudaMemcpyDeviceToHost);

	current_idx = (current_idx + 1) % params.buffer_size;
	frames_processed++;

	return output_frame;
}