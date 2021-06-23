/*
Copyright 2016 Fixstars Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http ://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include "sgm_interface_fast_system.h"


SgmInterface* SgmInterface::m_pHandler = nullptr;

template <class... Args>
static std::string format_string(const char* fmt, Args... args)
{
	const int BUF_SIZE = 1024;
	char buf[BUF_SIZE];
	std::snprintf(buf, BUF_SIZE, fmt, args...);
	return std::string(buf);
}


SgmInterface::SgmInterface(int disp_size)
:m_iDispSize(disp_size)
{
}

SgmInterface::~SgmInterface()
{
}


SgmInterface* SgmInterface::CreateSgmInterface(int disp_size)
{
	if (nullptr != m_pHandler)
		return m_pHandler;

	m_pHandler = new SgmInterface(disp_size);
	return m_pHandler;
}

void SgmInterface::DeleteSgmInterface()
{
	if (nullptr != m_pHandler)
	{
		delete m_pHandler;
		m_pHandler = nullptr;
	}
}

cv::Mat SgmInterface::ReadImg(char* path)
{
	const int first_frame = 1;
	cv::Mat img_mat = cv::imread(format_string(path, first_frame), CV_LOAD_IMAGE_COLOR);
	ASSERT_MSG(!img_mat.empty(), "imread failed.");

	if(img_mat.type() != CV_8U)
		cv::cvtColor(img_mat,img_mat,CV_BGR2GRAY);

	ASSERT_MSG(img_mat.type() == CV_8U || img_mat.type() == CV_16U, "input image format must be CV_8U or CV_16U.");
	return img_mat;
}

const double SgmInterface::Inference(char* left_path, char* right_path, unsigned char* disp_data)
{
	cv::Mat img_mat_1 = ReadImg(left_path);
	cv::Mat img_mat_2 = ReadImg(right_path);

	ASSERT_MSG(img_mat_1.size() == img_mat_2.size() 
		&& img_mat_1.type() == img_mat_2.type(), 
		"input images must be same size and type.");

	const int disp_size = m_iDispSize;
	ASSERT_MSG(disp_size == 64 || disp_size == 128 || disp_size == 256, 
		"disparity size must be 64, 128 or 256.");

	const int width = img_mat_1.cols;
	const int height = img_mat_1.rows;

	const int input_depth = img_mat_1.type() == CV_8U ? 8 : 16;
	const int input_bytes = input_depth * width * height / 8;
	const int output_depth = disp_size < 256 ? 8 : 16;
	const int output_bytes = output_depth * width * height / 8;

	sgm::StereoSGM sgm(width, height, disp_size, input_depth, output_depth, sgm::EXECUTE_INOUT_CUDA2CUDA);

	const int invalid_disp = output_depth == 8
			? static_cast<uint8_t>(sgm.get_invalid_disparity())
			: static_cast<uint16_t>(sgm.get_invalid_disparity());

	cv::Mat disparity(height, width, output_depth == 8 ? CV_8U : CV_16U, (void *) disp_data);
	// cv::Mat disparity_8u;

	device_buffer d_I1(input_bytes), d_I2(input_bytes), d_disparity(output_bytes);


	cudaMemcpy(d_I1.data, img_mat_1.data, input_bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_I2.data, img_mat_2.data, input_bytes, cudaMemcpyHostToDevice);

	const auto t1 = std::chrono::system_clock::now();

	sgm.execute(d_I1.data, d_I2.data, d_disparity.data);
	cudaDeviceSynchronize();

	const auto t2 = std::chrono::system_clock::now();
	const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
	const double fps = 1e6 / duration;

	cudaMemcpy(disparity.data, d_disparity.data, output_bytes, cudaMemcpyDeviceToHost);
	//disparity.convertTo(disparity_8u, CV_8U, 255. / disp_size);

	// cv::imwrite("1.png", disparity);

	return fps;
}


double c_sgm_interface(char* left_path, char* right_path, int disp_size, unsigned char* disp_data)
{
	double res = 0;
	SgmInterface* sgm_interface = SgmInterface::CreateSgmInterface(disp_size);
	res = sgm_interface->Inference(left_path, right_path, disp_data);
	SgmInterface::DeleteSgmInterface();
	return res;
}


/************************************************************************************************************
int main(int argc, char* argv[])
{
	if (argc < 3) {
		std::cout << "usage: " << argv[0] << " left-image-format right-image-format [disp_size]" << std::endl;
		std::exit(EXIT_FAILURE);
	}

	const int first_frame = 1;

	cv::Mat I1 = cv::imread(format_string(argv[1], first_frame), -1);
	cv::Mat I2 = cv::imread(format_string(argv[2], first_frame), -1);
	const int disp_size = argc >= 4 ? std::stoi(argv[3]) : 128;

	ASSERT_MSG(!I1.empty() && !I2.empty(), "imread failed.");
	ASSERT_MSG(I1.size() == I2.size() && I1.type() == I2.type(), "input images must be same size and type.");
	ASSERT_MSG(I1.type() == CV_8U || I1.type() == CV_16U, "input image format must be CV_8U or CV_16U.");
	ASSERT_MSG(disp_size == 64 || disp_size == 128 || disp_size == 256, "disparity size must be 64, 128 or 256.");

	const int width = I1.cols;
	const int height = I1.rows;

	const int input_depth = I1.type() == CV_8U ? 8 : 16;
	const int input_bytes = input_depth * width * height / 8;
	const int output_depth = disp_size < 256 ? 8 : 16;
	const int output_bytes = output_depth * width * height / 8;

	sgm::StereoSGM sgm(width, height, disp_size, input_depth, output_depth, sgm::EXECUTE_INOUT_CUDA2CUDA);

	const int invalid_disp = output_depth == 8
			? static_cast< uint8_t>(sgm.get_invalid_disparity())
			: static_cast<uint16_t>(sgm.get_invalid_disparity());

	cv::Mat disparity(height, width, output_depth == 8 ? CV_8U : CV_16U);
	cv::Mat disparity_8u;

	device_buffer d_I1(input_bytes), d_I2(input_bytes), d_disparity(output_bytes);

	while(true){
	int frame_no = first_frame;
	I1 = cv::imread(format_string(argv[1], frame_no), -1);
	I2 = cv::imread(format_string(argv[2], frame_no), -1);
	if (I1.empty() || I2.empty()) {
			frame_no = first_frame;
			return -1;
		}

	cudaMemcpy(d_I1.data, I1.data, input_bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_I2.data, I2.data, input_bytes, cudaMemcpyHostToDevice);

	const auto t1 = std::chrono::system_clock::now();

	sgm.execute(d_I1.data, d_I2.data, d_disparity.data);
	cudaDeviceSynchronize();

	const auto t2 = std::chrono::system_clock::now();
	const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
	const double fps = 1e6 / duration;

	cudaMemcpy(disparity.data, d_disparity.data, output_bytes, cudaMemcpyDeviceToHost);
	disparity.convertTo(disparity_8u, CV_8U, 255. / disp_size);

	//cv::imwrite("1.png", disparity_8u);
	std::cout << fps << std::endl;
}
}
*******************************************************************************************************/