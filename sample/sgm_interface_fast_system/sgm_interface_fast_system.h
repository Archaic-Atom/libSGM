#ifndef SGM_INTERFACE_FAST_SYSTEM_H
#define SGM_INTERFACE_FAST_SYSTEM_H
#include "define.h"

template <class... Args>
static std::string format_string(const char* fmt, Args... args);

extern "C"
{
	double c_sgm_interface(char* left_path, char* right_path, int disp_size, unsigned char* disp_data, bool is_right);
}



class SgmInterface
{
public:
	static SgmInterface* CreateSgmInterface(int disp_size);
	static void DeleteSgmInterface();
	inline void SetDispSize(int disp_size) {m_iDispSize = disp_size;}
	const double Inference(char* left_path, char* right_path, unsigned char* disp_data, bool is_right);

protected:
	cv::Mat ReadImg(char* path);
	const double SgmInference(cv::Mat left_img_mat, cv::Mat right_img_mat, unsigned char* disp_data, bool is_right);

	SgmInterface(int disp_size);
	virtual ~SgmInterface();

private:
	static SgmInterface* m_pHandler;
	int m_iDispSize;
};



#endif