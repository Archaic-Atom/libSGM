# -*- coding: utf-8 -*-
import ctypes
import numpy as np
import cv2


def add_module(lib_path: str) -> object:
    return ctypes.cdll.LoadLibrary(lib_path)


class CSgmInterface(object):
    """docstring for """
    __C_SGM_INTERFACE = None

    def __init__(self, lib_path: str):
        super().__init__()
        self._lib_path = lib_path
        self._sgm_module = self._load_module(lib_path)

    def __new__(cls, *args: str, **kwargs: str) -> object:
        if cls.__C_SGM_INTERFACE is None:
            cls.__C_SGM_INTERFACE = object.__new__(cls)
        return cls.__C_SGM_INTERFACE

    @staticmethod
    def _load_module(lib_path: str) -> object:
        return ctypes.cdll.LoadLibrary(lib_path)

    def inference(self, left_img_path: str, right_img_path: str, disp_num: str):
        left_img_path_c = ctypes.create_string_buffer(left_img_path.encode('utf-8'))
        right_img_path_c = ctypes.create_string_buffer(right_img_path.encode('utf-8'))
        self._sgm_module.c_sgm_interface.restype = ctypes.c_double
        height = 370
        width = 1226
        disp_img = np.zeros(dtype=np.uint16, shape=(height, width))
        fps = self._sgm_module.c_sgm_interface(left_img_path_c, right_img_path_c,
                                               disp_num,
                                               disp_img.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)))

        return fps, disp_img


def main():
    lib_path = '../../build/sample/sgm_interface_fast_system/libsgm_interface_fast_system.so'
    left_img_path = '../../example/000000_10_l.png'
    right_img_path = '../../example/000000_10_r.png'

    height = 370
    width = 1226

    sgm = CSgmInterface(lib_path)

    while True:
        fps, disp_img = sgm.inference(left_img_path, right_img_path, 256)
        print(fps)
        # print(disp_img.shape)
        # cv2.imwrite('2.png', disp_img)


if __name__ == '__main__':
    main()
