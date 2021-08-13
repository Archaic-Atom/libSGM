# -*- coding: utf-8 -*-
import ctypes
import numpy as np
import cv2
from multiprocessing.sharedctypes import RawArray


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

    @staticmethod
    def _covert_disparity(disp_img_array: RawArray, height: int, width: int) -> np.array:
        disp_img = np.frombuffer(disp_img_array, dtype=np.uint16)
        disp_img = disp_img.astype(np.float32)
        disp_img = disp_img.reshape((height, width))
        return disp_img

    def inference(self, left_img_path: str, right_img_path: str, disp_num: str,
                  height: int, width: int) -> tuple:
        uint16_2_byte = 2
        left_img_path_c = ctypes.create_string_buffer(left_img_path.encode('utf-8'))
        right_img_path_c = ctypes.create_string_buffer(right_img_path.encode('utf-8'))
        disp_img = RawArray(ctypes.c_ubyte, height * width * uint16_2_byte)
        ptr_disp_img = ctypes.byref(disp_img)

        self._sgm_module.c_sgm_interface.restype = ctypes.c_double

        # left_img
        fps = self._sgm_module.c_sgm_interface(left_img_path_c,
                                               right_img_path_c,
                                               disp_num,
                                               ptr_disp_img,
                                               False)
        left_disp_img = self._covert_disparity(disp_img, height, width)

        fps = self._sgm_module.c_sgm_interface(left_img_path_c,
                                               right_img_path_c,
                                               disp_num,
                                               ptr_disp_img,
                                               True)
        right_disp_img = self._covert_disparity(disp_img, height, width)

        return fps, left_disp_img, right_disp_img


def main():
    lib_path = '../../build/sample/sgm_interface_fast_system/libsgm_interface_fast_system.so'
    right_img_path = '../../example/000000_10_r.png'
    left_img_path = '../../example/000000_10_l.png'

    height = 370
    width = 1226

    sgm = CSgmInterface(lib_path)

    while True:
        fps, left_disp_img, right_disp_img = sgm.inference(left_img_path, right_img_path, 256, height, width)
        print(fps)
        # print(disp_img.shape)
        cv2.imwrite('1.png', left_disp_img)
        cv2.imwrite('2.png', right_disp_img)


if __name__ == '__main__':
    main()
