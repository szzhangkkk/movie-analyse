import unittest
from unittest.mock import patch, MagicMock
from PIL import Image
from io import BytesIO
import numpy as np
from src.poster_analyzer import get_hsv_features
#使用 Mock (模拟) 技术。我们要欺骗程序，让它以为下载了一张图片，实际上是我们塞给它的一个红色的像素点。
class TestPosterAnalyzer(unittest.TestCase):

    def create_dummy_image(self):
        """创建一个 50x50 的纯红色图片用于测试"""
        # RGB: (255, 0, 0) -> 纯红
        img = Image.new('RGB', (50, 50), color=(255, 0, 0))
        img_byte_arr = BytesIO()
        img.save(img_byte_arr, format='JPEG')
        return img_byte_arr.getvalue()

    @patch('src.poster_analyzer.requests.get')
    def test_get_hsv_features(self, mock_get):
        """
        测试颜色提取功能
        @patch 装饰器把 requests.get 变成了模拟对象 mock_get
        """

        # 1. 设定剧本：当代码调用 requests.get 时，返回什么？
        # 我们让它返回状态码 200，内容是我们造的假图片
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = self.create_dummy_image()
        mock_get.return_value = mock_response

        # 2. 运行你的函数 (传入一个假网址)
        result = get_hsv_features('http://fake-url.com/img.jpg')

        # 3. 验证结果
        # 纯红色的 HSV 应该是：H=0 (或接近0), S=1.0, V=1.0
        self.assertIsNotNone(result)

        # 验证亮度 (Brightness)
        # 因为我们造的是最亮的红色，亮度应该是 1.0
        self.assertAlmostEqual(result['Brightness'], 1.0, places=1, msg="亮度提取错误")

        # 验证饱和度 (Saturation)
        self.assertAlmostEqual(result['Saturation'], 1.0, places=1, msg="饱和度提取错误")

        print("✅ 图片分析模块测试通过 (Mock 模拟下载成功)")

    @patch('src.poster_analyzer.requests.get')
    def test_download_failure(self, mock_get):
        """测试：如果下载失败了，程序会崩吗？"""

        # 设定剧本：模拟 404 错误
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        # 运行函数
        result = get_hsv_features('http://bad-url.com')

        # 验证：应该返回 None，而不是报错崩溃
        self.assertIsNone(result, "下载失败时应该返回 None")
        print("✅ 异常处理测试通过")


if __name__ == '__main__':
    unittest.main()