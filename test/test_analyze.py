import unittest
from unittest.mock import patch, MagicMock
from PIL import Image
from io import BytesIO
import numpy as np
import cv2

from src.poster_analyzer import get_visual_features, process_single_row


class TestPosterAnalyzer(unittest.TestCase):

    def create_dummy_image_pil(self):
        """创建一个 50x50 的纯红色 PIL 图片"""
        return Image.new('RGB', (50, 50), color=(255, 0, 0))

    def create_dummy_image_bytes(self):
        """创建一个 50x50 的纯红色图片的字节数据"""
        img = Image.new('RGB', (50, 50), color=(255, 0, 0))
        buf = BytesIO()
        img.save(buf, format='JPEG')
        return buf.getvalue()

    def test_get_visual_features_returns_all_keys(self):
        """测试 get_visual_features 返回所有预期的特征键"""
        img_pil = self.create_dummy_image_pil()
        img_cv2 = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        features = get_visual_features(img_pil, img_cv2)

        self.assertIsNotNone(features)
        expected_keys = [
            'Hue_1', 'Sat_1', 'Val_1', 'Vibrancy_Ratio', 'Warm_Rating',
            'Edge_Density', 'Entropy', 'Text_Texture_Ratio',
            'Rule_of_Thirds_Score', 'Face_Count', 'Face_Ratio',
        ]
        for key in expected_keys:
            self.assertIn(key, features, f"缺少特征: {key}")

    def test_get_visual_features_red_image_values(self):
        """纯红色图片的 HSV 特征值应合理"""
        img_pil = self.create_dummy_image_pil()
        img_cv2 = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        features = get_visual_features(img_pil, img_cv2)

        # 纯红色：Hue 接近 0, Sat 接近 1, Val 接近 1
        self.assertAlmostEqual(features['Hue_1'], 0.0, places=1)
        self.assertAlmostEqual(features['Sat_1'], 1.0, places=1)
        self.assertAlmostEqual(features['Val_1'], 1.0, places=1)

    def test_get_visual_features_edge_density_non_negative(self):
        """边缘密度应为非负值"""
        img_pil = self.create_dummy_image_pil()
        img_cv2 = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        features = get_visual_features(img_pil, img_cv2)

        self.assertGreaterEqual(features['Edge_Density'], 0)
        self.assertGreaterEqual(features['Entropy'], 0)

    @patch('src.poster_analyzer.requests.get')
    def test_process_single_row_success(self, mock_get):
        """测试 process_single_row 正常下载并提取特征"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = self.create_dummy_image_bytes()
        mock_get.return_value = mock_response

        row = {'Poster': 'http://fake-url.com/img.jpg', 'Main_Genre': 'Drama'}
        result = process_single_row(row)

        self.assertIsNotNone(result)
        self.assertIn('Hue_1', result)
        self.assertIn('Edge_Density', result)

    @patch('src.poster_analyzer.requests.get')
    def test_process_single_row_download_failure(self, mock_get):
        """测试下载失败时返回 None"""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        row = {'Poster': 'http://bad-url.com/img.jpg', 'Main_Genre': 'Drama'}
        result = process_single_row(row)

        self.assertIsNone(result)

    def test_process_single_row_no_poster(self):
        """测试没有海报 URL 时返回 None"""
        row = {'Poster': None, 'Main_Genre': 'Drama'}
        result = process_single_row(row)
        self.assertIsNone(result)

        row2 = {'Poster': float('nan'), 'Main_Genre': 'Drama'}
        result2 = process_single_row(row2)
        self.assertIsNone(result2)


if __name__ == '__main__':
    unittest.main()
