import unittest
import pandas as pd
from io import StringIO
import os  # 修正点：之前这里拼写成了 oss，现在已改回 os
from src.data_loader import load_data


class TestDataLoader(unittest.TestCase):

    def setUp(self):
        """
        每次运行测试前，都会先执行这个 setup
        我们创建一个假的 CSV 文件内容
        """
        # 注意：Movie C 缺评分，Movie D 缺时长
        self.csv_content = """Title,Genres,Poster,IMDB Rating,Length in Min
Movie A,Action|Adventure,http://url1.com,7.5,120
Movie B,Comedy,http://url2.com,6.0,90
Movie C,Drama,http://url3.com,,100
Movie D,Horror,http://url4.com,5.5,
"""

    def test_load_data_logic(self):
        """测试核心逻辑：是否清洗了空值？是否拆分了 Genre？"""

        # 1. 保存临时文件
        with open('temp_test.csv', 'w') as f:
            f.write(self.csv_content)

        # 2. 运行你的代码
        df = load_data('temp_test.csv')

        # 3. 断言验证
        # Movie C 缺 IMDB Rating -> 应该被删
        # Movie D 缺 Length in Min -> 应该被删
        # 所以应该只剩下 Movie A 和 Movie B (共2行)
        self.assertEqual(len(df), 2, "清洗逻辑错误：应该只剩下 2 行完整数据")

        # 验证类型拆分：Movie A 的 Genres 是 "Action|Adventure"，Main_Genre 应该是 "Action"
        self.assertEqual(df.iloc[0]['Main_Genre'], 'Action', "类型拆分逻辑错误")

        print("✅ 数据加载模块测试通过")

    def tearDown(self):
        """测试结束后清理垃圾文件"""
        # 修正点：这里正确使用了 os 模块
        if os.path.exists('temp_test.csv'):
            os.remove('temp_test.csv')


if __name__ == '__main__':
    unittest.main()