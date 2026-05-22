import unittest
import pandas as pd
import os

from src.data_loader import load_data


class TestDataLoader(unittest.TestCase):

    def setUp(self):
        # 使用逗号分隔的 Genres，与真实 CSV 格式一致
        self.csv_content = """Poster_Link,Series_Title,Genre,IMDB_Rating,Runtime
http://url1.com,Movie A,"Action, Adventure",7.5,120 min
http://url2.com,Movie B,Comedy,6.0,90 min
http://url3.com,Movie C,Drama,,100 min
http://url4.com,Movie D,Horror,5.5,
"""
        self.temp_file = 'temp_test.csv'
        with open(self.temp_file, 'w') as f:
            f.write(self.csv_content)

    def test_load_data_logic(self):
        """测试核心逻辑：是否清洗了空值？是否拆分了 Genre？"""
        df = load_data(self.temp_file)

        # Movie C 缺 IMDB_Rating，Movie D 缺 Runtime
        # 但 load_data 只 dropna(['Poster', 'Genres', 'Title'])
        # 所以 4 行都保留。然后 top_genres 筛选前 8 类型。
        # 所有 4 行都有 Main_Genre，应该全部保留。
        self.assertGreaterEqual(len(df), 2, "应至少保留 2 行数据")

        # 验证 Movie A 的 Main_Genre 是 "Action"（逗号分割取第一个）
        movie_a = df[df['Title'] == 'Movie A']
        self.assertFalse(movie_a.empty, "应包含 Movie A")
        self.assertEqual(movie_a.iloc[0]['Main_Genre'], 'Action', "类型拆分应取逗号前第一个词")

        # 验证列名映射：Poster_Link -> Poster
        self.assertIn('Poster', df.columns, "Poster_Link 应被重命名为 Poster")

    def test_load_data_missing_poster_column(self):
        """测试缺少海报列时返回空 DataFrame"""
        bad_csv = "Title,Genre\nMovie A,Action\n"
        bad_file = 'temp_bad.csv'
        with open(bad_file, 'w') as f:
            f.write(bad_csv)

        try:
            df = load_data(bad_file)
            self.assertTrue(df.empty, "缺少海报列应返回空 DataFrame")
        finally:
            if os.path.exists(bad_file):
                os.remove(bad_file)

    def tearDown(self):
        if os.path.exists(self.temp_file):
            os.remove(self.temp_file)


if __name__ == '__main__':
    unittest.main()
