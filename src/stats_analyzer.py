import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd


class StatTester:
    def __init__(self, df):
        self.df = df
        # 设置绘图风格
        sns.set(style="whitegrid")

    def t_test(self, genre_a, genre_b, feature='Brightness'):
        """
        进行独立样本 T 检验，并绘制箱线图可视化结果
        """
        print(f"\n[统计检验] 正在对比 {genre_a} 和 {genre_b} 的 {feature}...")

        # 1. 数据提取
        group_a = self.df[self.df['Main_Genre'] == genre_a][feature]
        group_b = self.df[self.df['Main_Genre'] == genre_b][feature]

        # 2. 计算统计量
        t_stat, p_val = stats.ttest_ind(group_a, group_b, equal_var=False)

        # 3. 打印文字结果
        print(f"    --> {genre_a} 均值: {group_a.mean():.3f}")
        print(f"    --> {genre_b} 均值: {group_b.mean():.3f}")
        print(f"    --> P-value: {p_val:.5f}")

        if p_val < 0.05:
            print("    ✅ 结论: 差异显著 (Significant)！海报风格确实不同。")
            title_suffix = "(Significant Difference)"
        else:
            print("    ❌ 结论: 差异不显著，可能是巧合。")
            title_suffix = "(No Significant Difference)"

        # 4. 可视化
        self._plot_comparison(genre_a, genre_b, feature, title_suffix)

    def _plot_comparison(self, genre_a, genre_b, feature, title_suffix):
        """
        内部辅助函数：专门负责画图
        """
        plt.figure(figsize=(8, 6))

        # 筛选数据
        plot_data = self.df[self.df['Main_Genre'].isin([genre_a, genre_b])]

        # 绘制箱线图 (Box Plot)
        # 🟢 修正点在这里：
        # 1. 新增 hue='Main_Genre' (明确指定颜色跟随类型变化)
        # 2. 新增 legend=False (不需要图例，因为X轴已经写了类型)
        sns.boxplot(
            data=plot_data,
            x='Main_Genre',
            y=feature,
            hue='Main_Genre',  # <--- 修正点 1
            palette="Set2",
            width=0.5,
            legend=False  # <--- 修正点 2
        )

        # 加上抖动散点图 (Strip Plot)
        sns.stripplot(
            data=plot_data,
            x='Main_Genre',
            y=feature,
            color='black',
            alpha=0.5,
            jitter=True,
            legend=False  # 这里最好也加上 legend=False 以防万一，虽然通常不需要
        )

        # 设置标题和标签
        plt.title(f'T-Test Result: {genre_a} vs {genre_b} - {feature}\n{title_suffix}', fontsize=14)
        plt.ylabel(feature, fontsize=12)
        plt.xlabel('Movie Genre', fontsize=12)

        plt.show()


#  --- 测试用例 (保持不变) ---
# if __name__ == "__main__":
#     data = {
#         'Main_Genre': ['Horror'] * 50 + ['Comedy'] * 50,
#         'Brightness': list(stats.norm.rvs(loc=0.3, scale=0.1, size=50)) +
#                       list(stats.norm.rvs(loc=0.65, scale=0.1, size=50))
#     }
#     df_fake = pd.DataFrame(data)
#
#     tester = StatTester(df_fake)
#     tester.t_test('Horror', 'Comedy', feature='Brightness')