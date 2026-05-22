import logging

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd

logger = logging.getLogger(__name__)


class StatTester:
    def __init__(self, df):
        self.df = df
        sns.set(style="whitegrid")

    def t_test(self, genre_a, genre_b, feature='Brightness'):
        """进行独立样本 T 检验，并绘制箱线图可视化结果"""
        logger.info("正在对比 %s 和 %s 的 %s", genre_a, genre_b, feature)

        group_a = self.df[self.df['Main_Genre'] == genre_a][feature]
        group_b = self.df[self.df['Main_Genre'] == genre_b][feature]

        t_stat, p_val = stats.ttest_ind(group_a, group_b, equal_var=False)

        logger.info("%s 均值: %.3f", genre_a, group_a.mean())
        logger.info("%s 均值: %.3f", genre_b, group_b.mean())
        logger.info("P-value: %.5f", p_val)

        if p_val < 0.05:
            logger.info("差异显著 (Significant)")
            title_suffix = "(Significant Difference)"
        else:
            logger.info("差异不显著")
            title_suffix = "(No Significant Difference)"

        self._plot_comparison(genre_a, genre_b, feature, title_suffix)

    def _plot_comparison(self, genre_a, genre_b, feature, title_suffix):
        """绘制对比箱线图"""
        plt.figure(figsize=(8, 6))

        plot_data = self.df[self.df['Main_Genre'].isin([genre_a, genre_b])]

        sns.boxplot(
            data=plot_data,
            x='Main_Genre',
            y=feature,
            hue='Main_Genre',
            palette="Set2",
            width=0.5,
            legend=False,
        )

        sns.stripplot(
            data=plot_data,
            x='Main_Genre',
            y=feature,
            color='black',
            alpha=0.5,
            jitter=True,
            legend=False,
        )

        plt.title(f'T-Test: {genre_a} vs {genre_b} - {feature}\n{title_suffix}', fontsize=14)
        plt.ylabel(feature, fontsize=12)
        plt.xlabel('Movie Genre', fontsize=12)

        plt.show()
