import os
import logging

from src.data_loader import load_data
from src.poster_analyzer import process_images
from src.ml_models import MLProcessor
from src.visualizer import Visualizer
from src.stats_analyzer import StatTester

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

FILE_PATH = os.path.join("data", "imdb_top_1000.csv")


def main():
    logger.info("电影海报视觉全维度分析 开始")

    # 1. 加载数据
    df = load_data(FILE_PATH)
    if df.empty:
        logger.error("数据加载失败，退出")
        return

    # 2. 数据初始概览
    logger.info("[Visual 0] 原始数据基础概览")
    temp_viz = Visualizer(df)
    temp_viz.plot_basic_stats()

    # 3. 深度图像分析
    df_colors = process_images(df, sample_num=1000)
    if df_colors.empty:
        logger.error("海报特征提取失败，退出")
        return

    # 4. 机器学习
    ml = MLProcessor(df_colors)
    result_df, valid_features, rf_model = ml.run_classifier()
    ml.plot_confusion_matrix()

    # 同步 Visual_Type
    df_colors['Visual_Type'] = ml.df['Visual_Type']

    # 5. 统计检验
    logger.info("[Stats] T-test 统计检验")
    tester = StatTester(df_colors)
    tester.t_test('Animation (动画)', 'Thriller_Horror (惊悚/恐怖)', feature='Brightness')

    # 6. 高级可视化
    viz = Visualizer(df_colors)

    logger.info("[Visual 1] 特征相关性热力图")
    viz.plot_correlation_heatmap()

    logger.info("[Visual 2] 核心特征分布 (小提琴图)")
    viz.plot_violin_distribution()

    logger.info("[Visual 3] t-SNE 聚类地图")
    viz.plot_tsne_cluster()

    logger.info("[Visual 4] 巅峰对决: 动画片 vs 惊悚/恐怖")
    viz.plot_comparative_radar('Animation (动画)', 'Thriller_Horror (惊悚/恐怖)')

    logger.info("[Visual 5] 特征重要性排行")
    viz.plot_feature_importance(rf_model, valid_features)

    logger.info("[Visual 6] 错误置信度分析")
    viz.plot_error_scatter(result_df)

    logger.info("所有分析已完成")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("用户中断")
    except Exception:
        logger.exception("运行出错")
