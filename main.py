from src.data_loader import load_data
from src.poster_analyzer import process_images
from src.ml_models import MLProcessor
from src.visualizer import Visualizer

FILE_PATH = r"data/imdb_top_1000.csv"


def main():
    print("=== 🎬 电影海报视觉全维度分析 (完美图表版) ===")

    # 1. 加载数据
    df = load_data(FILE_PATH)
    if df.empty: return

    # 2. 【新增】数据初始表现 (在开始复杂分析前，先看原始数据)
    # 我们先临时创建一个 viz 对象来画基础图
    print("\n[Visual 0] 原始数据基础概览...")
    temp_viz = Visualizer(df)
    temp_viz.plot_basic_stats()  # 👈 这里调用新图表

    # 3. 深度图像分析
    df_colors = process_images(df, sample_num=1000)
    if df_colors.empty: return

    # 4. 机器学习
    ml = MLProcessor(df_colors)
    result_df, valid_features, rf_model = ml.run_classifier()

    # 把 Visual_Type 同步回 df_colors
    df_colors['Visual_Type'] = ml.df['Visual_Type']

    # 5. 高级可视化
    viz = Visualizer(df_colors)

    print("\n[Visual 1] 特征相关性热力图...")
    viz.plot_correlation_heatmap()

    print("\n[Visual 2] 核心特征分布 (小提琴图)...")
    viz.plot_violin_distribution()

    print("\n[Visual 3] t-SNE 聚类地图...")
    viz.plot_tsne_cluster()

    print("\n[Visual 4] 巅峰对决: 动画片 vs 惊悚/恐怖...")
    viz.plot_comparative_radar('Animation (动画)', 'Thriller_Horror (惊悚/恐怖)')

    print("\n[Visual 5] 特征重要性排行...")
    viz.plot_feature_importance(rf_model, valid_features)

    print("\n[Visual 6] 错误置信度分析...")
    viz.plot_error_scatter(result_df)

    print("\n=== ✅ 所有分析已完成 ===")


if __name__ == "__main__":
    main()