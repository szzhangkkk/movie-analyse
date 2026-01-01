import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from math import pi
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler


class Visualizer:
    def __init__(self, df):
        self.df = df
        sns.set(style="whitegrid")
        # 字体设置
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'Malgun Gothic']
        plt.rcParams['axes.unicode_minus'] = False
        self.palette = sns.color_palette("Set2")

    def plot_basic_stats(self):
        """基础概览图"""
        print("    正在绘制数据初始分布概览...")
        fig = plt.figure(figsize=(16, 14))

        ax1 = fig.add_subplot(2, 2, 1)
        top_genres = self.df['Main_Genre'].value_counts().head(10)
        sns.barplot(x=top_genres.values, y=top_genres.index, palette='viridis', ax=ax1)
        ax1.set_title('Top 10 电影类型数量', fontsize=14)

        ax2 = fig.add_subplot(2, 2, 2)
        sns.histplot(self.df['IMDB Rating'], bins=20, kde=True, color='orange', ax=ax2)
        ax2.set_title('IMDB 评分分布', fontsize=14)

        ax3 = fig.add_subplot(2, 2, 3)
        if 'Released_Year' in self.df.columns:
            years = pd.to_numeric(self.df['Released_Year'], errors='coerce').dropna()
            sns.histplot(years, bins=30, color='teal', ax=ax3)
            ax3.set_title('电影年代分布', fontsize=14)
        else:
            ax3.text(0.5, 0.5, '暂无年份数据', ha='center')

        ax4 = fig.add_subplot(2, 2, 4)
        if 'Length in Min' in self.df.columns:
            sns.scatterplot(x='Length in Min', y='IMDB Rating', data=self.df,
                            hue='Main_Genre', alpha=0.6, legend=False, ax=ax4)
            ax4.set_title('时长与评分关系', fontsize=14)
        else:
            ax4.text(0.5, 0.5, '暂无时长数据', ha='center')

        plt.suptitle('原始数据基础概览', fontsize=22, y=0.98)
        # 强制增加间距
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        plt.subplots_adjust(hspace=0.3, wspace=0.2)
        plt.show()

    def plot_violin_distribution(self):
        """小提琴图"""
        print("    正在绘制特征分布小提琴图...")
        key_features = ['Vibrancy_Ratio', 'Edge_Density', 'Face_Ratio']
        labels = ['鲜艳度', '纹理密度', '人脸占比']

        if 'Visual_Type' not in self.df.columns: return
        plot_df = self.df[self.df['Visual_Type'] != 'Other']

        plt.figure(figsize=(16, 9))  # 加高画布
        for i, feature in enumerate(key_features):
            if feature not in plot_df.columns: continue

            plt.subplot(1, 3, i + 1)
            sns.violinplot(x='Visual_Type', y=feature, hue='Visual_Type', data=plot_df,
                           palette="Set3", inner="quartile", legend=False)

            plt.title(f'{labels[i]}分布', fontsize=14, pad=15)
            # 标签倾斜度加大，防止重叠
            plt.xticks(rotation=45, ha='right', fontsize=11)
            plt.xlabel('')
            plt.ylabel(feature)

        plt.suptitle('核心视觉特征分布', fontsize=20, y=0.98)
        # 底部留白加倍
        plt.tight_layout(rect=[0, 0.15, 1, 0.95])
        plt.subplots_adjust(bottom=0.25, wspace=0.2)
        plt.show()

    def plot_correlation_heatmap(self):
        """【已修复】热力图：完整显示正方形，修复切边问题"""
        print("    正在绘制特征相关性热力图...")
        features = ['Hue_1', 'Sat_1', 'Val_1', 'Vibrancy_Ratio', 'Warm_Rating',
                    'Edge_Density', 'Entropy', 'Face_Ratio', 'Rule_of_Thirds_Score']
        valid_features = [f for f in features if f in self.df.columns]

        corr = self.df[valid_features].corr()

        # 增加高度，给上下标签留空间
        plt.figure(figsize=(12, 11))

        # ❌ 去掉遮罩 (mask=None)，显示完整正方形，这样第一行和最后一行都不会缺
        ax = sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm',
                         vmax=1, vmin=-1, square=True, linewidths=.5,
                         annot_kws={"size": 10}, cbar_kws={"shrink": 0.8})

        # ✅ 强制修复 matplotlib 切边 bug (手动设置 y 轴范围)
        # 有些版本如果不加这句，第一行和最后一行只显示一半
        ax.set_ylim(len(corr), 0)

        plt.title('视觉特征相关性矩阵', fontsize=18, pad=20)

        # x轴标签倾斜，防止重叠
        plt.xticks(rotation=45, ha='right', fontsize=11)
        plt.yticks(rotation=0, fontsize=11)

        # 上下左右留白
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.subplots_adjust(bottom=0.15, top=0.93)
        plt.show()

    def plot_error_scatter(self, result_df):
        """误差散点图"""
        print("    正在绘制误差分析散点图...")
        fig, ax = plt.subplots(figsize=(14, 9))

        correct = result_df[result_df['Is_Correct'] == True]
        wrong = result_df[result_df['Is_Correct'] == False]

        ax.scatter(correct['Vibrancy_Ratio'], correct['Edge_Density'],
                   c='green', alpha=0.3, s=30, label='预测正确')
        ax.scatter(wrong['Vibrancy_Ratio'], wrong['Edge_Density'],
                   c='red', alpha=0.8, s=wrong['Confidence'] * 150, edgecolors='black', label='预测错误')

        ax.set_title('AI 误判分析图', fontsize=18, pad=20)
        ax.set_xlabel('鲜艳度', fontsize=12)
        ax.set_ylabel('纹理密度', fontsize=12)

        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
        plt.subplots_adjust(right=0.8, left=0.1, top=0.9, bottom=0.1)
        plt.show()

    def plot_feature_importance(self, model, feature_names):
        """特征重要性"""
        print("    正在绘制特征重要性排行...")
        if model is None or not hasattr(model, 'feature_importances_'): return

        importances = model.feature_importances_
        indices = np.argsort(importances)

        plt.figure(figsize=(12, 8))
        plt.title('特征重要性排行', fontsize=16, pad=20)
        plt.barh(range(len(indices)), importances[indices], color='purple', align='center', alpha=0.7)
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices], fontsize=12)
        plt.xlabel('相对重要性')

        plt.tight_layout()
        # 左边留白给长标签
        plt.subplots_adjust(left=0.2, top=0.9)
        plt.show()

    def plot_tsne_cluster(self):
        """t-SNE"""
        print("    正在绘制 t-SNE 聚类地图...")
        try:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            drop_cols = ['id', 'vote_average', 'vote_count', 'popularity', 'budget', 'revenue', 'runtime']
            features = [c for c in numeric_cols if c not in drop_cols]
            X = self.df[features].fillna(0)

            tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
            X_embedded = tsne.fit_transform(X)

            fig, ax = plt.subplots(figsize=(14, 10))
            hue_col = 'Visual_Type' if 'Visual_Type' in self.df.columns else 'Main_Genre'

            counts = self.df[hue_col].value_counts()
            valid_hues = counts[counts > 10].index
            mask = self.df[hue_col].isin(valid_hues)

            sns.scatterplot(x=X_embedded[mask, 0], y=X_embedded[mask, 1],
                            hue=self.df.loc[mask, hue_col],
                            palette="tab10", s=80, alpha=0.8, edgecolor='w', ax=ax)

            ax.set_title('电影视觉风格 t-SNE 聚类地图', fontsize=18, pad=20)
            ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., fontsize=11)
            plt.subplots_adjust(right=0.75, top=0.9)
            plt.show()
        except Exception as e:
            print(f"t-SNE 绘制跳过: {e}")

    def plot_comparative_radar(self, genre_a, genre_b):
        """雷达图"""
        print(f"    正在绘制对比雷达图: {genre_a} vs {genre_b}...")
        features = ['Brightness', 'Vibrancy_Ratio', 'Warm_Rating', 'Edge_Density', 'Entropy', 'Face_Ratio',
                    'Rule_of_Thirds_Score']
        feature_names = ['亮度', '鲜艳度', '暖色调', '纹理密度', '复杂度', '人脸占比', '构图分']
        valid_features = [f for f in features if f in self.df.columns]
        if not valid_features: return

        scaler = MinMaxScaler()
        df_scaled = self.df.copy()
        df_scaled[valid_features] = scaler.fit_transform(df_scaled[valid_features])

        col = 'Visual_Type' if 'Visual_Type' in self.df.columns else 'Main_Genre'
        mean_a = df_scaled[df_scaled[col] == genre_a][valid_features].mean()
        mean_b = df_scaled[df_scaled[col] == genre_b][valid_features].mean()

        if mean_a.isna().any() or mean_b.isna().any(): return

        N = len(valid_features)
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': 'polar'})

        val_a = mean_a.tolist() + mean_a.tolist()[:1]
        ax.plot(angles, val_a, linewidth=2, linestyle='solid', label=genre_a, color='blue')
        ax.fill(angles, val_a, alpha=0.1, color='blue')

        val_b = mean_b.tolist() + mean_b.tolist()[:1]
        ax.plot(angles, val_b, linewidth=2, linestyle='solid', label=genre_b, color='red')
        ax.fill(angles, val_b, alpha=0.1, color='red')

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(feature_names[:len(valid_features)], fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.set_title(f'巅峰对决: {genre_a} vs {genre_b}', fontsize=16, y=1.08)

        ax.legend(bbox_to_anchor=(1.15, 1), loc='upper left')
        plt.subplots_adjust(right=0.75, top=0.85)
        plt.show()