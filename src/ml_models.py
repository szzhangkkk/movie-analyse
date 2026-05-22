import logging

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.utils import resample
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

logger = logging.getLogger(__name__)


def map_visual_genre(row):
    """将电影类型映射为视觉流派"""
    genres = str(row.get('Genres', row['Main_Genre']))
    if 'Animation' in genres:
        return 'Animation (动画)'
    elif 'Horror' in genres or 'Thriller' in genres:
        return 'Thriller_Horror (惊悚/恐怖)'
    elif 'Sci-Fi' in genres:
        return 'Sci-Fi (科幻)'
    elif 'Action' in genres or 'Crime' in genres or 'Adventure' in genres:
        return 'Action_Adventure (动作/冒险)'
    elif 'Comedy' in genres:
        return 'Comedy (喜剧)'
    elif 'Drama' in genres or 'Biography' in genres or 'Romance' in genres:
        return 'Drama_Romance (剧情/情感)'
    else:
        return 'Other'


class MLProcessor:
    def __init__(self, df):
        self.df = df.copy()
        self._cm_data = None

    def run_classifier(self):
        logger.info("正在进行智能视觉流派分类")

        self.df['Visual_Type'] = self.df.apply(map_visual_genre, axis=1)

        counts = self.df['Visual_Type'].value_counts()
        valid_types = counts[counts > 20].index
        df_clean = self.df[self.df['Visual_Type'].isin(valid_types)].copy()

        logger.info("分析流派: %s", list(valid_types))

        # 上采样平衡
        max_size = df_clean['Visual_Type'].value_counts().max()
        df_balanced_list = []
        for g in df_clean['Visual_Type'].unique():
            df_g = df_clean[df_clean['Visual_Type'] == g]
            df_g_upsampled = resample(df_g, replace=True, n_samples=max_size, random_state=42)
            df_balanced_list.append(df_g_upsampled)
        df_balanced = pd.concat(df_balanced_list)

        features = [
            'Hue_1', 'Sat_1', 'Val_1', 'Color_Ratio_1', 'Vibrancy_Ratio', 'Warm_Rating',
            'Edge_Density', 'Entropy', 'Text_Texture_Ratio',
            'Face_Count', 'Face_Ratio', 'Rule_of_Thirds_Score'
        ]
        valid_features = [f for f in features if f in df_balanced.columns]
        X = df_balanced[valid_features]
        y = df_balanced['Visual_Type']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # 训练投票分类器
        clf1 = RandomForestClassifier(n_estimators=300, max_depth=20, random_state=42)
        clf2 = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
        eclf = VotingClassifier(estimators=[('rf', clf1), ('gb', clf2)], voting='soft')
        eclf.fit(X_train, y_train)

        y_pred = eclf.predict(X_test)
        y_proba = eclf.predict_proba(X_test)
        acc = accuracy_score(y_test, y_pred)
        logger.info("模型准确率: %.2f", acc)

        # 保存混淆矩阵数据供后续绘图
        labels = sorted(df_balanced['Visual_Type'].unique())
        cm = confusion_matrix(y_test, y_pred, labels=labels, normalize='true')
        self._cm_data = (cm, labels)

        # 准备结果
        result_df = X_test.copy()
        result_df['True_Label'] = y_test
        result_df['Pred_Label'] = y_pred
        result_df['Confidence'] = np.max(y_proba, axis=1)
        result_df['Is_Correct'] = y_test == y_pred

        logger.info("正在提取特征重要性")
        rf_viz = RandomForestClassifier(n_estimators=300, max_depth=20, random_state=42)
        rf_viz.fit(X_train, y_train)

        return result_df, valid_features, rf_viz

    def plot_confusion_matrix(self):
        """绘制归一化混淆矩阵（从 run_classifier 分离）"""
        if self._cm_data is None:
            return
        cm, labels = self._cm_data

        plt.figure(figsize=(10, 8))
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
        sns.heatmap(cm, annot=True, fmt='.2%', cmap='Greens', xticklabels=labels, yticklabels=labels)
        plt.title('归一化混淆矩阵', fontsize=14)
        plt.xticks(rotation=30, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()