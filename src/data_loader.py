import logging

import pandas as pd

logger = logging.getLogger(__name__)

RENAME_MAP = {
    'Series_Title': 'Title',
    'IMDB_Rating': 'IMDB Rating',
    'Genre': 'Genres',
    'Poster_Link': 'Poster',
    'Runtime': 'Length in Min',
}

POSTER_COL_ALIASES = {'Poster_Link', 'Poster', 'poster', 'poster_link', 'poster_url'}


def load_data(file_path):
    logger.info("正在加载 IMDB Top 1000 数据集")

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        logger.error("读取失败: %s", e)
        return pd.DataFrame()

    # 自动检测海报列名
    poster_col = None
    for col in df.columns:
        if col in POSTER_COL_ALIASES:
            poster_col = col
            break

    if poster_col is None:
        logger.error("CSV 中未找到海报列 (尝试匹配: %s)，请确认文件格式", POSTER_COL_ALIASES)
        return pd.DataFrame()

    # 动态构建重命名映射
    rename = {k: v for k, v in RENAME_MAP.items() if k in df.columns}
    df = df.rename(columns=rename)

    # 数据清洗
    df_clean = df.dropna(subset=['Poster', 'Genres', 'Title']).copy()

    df_clean['Main_Genre'] = df_clean['Genres'].astype(str).str.split(',').str[0].str.strip()

    if 'Length in Min' in df_clean.columns:
        df_clean['Length in Min'] = (
            df_clean['Length in Min'].astype(str).str.replace(' min', '').apply(pd.to_numeric, errors='coerce')
        )

    top_genres = df_clean['Main_Genre'].value_counts().head(8).index
    df_final = df_clean[df_clean['Main_Genre'].isin(top_genres)]

    logger.info("原始数据: %d 条, 清洗后: %d 条, 包含类型: %s", len(df), len(df_final), list(top_genres))

    return df_final