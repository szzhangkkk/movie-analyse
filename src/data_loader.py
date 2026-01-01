import pandas as pd


def load_data(file_path):
    print(f"[1/5] 正在加载 IMDB Top 1000 数据集...")

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"❌ 读取失败: {e}")
        return pd.DataFrame()

    # --- 1. 列名映射 ---
    # 把 Kaggle 的列名翻译成我们代码通用的名字
    rename_map = {
        'Series_Title': 'Title',
        'IMDB_Rating': 'IMDB Rating',
        'Genre': 'Genres',
        'Poster_Link': 'Poster',  # 它的列名直接叫 Poster_Link
        'Runtime': 'Length in Min'
    }

    # 检查是不是下对文件了
    if 'Poster_Link' not in df.columns:
        print("❌ 错误：CSV里没找到 'Poster_Link' 列！请确认你用的是 imdb_top_1000.csv")
        return pd.DataFrame()

    # 重命名
    df = df.rename(columns=rename_map)

    # --- 2. 数据清洗 ---
    # 删除缺失关键信息的行
    df_clean = df.dropna(subset=['Poster', 'Genres', 'Title']).copy()

    # 处理类型：取逗号前的第一个词作为主类型
    # 例如 "Crime, Drama" -> "Crime"
    df_clean['Main_Genre'] = df_clean['Genres'].astype(str).str.split(',').str[0].str.strip()

    # 处理时长：把 "142 min" 变成数字 142
    if 'Length in Min' in df_clean.columns:
        df_clean['Length in Min'] = df_clean['Length in Min'].astype(str).str.replace(' min', '').apply(pd.to_numeric,
                                                                                                        errors='coerce')

    # 筛选热门类型 (只保留前 8 大类型，保证统计图好看)
    top_genres = df_clean['Main_Genre'].value_counts().head(8).index
    df_final = df_clean[df_clean['Main_Genre'].isin(top_genres)]

    print(f"    原始数据: {len(df)} 条")
    print(f"    清洗后有效数据: {len(df_final)} 条")
    print(f"    包含类型: {list(top_genres)}")

    return df_final