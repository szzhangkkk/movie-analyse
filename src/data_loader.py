import pandas as pd
def load_data(file_path):
    print("[1/5]正在加载并清洗数据...")
    df = pd.read_csv(file_path,encoding = 'utf-8')
    #删除不合规定的数据
    df_clean = df.dropna(subset=['Poster','IMDB Rating','Genres','Titlee']).copy()
    df_clean['Main_Genre'] = df_clean['Genres'].str.split('|').str[0]
    #筛选热门类型
    top_genres = df_clean['Main_Genre'].value_counts().head(10).index
    df_final = df_clean[df_clean['Main_Genre'].isin(top_genres)]
    print(f"    数据清洗完成，剩余有效电影: {len(df_final)} 部")
    return df_final