import os

os.environ["OMP_NUM_THREADS"] = "1"
import requests
from PIL import Image, ImageFilter, ImageStat
from io import BytesIO
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import colorsys
import urllib3
import math
import cv2

MY_PROXY_PORT = 7897
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
URL_CACHE = {}

try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
except:
    face_cascade = None


def get_dominant_colors(img_array, k=3):
    """提取 Top K 主色调"""
    try:
        kmeans = KMeans(n_clusters=k, n_init=5, random_state=42)
        kmeans.fit(img_array)
        centers = kmeans.cluster_centers_
        # 计算每个聚类的权重 (像素数量)
        labels = kmeans.labels_
        counts = np.bincount(labels)
        total = np.sum(counts)

        # 按权重排序
        sorted_indices = np.argsort(counts)[::-1]
        top_centers = centers[sorted_indices]
        top_ratios = counts[sorted_indices] / total

        return top_centers, top_ratios
    except:
        return np.zeros((k, 3)), np.zeros(k)


def get_visual_features(img_pil, img_cv2):
    """提取全方位视觉特征"""
    features = {}

    # --- 1. 颜色 (Color Palette & Vibrancy) ---
    img_small = img_pil.resize((50, 50))
    img_array = np.array(img_small).reshape((-1, 3))

    # 提取 Top 3 颜色
    colors, ratios = get_dominant_colors(img_array, k=3)

    for i in range(3):
        r, g, b = colors[i]
        h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
        features[f'Hue_{i + 1}'] = h
        features[f'Sat_{i + 1}'] = s
        features[f'Val_{i + 1}'] = v
        features[f'Color_Ratio_{i + 1}'] = ratios[i]

    # 鲜艳度 (Vibrancy): 高饱和度像素(S>0.5)的占比
    hsv_array = np.array(img_small.convert('HSV')).reshape((-1, 3))
    sat_channel = hsv_array[:, 1] / 255.0
    features['Vibrancy_Ratio'] = np.mean(sat_channel > 0.5)

    # 暖色调指数 (基于第一主色)
    h1 = features['Hue_1']
    dist_to_orange = min(abs(h1 * 360 - 30), abs((h1 * 360 - 30) - 360))
    features['Warm_Rating'] = 1 - (dist_to_orange / 180.0)

    # --- 2. 纹理与排版估算 (Texture & Layout) ---
    gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)

    # 边缘密度 (Edge Density)
    edges = cv2.Canny(gray, 100, 200)
    features['Edge_Density'] = np.mean(edges) / 255.0

    # 视觉熵 (Entropy)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_norm = hist.ravel() / hist.sum()
    hist_norm = hist_norm[hist_norm > 0]
    features['Entropy'] = -np.sum(hist_norm * np.log2(hist_norm))

    # 排版/文字区域估算 (Text Area Estimate)
    # 文字通常是高频纹理区域。我们用形态学梯度来寻找密集边缘块
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
    # 二值化，保留强纹理
    _, binary_text = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    features['Text_Texture_Ratio'] = np.mean(binary_text) / 255.0

    # --- 3. 构图 (Composition) ---
    h, w = gray.shape

    # 三分法能量 (Rule of Thirds Energy)
    # 检查图像的能量(边缘)是否集中在 1/3 和 2/3 的交叉点附近
    third_h, third_w = h // 3, w // 3
    # 定义4个交叉点附近的区域
    roi_energy = 0
    centers = [(third_h, third_w), (third_h, 2 * third_w), (2 * third_h, third_w), (2 * third_h, 2 * third_w)]
    box_size = min(h, w) // 6

    for cy, cx in centers:
        y1, y2 = max(0, cy - box_size), min(h, cy + box_size)
        x1, x2 = max(0, cx - box_size), min(w, cx + box_size)
        roi = edges[y1:y2, x1:x2]
        roi_energy += np.sum(roi)

    total_energy = np.sum(edges) + 1e-5
    features['Rule_of_Thirds_Score'] = roi_energy / total_energy

    # 人脸
    if face_cascade:
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        features['Face_Count'] = len(faces)
        max_area = 0
        for (fx, fy, fw, fh) in faces:
            max_area = max(max_area, fw * fh)
        features['Face_Ratio'] = max_area / (h * w)
    else:
        features['Face_Count'] = 0
        features['Face_Ratio'] = 0

    return features


def process_single_row(row_data):
    url = row_data.get('Poster')
    if not url or pd.isna(url): return None

    if not url.startswith('http'):
        url = "https://image.tmdb.org/t/p/w500" + (url if url.startswith('/') else '/' + url)

    if url in URL_CACHE:
        row_data.update(URL_CACHE[url])
        return row_data

    try:
        proxies = {
            "http": f"http://127.0.0.1:{MY_PROXY_PORT}",
            "https": f"http://127.0.0.1:{MY_PROXY_PORT}",
        }
        headers = {'User-Agent': 'Mozilla/5.0'}

        response = requests.get(url, headers=headers, proxies=proxies, timeout=10, verify=False)
        if response.status_code != 200: return None

        img_pil = Image.open(BytesIO(response.content)).convert('RGB')
        img_cv2 = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        # 提取全套特征
        features = get_visual_features(img_pil, img_cv2)

        # 兼容旧代码，保留基本特征名
        features['Hue'] = features['Hue_1']
        features['Saturation'] = features['Sat_1']
        features['Brightness'] = features['Val_1']
        features['Sat_Std'] = 0  # 占位
        features['Bright_Std'] = 0  # 占位

        URL_CACHE[url] = features
        row_data.update(features)
        return row_data

    except Exception:
        return None


def process_images(df, sample_num=1000):
    print(f"[2/5] 正在提取海报特征 (包含配色/排版/构图/纹理 20+维特征)...")
    tasks = df.to_dict('records')
    results = []

    with ThreadPoolExecutor(max_workers=20) as executor:
        results = list(tqdm(executor.map(process_single_row, tasks), total=len(tasks)))

    valid_results = [res for res in results if res is not None]
    if not valid_results: return pd.DataFrame()

    return pd.DataFrame(valid_results)