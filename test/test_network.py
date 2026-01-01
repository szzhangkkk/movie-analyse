import requests

# 你的端口
PORT = 7897


def test_connection():
    proxies = {
        "http": f"http://127.0.0.1:{PORT}",
        "https": f"http://127.0.0.1:{PORT}",
    }

    # 测试 1: 访问 Google (测试代理通不通)
    print(f"--- 测试 1: 尝试通过端口 {PORT} 访问 Google ---")
    try:
        resp = requests.get("https://www.google.com", proxies=proxies, timeout=5)
        print(f"✅ Google 连接成功！状态码: {resp.status_code}")
    except Exception as e:
        print(f"❌ Google 连接失败，具体报错: {e}")

    # 测试 2: 访问 IMDB 图片 (测试图片能不能下)
    print(f"\n--- 测试 2: 尝试下载一张 IMDB 海报 ---")
    url = "https://m.media-amazon.com/images/M/MV5BMjAxMzY3NjcxNF5BMl5BanBnXkFtZTcwNTI5OTM0Mw@@._V1_SX300.jpg"
    try:
        resp = requests.get(url, proxies=proxies, timeout=10, verify=False)  # 关掉 SSL 验证试试
        if resp.status_code == 200:
            print(f"✅ 图片下载成功！大小: {len(resp.content)} bytes")
        else:
            print(f"❌ 图片连接通了，但状态码不对: {resp.status_code}")
    except Exception as e:
        print(f"❌ 图片下载失败，具体报错: {e}")


if __name__ == "__main__":
    test_connection()