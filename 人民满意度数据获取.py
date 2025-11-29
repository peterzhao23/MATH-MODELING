import requests
from bs4 import BeautifulSoup
import time
import random

class TourismDataCrawler:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    
    def crawl_tourist_reviews(self, base_url, max_pages=5):
        """爬取旅游评价数据"""
        all_reviews = []
        
        for page in range(1, max_pages + 1):
            try:
                url = f"{base_url}?page={page}"
                response = requests.get(url, headers=self.headers)
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # 根据网站结构提取评价内容
                reviews = soup.find_all('div', class_='review-content')  # 需要根据实际网站调整
                
                for review in reviews:
                    text = review.get_text().strip()
                    if len(text) > 10:  # 过滤过短的评论
                        all_reviews.append(text)
                
                # 随机延时，避免请求过快
                time.sleep(random.uniform(1, 3))
                
            except Exception as e:
                print(f"爬取第{page}页时出错: {e}")
                continue
        
        return all_reviews
    
    def crawl_scenic_spot_info(self):
        """爬取景区基本信息"""
        # 示例：从某旅游网站获取景区信息
        scenic_data = []
        try:
            url = "https://www.ctrip.com/"  # 替换为实际URL
            response = requests.get(url, headers=self.headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 解析景区信息
            spots = soup.find_all('div', class_='scenic-item')
            for spot in spots:
                name = spot.find('h3').text
                rating = spot.find('span', class_='rating').text
                reviews_count = spot.find('span', class_='review-count').text
                
                scenic_data.append({
                    'name': name,
                    'rating': float(rating),
                    'reviews_count': int(reviews_count)
                })
                
        except Exception as e:
            print(f"爬取景区信息出错: {e}")
        
        return scenic_data
Crawler=TourismDataCrawler()
reviews=Crawler.crawl_tourist_reviews("https://www.ctrip.com/", max_pages=5)
print(reviews)