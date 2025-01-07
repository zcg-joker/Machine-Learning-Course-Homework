import requests
from bs4 import BeautifulSoup
import os
import pandas as pd
import zhplot
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

def fetch_names_from_page(url):
    """从指定页面获取英文名字"""
    names = []
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        name_elements = soup.find_all('span', class_='listname')
        for element in name_elements:
            link = element.find('a')
            if link:
                names.append(link.text.strip())
        
        return names
    except Exception as e:
        print(f"爬取过程中出现错误: {e}")
        return []

def fetch_all_names():
    """爬取所有页面的英文名字"""
    base_url = 'https://www.behindthename.com/names/usage/english'
    all_names = []
    
    # 爬取第一页
    print(f"正在爬取第1页: {base_url}")
    names = fetch_names_from_page(base_url)
    all_names.extend(names)
    
    # 爬取第2-15页
    for page in range(2, 16):
        url = f"{base_url}/{page}"
        print(f"正在爬取第{page}页: {url}")
        names = fetch_names_from_page(url)
        all_names.extend(names)
    
    return all_names

def save_names_to_file(names, filename):
    """保存名字列表到文件"""
    os.makedirs('data', exist_ok=True)
    with open(f'data/{filename}', 'w', encoding='utf-8') as f:
        for name in names:
            f.write(name + '\n')

def analyze_names(names):
    """分析名字数据"""
    # 创建DataFrame
    df = pd.DataFrame({'name': names})
    
    # 计算基本统计信息
    df['name_length'] = df['name'].str.len()
    df['first_letter'] = df['name'].str[0]
    
    # 生成统计报告
    stats = {
        '样本总数': len(names),
        '平均名字长度': df['name_length'].mean(),
        '最短名字长度': df['name_length'].min(),
        '最长名字长度': df['name_length'].max(),
        '名字长度中位数': df['name_length'].median()
    }
    
    return df, stats

def visualize_data(df, stats):
    """数据可视化"""
    os.makedirs('data', exist_ok=True)
    
    # 1. 名字长度分布图
    plt.figure(figsize=(10, 6))
    counts, bins, patches = plt.hist(df['name_length'], bins=20, edgecolor='black')
    plt.title('名字长度分布')
    plt.xlabel('名字长度')
    plt.ylabel('频率')
    plt.grid(True, alpha=0.3)
    
    # 在每个柱子上显示值
    for count, bin in zip(counts, bins):
        # 计算柱子的中心位置
        plt.text(bin + (bins[1] - bins[0]) / 2, count, int(count), ha='center', va='bottom')
    
    plt.savefig('data/name_length_distribution.png')
    plt.close()
    
    # 2. 首字母分布图
    plt.figure(figsize=(12, 6))
    first_letter_counts = df['first_letter'].value_counts()
    bars = plt.bar(first_letter_counts.index, first_letter_counts.values)
    plt.title('首字母分布')
    plt.xlabel('首字母')
    plt.ylabel('频率')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 在每个柱子上显示值
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, int(yval), ha='center', va='bottom')
    
    plt.tight_layout()  # 调整布局，防止标签被切割
    plt.savefig('data/first_letter_distribution.png')
    plt.close()
    
    # 3. 生成统计报告
    with open('data/statistics_report.txt', 'w', encoding='utf-8') as f:
        f.write("英文名字统计分析报告\n")
        f.write("=" * 30 + "\n\n")
        for key, value in stats.items():
            f.write(f"{key}: {value:.2f}\n")

if __name__ == '__main__':
    # 爬取名字
    names = fetch_all_names()
    print(f"共获取到 {len(names)} 个名字")
    
    # 保存原始数据
    save_names_to_file(names, 'english_names.txt')
    
    # 分析数据
    df, stats = analyze_names(names)
    
    # 可视化
    visualize_data(df, stats)
    
    print("分析完成！请查看data文件夹中的结果。")
