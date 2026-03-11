#!/usr/bin/env python3
"""
每日学术论文抓取与分类筛选脚本（DeepSeek API 版）
功能：从Nature、Science、PRL的RSS源获取最新论文，调用DeepSeek API判断论文是否属于指定领域，保存相关论文信息。
"""

import feedparser
import openai
import json
import csv
import os
import logging
import time
import re
from datetime import datetime
from configparser import ConfigParser
from typing import List, Dict, Any, Optional

# ---------------------------- 配置加载 ----------------------------
def load_config(config_file='config.ini'):
    """读取配置文件"""
    config = ConfigParser()
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"配置文件 {config_file} 不存在，请参考以下内容创建：\n"
                                "[API]\n"
                                "openai_api_key = your-deepseek-key\n"
                                "api_base = https://api.deepseek.com/v1\n\n"
                                "[SETTINGS]\n"
                                "interested_fields = 量子物理, 材料科学, 人工智能\n"
                                "output_file = relevant_papers.csv\n"
                                "log_file = script.log\n"
                                "max_abstract_length = 1000\n"
                                "request_delay = 1")
    config.read(config_file, encoding='utf-8')
    return config

# ---------------------------- 日志设置 ----------------------------
def setup_logging(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

# ---------------------------- RSS源定义 ----------------------------
RSS_FEEDS = {
    'Nature': 'https://www.nature.com/nature.rss',
    'Science': 'https://www.science.org/action/showFeed?type=etoc&feed=rss&jc=science',
    'PRL': 'https://journals.aps.org/prl/feed'
}

# ---------------------------- 论文获取 ----------------------------
def fetch_papers_from_rss(feed_url: str, source_name: str, max_retries=3) -> List[Dict[str, Any]]:
    """从RSS源获取论文列表，返回包含标题、摘要、链接、作者、发布日期的字典列表"""
    papers = []
    for attempt in range(max_retries):
        try:
            feed = feedparser.parse(feed_url)
            if feed.bozo:
                logging.warning(f"解析 {source_name} RSS时出现异常: {feed.bozo_exception}")
            for entry in feed.entries:
                title = entry.get('title', '').strip()
                summary = entry.get('summary', '') or entry.get('description', '')
                # 清理HTML标签
                summary = re.sub('<[^<]+?>', '', summary).strip()
                link = entry.get('link', '')
                authors = entry.get('authors', [])
                if authors and isinstance(authors, list):
                    author_str = ', '.join([a.get('name', '') for a in authors])
                else:
                    author_str = entry.get('author', '')
                published = entry.get('published', '') or entry.get('pubDate', '')
                papers.append({
                    'source': source_name,
                    'title': title,
                    'abstract': summary,
                    'link': link,
                    'authors': author_str,
                    'published': published
                })
            logging.info(f"从 {source_name} 获取到 {len(papers)} 篇论文")
            break
        except Exception as e:
            logging.error(f"获取 {source_name} RSS失败 (尝试 {attempt+1}/{max_retries}): {e}")
            time.sleep(2)
    return papers

# ---------------------------- AI分类（DeepSeek API） ----------------------------
def classify_paper(paper: Dict[str, Any], interested_fields: List[str], 
                   api_key: str, api_base: str, max_length: int = 1000, delay: float = 1.0) -> Optional[Dict]:
    """调用DeepSeek API判断论文是否属于感兴趣领域，返回结果字典或None（失败）"""
    # 创建OpenAI客户端，但指向DeepSeek的base_url
    client = openai.OpenAI(api_key=api_key, base_url=api_base)
    
    fields_str = ', '.join(interested_fields)
    title = paper['title']
    abstract = paper['abstract'][:max_length]
    
    messages = [
        {"role": "system", "content": "你是一个科学论文分类助手。"},
        {"role": "user", "content": f"""请根据论文标题和摘要，判断它是否属于以下领域：{fields_str}。
如果属于，请返回一个JSON对象：{{"relevant": true, "field": "具体领域"}}（具体领域必须来自给定的领域列表，选择最匹配的一个）。
如果不属于，返回{{"relevant": false}}。
只返回JSON，不要有其他文字。

标题：{title}
摘要：{abstract}"""}
    ]
    
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",  # DeepSeek的对话模型名称
            messages=messages,
            temperature=0,
            max_tokens=100,
            timeout=30
        )
        reply = response.choices[0].message.content.strip()
        result = json.loads(reply)
        if isinstance(result, dict) and 'relevant' in result:
            time.sleep(delay)
            return result
        else:
            logging.warning(f"API返回格式异常: {reply}")
            return None
    except json.JSONDecodeError:
        logging.warning(f"API返回非JSON: {reply}")
        return None
    except Exception as e:
        logging.error(f"API调用失败: {e}")
        return None

# ---------------------------- 保存结果 ----------------------------
def save_paper(paper: Dict[str, Any], classification: Dict, output_file: str):
    """将相关论文追加到CSV文件"""
    file_exists = os.path.isfile(output_file)
    with open(output_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['日期', '来源', '标题', '作者', '发表时间', '链接', '摘要', '相关领域'])
        writer.writerow([
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            paper['source'],
            paper['title'],
            paper['authors'],
            paper['published'],
            paper['link'],
            paper['abstract'][:500] + ('...' if len(paper['abstract']) > 500 else ''),
            classification.get('field', '')
        ])

# ---------------------------- 主流程 ----------------------------
def main():
    config = load_config()
    api_key = config.get('API', 'openai_api_key')
    api_base = config.get('API', 'api_base', fallback='https://api.deepseek.com/v1')
    interested_fields = [f.strip() for f in config.get('SETTINGS', 'interested_fields').split(',')]
    output_file = config.get('SETTINGS', 'output_file', fallback='relevant_papers.csv')
    log_file = config.get('SETTINGS', 'log_file', fallback='script.log')
    max_abstract = config.getint('SETTINGS', 'max_abstract_length', fallback=1000)
    request_delay = config.getfloat('SETTINGS', 'request_delay', fallback=1.0)
    
    setup_logging(log_file)
    logging.info("===== 开始每日论文抓取 =====")
    logging.info(f"感兴趣的领域: {interested_fields}")
    logging.info(f"使用API: {api_base}")
    
    all_papers = []
    for source, url in RSS_FEEDS.items():
        papers = fetch_papers_from_rss(url, source)
        all_papers.extend(papers)
    
    logging.info(f"总计获取论文: {len(all_papers)} 篇")
    
    relevant_count = 0
    for idx, paper in enumerate(all_papers):
        logging.info(f"处理第 {idx+1}/{len(all_papers)} 篇: {paper['title'][:50]}...")
        classification = classify_paper(paper, interested_fields, api_key, api_base, max_abstract, request_delay)
        if classification and classification.get('relevant'):
            relevant_count += 1
            save_paper(paper, classification, output_file)
            logging.info(f"相关论文已保存，领域: {classification.get('field')}")
    
    logging.info(f"处理完成，共发现 {relevant_count} 篇相关论文，结果保存至 {output_file}")

if __name__ == '__main__':
    main()