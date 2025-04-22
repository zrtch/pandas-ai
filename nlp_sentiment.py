#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandasai as pai
import os
import re
import datetime
import jieba
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# 配置API密钥
pai.api_key.set("PAI-8cf934ae-f430-4325-9f7c-ec80cc2f7c88")

# 创建输出目录
os.makedirs("exports/nlp_charts", exist_ok=True)

# 创建示例数据：社交媒体评论
def create_social_media_data(n_samples=500):
    """创建包含社交媒体评论的模拟数据集"""
    
    # 设置随机种子以保证可重复性
    np.random.seed(42)
    
    # 评论主题
    topics = ["手机", "电脑", "平板", "智能手表", "耳机", "智能家居", "相机", "游戏机"]
    
    # 品牌
    brands = {
        "手机": ["苹果", "华为", "小米", "三星", "OPPO", "vivo"],
        "电脑": ["联想", "戴尔", "惠普", "华硕", "苹果", "微软"],
        "平板": ["苹果", "华为", "小米", "三星", "联想"],
        "智能手表": ["苹果", "华为", "小米", "三星", "OPPO"],
        "耳机": ["苹果", "华为", "小米", "索尼", "Bose", "森海塞尔"],
        "智能家居": ["小米", "华为", "阿里", "海尔", "美的"],
        "相机": ["佳能", "尼康", "索尼", "富士", "徕卡"],
        "游戏机": ["索尼", "微软", "任天堂"]
    }
    
    # 创建常见的正面、中性、负面评论模板
    positive_templates = [
        "我很喜欢这款{brand}的{topic}，用起来非常舒适。",
        "刚买了{brand}的新{topic}，真的太好用了！",
        "{brand}的{topic}质量很好，很耐用，推荐购买。",
        "我用了{brand}的{topic}已经两年了，一直很稳定，没有任何问题。",
        "{brand}这款{topic}的性价比真的很高，物超所值。",
        "{topic}界的标杆，{brand}实至名归。",
        "设计精美，做工精良，{brand}的{topic}永远让人满意。",
        "买了{brand}的{topic}，颜值和性能都很满意，赞一个！",
        "{brand}的客服很好，{topic}有小问题马上就解决了。",
        "这是我用过最好的{topic}，{brand}没有让我失望。"
    ]
    
    neutral_templates = [
        "{brand}的{topic}还可以，但价格有点贵。",
        "刚入手{brand}的{topic}，感觉一般般吧。",
        "这款{brand}{topic}有优点也有缺点，总体来说中规中矩。",
        "{brand}的{topic}外观不错，但功能上没什么特别之处。",
        "用了一周{brand}的{topic}，感觉还行，不算特别惊艳。",
        "{brand}这款{topic}质量还行，但创新不足。",
        "比起上一代，{brand}这款新{topic}改进不大。",
        "{topic}本身没问题，就是{brand}的售后服务一般。",
        "买{brand}的{topic}主要是冲着品牌去的，产品本身中规中矩。",
        "{brand}的{topic}和竞品相比各有千秋。"
    ]
    
    negative_templates = [
        "不推荐{brand}的这款{topic}，质量太差了。",
        "买了{brand}的{topic}后悔死了，系统经常崩溃。",
        "{brand}的{topic}价格那么贵，但体验却很差。",
        "用了一个月，{brand}的{topic}就出问题了，太失望了。",
        "{brand}的{topic}外观不错，但电池续航太差。",
        "千万别买{brand}的{topic}，浪费钱！",
        "{brand}的售后服务太差了，{topic}坏了维修很麻烦。",
        "同价位下，{brand}的{topic}是我用过最差的一款。",
        "{brand}的{topic}发热严重，体验很差。",
        "对{brand}这款{topic}完全失望，将来再也不会选择这个品牌了。"
    ]
    
    # 创建日期范围：过去一年
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=365)
    date_range = (end_date - start_date).days
    
    # 创建数据
    data = {
        "ID": range(1, n_samples + 1),
        "日期": [],
        "主题": [],
        "品牌": [],
        "评论": [],
        "情感": [],
        "情感分数": [],
        "点赞数": [],
        "评论长度": [],
        "是否包含表情": [],
        "平台": []
    }
    
    platforms = ["微博", "知乎", "小红书", "抖音", "B站", "微信"]
    
    for i in range(n_samples):
        # 随机选择日期
        random_days = np.random.randint(0, date_range)
        post_date = start_date + datetime.timedelta(days=random_days)
        data["日期"].append(post_date.strftime("%Y-%m-%d"))
        
        # 随机选择主题和品牌
        topic = np.random.choice(topics)
        brand = np.random.choice(brands[topic])
        data["主题"].append(topic)
        data["品牌"].append(brand)
        
        # 随机选择情感倾向
        sentiment_probs = [0.6, 0.25, 0.15]  # 正面, 中性, 负面的概率
        sentiment = np.random.choice(["正面", "中性", "负面"], p=sentiment_probs)
        data["情感"].append(sentiment)
        
        # 基于情感选择评论模板
        if sentiment == "正面":
            template = np.random.choice(positive_templates)
            sentiment_score = round(np.random.uniform(0.6, 1.0), 2)
        elif sentiment == "中性":
            template = np.random.choice(neutral_templates)
            sentiment_score = round(np.random.uniform(0.4, 0.6), 2)
        else:
            template = np.random.choice(negative_templates)
            sentiment_score = round(np.random.uniform(0.0, 0.4), 2)
        
        # 填充模板
        comment = template.format(brand=brand, topic=topic)
        
        # 随机添加一些表情符号
        has_emoji = np.random.choice([True, False], p=[0.3, 0.7])
        if has_emoji:
            emojis = ["😊", "👍", "❤️", "😢", "😡", "🤔", "👎", "🎉", "🔥", "👏"]
            # 在评论末尾添加1-3个随机表情
            n_emojis = np.random.randint(1, 4)
            for _ in range(n_emojis):
                comment += np.random.choice(emojis)
        
        data["评论"].append(comment)
        data["情感分数"].append(sentiment_score)
        data["点赞数"].append(np.random.randint(0, 1000))
        data["评论长度"].append(len(comment))
        data["是否包含表情"].append(has_emoji)
        data["平台"].append(np.random.choice(platforms))
    
    return pd.DataFrame(data)

# 分词和词频统计
def analyze_word_frequency(df, stop_words=None):
    """分析评论中的词频，返回词频统计结果"""
    if stop_words is None:
        # 常见停用词
        stop_words = set(["的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", 
                        "一", "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", 
                        "着", "没有", "看", "好", "自己", "这", "那", "这个", "那个", "啊",
                        "吧", "把", "被", "给", "让", "从", "但", "但是", "并", "个", "呢",
                        "呀", "哦", "喔", "嗯", "这样", "那样", "只", "只有", "可", "可以",
                        "或", "如果", "所以", "什么", "这么", "那么", "为", "为了", "么", "还"])
    
    all_words = []
    
    # 遍历所有评论
    for comment in df["评论"]:
        # 移除表情符号和标点符号
        clean_comment = re.sub(r'[^\u4e00-\u9fa5]', '', comment)
        # 使用jieba分词
        words = jieba.cut(clean_comment)
        # 过滤停用词
        filtered_words = [word for word in words if word not in stop_words and len(word) > 1]
        all_words.extend(filtered_words)
    
    # 统计词频
    word_counts = Counter(all_words)
    
    # 转换为DataFrame
    word_freq_df = pd.DataFrame({
        "词语": list(word_counts.keys()),
        "频率": list(word_counts.values())
    })
    
    # 按频率排序
    word_freq_df = word_freq_df.sort_values("频率", ascending=False).reset_index(drop=True)
    
    return word_freq_df

# 主函数
def main():
    # 创建模拟数据
    print("正在生成社交媒体评论数据...")
    df = create_social_media_data(500)
    print(f"数据生成完成，共 {len(df)} 条评论")
    
    # 分词并统计词频
    print("正在进行分词和词频统计...")
    word_freq_df = analyze_word_frequency(df)
    print(f"词频统计完成，共 {len(word_freq_df)} 个有效词语")
    
    # 初始化PandasAI
    pandas_ai = pai.PandasAI()
    
    # 1. 基本统计分析
    print("\n基本统计分析:")
    response = pandas_ai.run(
        df,
        "对整个数据集进行基本统计分析，包括各品牌的评论数量、各主题的评论分布、情感分布情况等，并用图表直观展示"
    )
    print(response)
    
    # 2. 情感分析
    print("\n品牌情感分析:")
    response = pandas_ai.run(
        df,
        "分析不同品牌的情感评分对比，哪些品牌评价最好？哪些品牌评价最差？请给出数据支持的结论并用图表展示"
    )
    print(response)
    
    # 3. 时间趋势分析
    print("\n时间趋势分析:")
    response = pandas_ai.run(
        df,
        "分析评论情感随时间的变化趋势，是否有明显的季节性模式或重大事件影响？展示一个按月的情感变化曲线图"
    )
    print(response)
    
    # 4. 词频分析
    print("\n热门词汇分析:")
    response = pandas_ai.run(
        word_freq_df,
        "分析最常见的20个词语，这些词语反映了用户对产品的哪些方面最关注？生成一个词频柱状图"
    )
    print(response)
    
    # 5. 平台比较分析
    print("\n平台比较分析:")
    response = pandas_ai.run(
        df,
        "比较不同平台上的评论特点，包括情感倾向、评论长度、点赞数等方面，不同平台的用户表现出哪些差异？"
    )
    print(response)
    
    # 6. 主题与情感交叉分析
    print("\n主题情感分析:")
    response = pandas_ai.run(
        df,
        "分析不同主题的产品获得的情感评价，哪类产品评价最高？哪类最低？可能的原因是什么？"
    )
    print(response)
    
    # 7. 表情符号影响分析
    print("\n表情符号影响分析:")
    response = pandas_ai.run(
        df,
        "分析包含表情符号的评论与不包含表情符号的评论在情感分数、点赞数等方面的差异，表情符号的使用是否与特定情感相关？"
    )
    print(response)
    
    # 8. 点赞数与情感关系
    print("\n点赞数与情感关系:")
    response = pandas_ai.run(
        df,
        "分析评论的点赞数与情感分数之间的关系，正面评论是否获得更多点赞？负面评论是否也能获得高点赞？"
    )
    print(response)
    
    # 9. 品牌声誉分析
    print("\n品牌声誉综合分析:")
    response = pandas_ai.run(
        df,
        "基于情感分数、点赞数和评论数量，综合评估各品牌的社交媒体声誉，并对各品牌的社交媒体表现进行排名"
    )
    print(response)
    
    # 10. 消费者洞察报告
    print("\n消费者洞察报告:")
    response = pandas_ai.run(
        [df, word_freq_df],
        "生成一份全面的消费者洞察报告，包括市场趋势、品牌表现、用户关注点和改进建议等内容"
    )
    print(response)
    
    # 导出数据供后续使用
    df.to_csv("exports/social_media_comments.csv", index=False, encoding="utf-8-sig")
    word_freq_df.to_csv("exports/word_frequency.csv", index=False, encoding="utf-8-sig")

if __name__ == "__main__":
    main()
    print("分析完成，请查看生成的图表和分析结果。") 