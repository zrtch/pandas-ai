#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandasai as pai

# 设置API密钥
pai.api_key.set("PAI-8cf934ae-f430-4325-9f7c-ec80cc2f7c88")

# 创建非常简单的数据集
df = pai.DataFrame({
    "city": ["北京", "上海", "广州", "深圳", "杭州"],
    "population": [21500000, 24000000, 15000000, 12500000, 10500000]
})

# 尝试最简单的chat调用
print("开始调用chat方法...")
try:
    response = df.chat("哪个城市人口最多？")
    print(f"回答: {response}")
except Exception as e:
    print(f"发生错误: {e}")
    print(f"错误类型: {type(e)}")