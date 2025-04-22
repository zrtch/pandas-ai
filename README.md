# pandas-ai 高级应用示例集合

本项目展示了如何在 Python 中充分发挥 [pandas-ai](https://github.com/sinaptik-ai/pandas-ai) 的强大功能，实现自然语言驱动的数据分析。

## 项目简介

- 基于 pandas-ai，可以用自然语言提问的方式分析数据和生成可视化
- 支持中文提问和回答
- 包含了从基础到高级的多种使用场景
- 提供了完整的代码示例和数据集

## 示例集合

本项目包含以下示例：

1. **基础示例** (demo.py)：演示最基本的 pandas-ai 用法
2. **数据可视化** (charts.py)：展示如何使用自然语言生成数据图表
3. **高级分析** (advanced_demo.py)：包含更复杂的数据分析场景和技巧
4. **金融数据分析** (financial_analysis.py)：股票市场分析和投资组合优化
5. **NLP情感分析** (nlp_sentiment.py)：社交媒体评论情感分析和消费者洞察

## 安装方法

建议使用虚拟环境：

```bash
# 创建并激活虚拟环境
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

## 使用方法

每个示例都可以直接运行：

```bash
# 基础示例
python demo.py

# 数据可视化示例
python charts.py

# 高级分析示例
python advanced_demo.py

# 金融数据分析
python financial_analysis.py

# NLP情感分析
python nlp_sentiment.py
```

## 关键特性展示

### 1. 多数据源分析

pandas-ai 可以同时处理多个 DataFrame，进行关联分析：

```python
# 传入多个DataFrame进行分析
response = pandas_ai.run(
    [df1, df2],
    "分析两个数据集的关系，找出共同模式"
)
```

### 2. 自定义分析函数

可以要求 pandas-ai 生成和执行自定义分析代码：

```python
# 引导AI执行复杂任务
response = pandas_ai.run(
    df,
    """
    执行以下任务：
    1. 创建一个新指标，计算方法为...
    2. 使用聚类算法将数据分为3组
    3. 可视化结果并解释
    """
)
```

### 3. 上下文连续提问

pandas-ai 支持基于上下文的连续提问：

```python
# 首次分析
response = pandas_ai.run(df, "哪些产品销量最高？")

# 基于上下文的后续问题
follow_up = pandas_ai.run(None, "这些产品的利润率如何？")
```

### 4. 报告生成

可以要求 pandas-ai 生成完整的分析报告：

```python
report = pandas_ai.run(
    df,
    "生成一份详细的市场分析报告，包括趋势、机会和风险"
)
```

## 注意事项

- 示例中使用的API密钥仅供演示，在实际使用时请替换为您自己的密钥
- 生成的图表默认保存在 exports 目录下
- 您可以根据需要调整提示词，改变分析方向和深度

## 进阶技巧

1. **提示词优化**：提供详细、明确的提示词可以获得更好的分析结果
2. **处理大数据集**：对于大型数据集，考虑先进行采样或聚合
3. **结合传统方法**：将pandas-ai与传统pandas操作结合使用效果更佳
4. **定制数据预处理**：在传入数据前先进行清洗和转换
5. **结果验证**：始终验证AI生成的分析结果是否合理

## 依赖库

- pandas-ai
- pandas
- numpy
- matplotlib
- seaborn
- yfinance (金融示例)
- jieba (NLP示例)

## 贡献指南

欢迎提交Pull Request或Issues来改进此项目！
