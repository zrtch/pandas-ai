import pandasai as pai

# Sample DataFrame
df = pai.DataFrame({
    "country": ["United States", "United Kingdom", "France", "Germany", "Italy", "Spain", "Canada", "Australia", "Japan", "China"],
    "revenue": [5000, 3200, 2900, 4100, 2300, 2100, 2500, 2600, 4500, 7000]
})

# By default, unless you choose a different LLM, it will use BambooLLM.
# You can get your free API key signing up at https://app.pandabi.ai (you can also configure it in your .env file)
pai.api_key.set("PAI-8cf934ae-f430-4325-9f7c-ec80cc2f7c88")

# 要求 PandaAI 为您生成图表, 会在exports文件夹下面生成一张图标图片。
df.chat(
    "Plot the histogram of countries showing for each one the gd. Use different colors for each bar",
)