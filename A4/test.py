# import requests

# # 目标网址
# url = "https://www.behindthename.com/names/usage/english"

# # 发起GET请求
# response = requests.get(url)

# # 检查请求是否成功
# if response.status_code == 200:
#     # 获取HTML内容
#     with open('response.html', 'w', encoding='utf-8') as f:
#         f.write(response.text)
#     # print(html_content)  # 打印HTML内容
# else:
#     print(f"请求失败，状态码：{response.status_code}")

# 查询当前系统所有字体
from matplotlib.font_manager import FontManager
import subprocess

mpl_fonts = set(f.name for f in FontManager().ttflist)

print('all font list get from matplotlib.font_manager:')
for f in sorted(mpl_fonts):
    print('\t' + f)