import json
import os
from API_KEY import *
from openai import OpenAI
import requests
import torch
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# 导入中文字体，指定字号
font = ImageFont.truetype('asset/SimHei.ttf', 26)

# 系统提示词
SYS_PROMPT = '''
我即将要说一句机械臂指令，我现在希望你从这句话中提取出起始物体和终止物体。并输出其中文和英文翻译,输出json数据结构。

例如，如果我的指令是：把电池放到红色方块上
你输出应为：
{
'start_C':['电池', '红色方块']
'start_E':['Battery', 'Red Square']
}

只回复json本身即可，不要回复其它内容

我现在的指令是：

'''

def pre_yi(PROMPT='帮我把蓝色三角片放在钢笔上'):
    '''
    零一万物大模型API
    '''
    os.environ["YI_KEY"] = YI_KEY

    API_BASE = "https://api.lingyiwanwu.com/v1"
    API_KEY = YI_KEY

    MODEL = 'yi-spark'
    # MODEL = 'yi-medium'
    # MODEL = 'yi-spark'

    # 访问大模型API
    client = OpenAI(api_key=API_KEY, base_url=API_BASE)
    completion = client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": SYS_PROMPT + PROMPT}])
    result = completion.choices[0].message.content.strip()
    return result


# 示例调用
PROMPT = '把紫色方块放到红色方块上'  # 你可以改变这个指令
result = pre_yi(PROMPT)
# 清理不必要的标记符号
result = result.strip('```json').strip()
# 校验并返回 JSON 数据

# 打印结果
try:
    # 解析并打印JSON格式的输出
    result_json = json.loads(result)
    print(result_json["start_C"][0], result_json["start_C"][1])
    # print(result_json["start_E"][0], result_json["start_E"][1])
    print(f"{result_json['start_E'][0]}.{result_json['start_E'][1]}")

    # 1. 加载模型
    model_id = "IDEA-Research/grounding-dino-base"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    # 2. 下载并加载图像
    image_path = "temp/vl_now.jpg"
    image = Image.open(image_path)
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)  # 转换为 OpenCV 格式

    # 3. 定义检测目标
    text = f"{result_json['start_E'][0]}. {result_json['start_E'][1]}."

    # 4. 预处理输入
    inputs = processor(images=image, text=text, return_tensors="pt").to(device)

    # 5. 模型推理
    with torch.no_grad():
        outputs = model(**inputs)

    # 6. 解析检测结果
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.4,
        text_threshold=0.3,
        target_sizes=[image.size[::-1]]
    )

    # 7. 可视化检测结果
    colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (255, 255, 0), (128, 0, 128)]  # 颜色列表 (BGR 格式)

    # 获取文本类别列表（去掉空白项）
    category_names = [t.strip() for t in text.split(".") if t.strip()]

    for i, (box, score, label) in enumerate(zip(results[0]["boxes"], results[0]["scores"], results[0]["labels"])):
        box = [int(coord) for coord in box.tolist()]
        color = colors[i % len(colors)]

        # 使用中文标签（确保顺序正确）
        category_name = result_json["start_C"][1-i]

        # 绘制中文标签（根据框的坐标绘制）
        text_label = f"{category_name} ({score:.2f})"

        # 计算中心点坐标
        center_x, center_y = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2

        # 绘制边界框
        cv2.rectangle(image_cv, (box[0], box[1]), (box[2], box[3]), color, 2)

        # 绘制目标中心点
        cv2.circle(image_cv, (center_x, center_y), 3, color, -1)

        # 在目标中心点附近显示坐标
        cv2.putText(image_cv, f"({center_x},{center_y})", (center_x + 5, center_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    color, 2)

        # 创建一个PIL图像用于绘制中文
        pil_image = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        draw.text((box[0], box[1] - 30), text_label, font=font, fill=color[::-1])

        # 将PIL图像转换回OpenCV图像
        image_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    # 8. 显示结果
    cv2.imshow("Detection Result", image_cv)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

except json.JSONDecodeError:
    print("Error: Failed to decode the JSON response.")

