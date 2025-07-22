from openai import OpenAI
import requests


class EmotionFusionGenerator:
    def __init__(self, llm_model="deepseek-chat"):
        self.api_key = 'sk-62167b879ab04cac9b75e04401351095'
        self.llm_model = llm_model
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.deepseek.com/v1"
        ) if self.api_key else None

    @staticmethod
    def check_api_key(api_key):
        url = "https://api.deepseek.com/v1/models"
        headers = {"Authorization": f"Bearer {api_key}"}
        try:
            response = requests.get(url, headers=headers)
            print(f"API Key Status: {response.status_code}")
            if response.status_code == 200:
                print("API Key Valid")
                return True
            print(f"Response: {response.json()}")
            return False
        except Exception as e:
            print(f"Fail in examination: {str(e)}")
            return False

    def generate_prompt(self, emotions, percentages, context=None):
        emotion_desc = []
        for emo, pct in zip(emotions, percentages):
            emotion_desc.append(f"{emo}({pct:.1%})")
        print(emotion_desc)

        prompt = (
            "你是一个专业的情绪分析文案生成器。根据以下情绪分布，找到最能够描述这种复杂情绪的语句，创作一句简洁的表情包文案：\n"
            f"情绪分布: {', '.join(emotion_desc)}\n\n"
            "要求：\n"
            "1. 使用生动形象的感官描写\n"
            "2. 反映情绪之间的微妙平衡\n"
            "3. 字数在1-5字之间\n"
            "4. 输出的文本不要用双引号包裹\n"
            "5. 避免重复描述\n"
            "6. 确保输出的文案没有错误\n"
            "7. 确保输出的语言浅显易懂经典自然\n"
            "8. 避免直接把表达不同情绪的词语连接在一起\n"
            "9. 可以适当融入网络热梗\n"
            "10. 降低占比远低于其他情绪的情绪对文案的影响\n"
            "11. 除非'neutral'情绪占比远高于其他情绪，否则将除'neutral'情绪外的、占比最大的情绪视为主要情绪，'neutral'占比只影响该情绪强烈程度\n"
            "例子：\n"
            "['happy(90.1%)', 'neutral(9.9%)']输出'哈哈哈哈哈'\n"
            "解释：这个例子中'happy'占比远高于'neutral'，因此生成的文案更偏向'happy'情绪\n"
            "['happy(50.1%)', 'neutral(49.9%)']输出'礼貌性微笑'\n"
            "解释：这个例子中'happy'占比与'neutral'相近，因此生成的文案以'happy'作为主要情绪，'neutral'占比较高代表'happy'情绪不是非常强烈，因此输出文案比较平静\n"
        )

        if context:
            prompt += f"\nExtra context: {context}\n"

        return prompt

    def generate_caption(self, emotions, percentages, context=None):
        if not self.client:
            return "LLM API key not provided or invalid"

        prompt = self.generate_prompt(emotions, percentages, context)

        try:
            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system",
                     "content": "你是一个熟悉网络用语的文案作家，擅长为表情包撰写前沿潮流、生动有趣的文案。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=10
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error in LLM generation: {str(e)}")
            return
