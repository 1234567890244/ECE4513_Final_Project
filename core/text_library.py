import random

TEXT_LIBRARY = {
    "happy": [
        "哈哈哈哈哈哈", "笑死我了", "今天是个好日子", "开心到飞起",
        "笑口常开", "乐不思蜀", "开心每一天", "喜笑颜开"
    ],
    "sad": [
        "我太难了", "蓝瘦香菇", "悲伤逆流成河", "哭唧唧",
        "泪流满面", "心如刀割", "伤心欲绝", "黯然神伤"
    ],
    "angry": [
        "气死我了", "怒火中烧", "我生气了", "别惹我",
        "怒发冲冠", "火冒三丈", "气不打一处来", "愤怒的小鸟"
    ],
    "surprise": [
        "惊呆了", "我的天啊", "难以置信", "太意外了",
        "大吃一惊", "目瞪口呆", "意外惊喜", "匪夷所思"
    ],
    "neutral": [
        "淡定", "面无表情", "冷静思考中", "佛系",
        "心如止水", "波澜不惊", "面无表情", "一切如常"
    ],
    "fear": [
        "吓死宝宝了", "瑟瑟发抖", "好害怕", "惊恐万分",
        "胆战心惊", "毛骨悚然", "不寒而栗", "惊魂未定"
    ],
    "disgust": [
        "嫌弃", "恶心", "受不了了", "呕",
        "令人作呕", "不忍直视", "反感至极", "厌恶万分"
    ]
}


def get_random_text(emotion):
    if emotion in TEXT_LIBRARY:
        return random.choice(TEXT_LIBRARY[emotion])
    return random.choice(TEXT_LIBRARY["neutral"])
