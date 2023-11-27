from transformers import BitsAndBytesConfig
import torch


def get_prompt(instruction: str) -> str:
    '''Format the instruction as a prompt for LLM.'''
    return f"你是人工智慧助理，以下是用戶和人工智能助理之間的對話。你要對用戶的問題提供有用、安全、詳細和禮貌的回答。USER: {instruction} ASSISTANT:"

def get_few_shot_prompt(instruction: str) -> str:
    return f" 你是人工智慧助理，以下是用戶和人工智能助理之間的對話。你要對用戶的問題提供有用、安全、詳細和禮貌的回答。 \
              Example 1: [USER] 有的說不吃葷腥，不近女色，遇上這樣的好藥就能成為地仙。把這句話翻譯成文言文： [ASSISTANT] 或不葷血，不色欲，遇之必能降真為地仙矣。 \
              Example 2: [USER] 敢有犯者，請以故違敕論。\n翻譯成白話文： [ASSISTANT] 膽敢有違犯的，請以故意違抗罪處理。 \
              Example 3: [USER] 實在沒有辦法瞭，工匠就暗地裏讓妻子去見喻皓的妻子，給她送瞭金釵，求她嚮喻皓打聽木塔晃動的原因。\n這句話在古代怎麼說： [ASSISTANT] 無可奈何，密使其妻見喻皓之妻，賂以金釵，問塔動之因。 \
              Example 4: [USER] 吾兵以義舉，往無不剋，烏用此物，使暴殄百姓哉！\n把這句話翻譯成現代文。[ASSISTANT] 我軍以義起事勇往直前，攻無不剋，何必動用這傢夥，讓他暴虐殘害百姓呢！ \
              Example 5: [USER] 五月甲申，京師雨雹，地震。\n把這句話翻譯成現代文。 [ASSISTANT] 五月初五，京城降冰雹，發生地震。\
              USER: {instruction} ASSISTANT: \
              "


def get_bnb_config() -> BitsAndBytesConfig:
    '''Get the BitsAndBytesConfig.'''
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )