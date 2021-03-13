import pandas as pd
import re

def make_trans():
    a = 'a b c d e f g h i j k l m n o p q r s t u v w x y z'.split()
    b = 'а в с д е ф г н и ж к л м н о р к р с т у в в х у з'.split()
    trans_dict = dict(zip(a, b))
    trans_table = ''.join(a).maketrans(trans_dict)
    return trans_table

def normalize(ser: pd.Series):
#   "СокДобрый" -> "Сок Добрый"
    camel_case_pat = re.compile(r'([а-яa-z])([А-ЯA-Z])')
#   "lmno" -> "лмно"
    trans_table = make_trans()
#   "14х15х30" -> "DxDxD"
    dxdxd_pat = re.compile(r'((?:\d+\s*[х\*]\s*){2}\d+)')
#   "1.2 15,5" -> "1p2 15p5"  
    digit_pat = re.compile(r'(\d+)[\.,](\d+)')
#   "15 мл" -> "15мл"
    unit = 'мг|г|гр|кг|мл|л|шт'
    unit_pat = re.compile(fr'(\d+)\s+({unit})\b')
#   "ж/б ст/б" -> "жб стб"
    w_w_pat = re.compile(r'\b([а-я]{1,2})/([а-я]{1,2})\b')
#   "a b c d" -> "abcd"
    glue_pat = re.compile(r'(?<=(?<!\w)\w) (?=\w(?!\w))', re.UNICODE)
    
    return ser \
            .str.replace(camel_case_pat, r'\1 \2') \
            .str.lower() \
            .str.replace(r'ъ\b', '') \
            .str.replace('ё', 'е') \
            .str.translate(trans_table) \
            .str.replace(dxdxd_pat, ' DxDxD ') \
            .str.replace('№', ' NUM ') \
            .str.replace('%', ' PERC ') \
            .str.replace(digit_pat, r' \1p\2 ') \
            .str.replace(unit_pat, r'\1\2 ') \
            .str.replace(w_w_pat, r' \1\2 ') \
            .str.replace(r'[\W_]', ' ') \
            .str.replace(glue_pat, '')