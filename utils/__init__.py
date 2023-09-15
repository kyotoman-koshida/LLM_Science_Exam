import datetime

def get_str_today():
    now = datetime.datetime.now() # 現在時刻の取得
    str_now = now.strftime('%Y%m%d%H%M%S') # 現在時刻を年月日で表示
    return str_now