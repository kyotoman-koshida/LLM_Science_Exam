import datetime

def get_str_today():
    now = datetime.datetime.now() # 現在時刻の取得
    str_today = now.strftime('%Y_%m_%d') # 現在時刻を年月日で表示
    return str_today