import logging
from posixpath import dirname
import sys
import os
from utils import get_str_today

class MyLogger() :
    '''
    ログ管理クラス
    '''
    def __init__(self,filename = '',format = '',level=logging.INFO,stdout = False):
        '''
        コンストラクタ
        
        Parameters
        ----------
        filename:str 
            ログファイルのフルパス
        format:str
            ログの出力フォーマット
        level:int
            ログの出力レベル
            DEBUG=10  INFO=20 WARINNG=30 ERROR=40 CRITICAL=50
        stdout:bool
            ログを標準出力に表示するか否かの設定 
        '''
        
        #ログファイル名が指定されていなければ、生成する
        self.filename = self.__create_log_path() if filename == '' else filename
        #フォーマットが指定されていなければ、初期値を設定する。
        self.format = '[%(asctime)s] %(name)s (%(levelname)s) %(message)s' if format == '' else format
        
        #ロガーを生成
        self.logger = logging.getLogger(os.path.basename(__file__))
        #エラーレベルを設定
        self.logger.setLevel(level)
        #ハンドラを登録
        self.set_handler(self.filename,self.format,stdout)
    
    def set_handler(self,filename,format,stdout = False):
        '''
        ハンドラの登録
        
        Parameters
        ----------
        filename:str 
            ログファイルのフルパス
        format:str
            ログの出力フォーマット
        stdout:bool
            ログを標準出力に表示するか否かの設定 
        '''
        fmt = logging.Formatter(format)    #フォーマッタの作成
        fh = logging.FileHandler(filename) #ファイルハンドラの生成
        fh.setFormatter(fmt)               #ファイルハンドラにフォーマッタを登録
        self.logger.addHandler(fh)         #ロガーにファイルハンドラを登録
        
        #標準出力の出力指示がされていたら、標準出力用のハンドラとフォーマッタを登録
        if stdout:
            sd = self.logger.StreamHandler(sys.stdout)
            sd.setFormatter(fmt)     
            self.logger.addHandler(sd)
    
    def setLevel(self,level):
        '''
        ログに出力するエラーレベルを設定する
        '''
        self.logger.setLevel(level)
    
    def info(self,message):
        '''
        ログに情報メッセージを出力する
        '''
        self.logger.info(message)
    
    def error(self,message):
        '''
        ログにエラーメッセージを出力する
        '''
        self.logger.error(message)
    
    def exception(self,message):
        '''
        ログに例外処理のメッセージを付加して出力する
        '''
        self.logger.exception(message)
    
    def warning(self,message):
        '''
        ログに警告メッセージを出力する
        '''
        self.logger.warning(message)
    
    def debug(self,message):
        '''
        ログにデバッグメッセージを出力する
        '''
        self.logger.debug(message)

    def critical(self,message):
        '''
        ログにクリティカルメッセージを出力する
        '''
        self.logger.critical(message)
    
    def __create_log_path(self):
        '''
        __file__ の内容を使って、ログファイルのパスの初期値を生成する
        '''
        folder = os.path.join(self.__get_parent(__file__),'code_analysis/logs')
        today = get_str_today()
        name_without_ext = self.__get_name_without_ext(f'log_{today}')
        return os.path.join(folder,name_without_ext+'.log')
    
    def __get_parent(self,path):
        '''
        指定したパスの１つ上のフォルダ（親フォルダ）を返す
        '''
        return '/'.join(os.path.dirname(path).replace('\\','/').split('/')[0:-1])
    
    def __get_name_without_ext(self,filename):
        '''
        拡張子を除いたファイル名を返す
        '''
        return os.path.splitext(os.path.basename(filename))[0]