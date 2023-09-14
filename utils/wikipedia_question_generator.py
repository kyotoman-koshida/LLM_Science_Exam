import logging
import openai
import os
import pandas as pd
import pickle
import random
import wikipediaapi
from utils import get_str_today
from utils.my_logger import MyLogger
from dotenv import load_dotenv

class WikipediaQuestionGenerator:
    """Wikipediaから指定の分野からランダムに記事を取得して多肢選択問題を作る"""

    def __init__(
            self, 
            category: dict[str: list[str]], 
            category_weight: list[float], 
            exclude_category:list[str], 
            prompt:str, 
            options_set: set,
            response_keys_set: set,
            seed: int=0
        ):
        self.category: dict[str: list[str]] = category
        """抽出したWikipediaのページのカテゴリをdict形式で格納したもの
        
        Note
        ----------
        重点的に抽出したいカテゴリがあるため、自身で大ジャンルを設定し辞書のキーとする。
        例えば、
        {"M": ["Category:Fields_of_mathematics", "Category:Physical_sciences"]}
        のように格納されている。
        """
        self.category_weight = category_weight
        """自分で設定した大ジャンルの重みつけ"""
        self.exclude_category = exclude_category
        """作問するために使わないカテゴリ"""
        self.prompt = prompt
        """多肢選択問題をOpenAIのAPIに作ってもらう時のプロンプト"""
        self.wiki_wiki = wikipediaapi.Wikipedia('MyProjectName (merlin@example.com)', 'en')
        """WikipediaのAPIを使用して記事を取得する際のインスタンス"""
        self.options_set = options_set
        """多肢選択問題における、各選択肢のラベル"""
        self.respons_keys_set = response_keys_set
        """多肢選択肢問題における、問題文と各選択肢と解答のラベル"""
        self.delimiter = "####"
        """区切り文字"""
        self.seed = seed
        """ランダムに記事を取得する際のシード"""
        self.logger = MyLogger()
        self.logger.setLevel(logging.INFO)
        """ロガーのインスタンス"""

    def gather_multiple_choice_question_dataset(
        self,
        pages_count: int,
        max_completion_attempts: int = 10,
        seen_pages: list = [],
        seed: int = 0
    ) -> tuple[list, list, list]:
        """多肢選択問題の元データを作成する
        
        Parameters
        ----------
        pages_count: int 
            作成する問題数(参照する記事数)
        max_completion_attempts: int
            サードパーティのAPI接続などによるエラーが起きた場合に、何度まで試行を続けるか
        seen_pages: list
            すでに問題を作るのに参照したページIDを格納する

        Returns
        ----------
        multiple_choice_questions: list
            作成した問題が格納されている
        seen_pages: list
            すでに問題を作るのに参照したページIDを格納する
        attempts_list: list
            エラーが出た後に解決するまでの試行回数のリスト
        seed: ランダムのシード

        Note:
        ----------
        このメソッドではpages_countの数だけWikipediaの記事を見て作問を行う。
        作問を行うたびに、outputディレクトリ以下に途中までの状態を保持する。
        何らかの異常が起きて途中終了した場合には、
        outputディレクトリ以下にあるpickle化したseen_pagesとログで確認できるseed値から
        作問を再開することができる。
        また、途中終了するまでに作問した問題もoutputディレクトリ以下にcsvで出力されている。
        """
        attempts_list = []
        multiple_choice_questions = []

        generated_count = 0
        attempts_count = 0
        iterate_count = 0
        while generated_count < pages_count:
            iterate_count += 1
            self.logger.info(f"generated_count:{generated_count}")
            seed = generated_count+attempts_count+iterate_count+seed
            wiki_text, page_id, page_title, category_label = self.__get_wiki_text(seen_pages=seen_pages, seed=seed, sentences_include=7)
            self.logger.info(f"\nStart multiple choice questions generation: page_id={page_id}, page_title={page_title}, category_label={category_label}")
            
            messages = self.__get_completion_messages(wiki_text)
            today = get_str_today()
            while True:
                try:
                    chatgpt_response = self.__get_completion_from_messages(messages)
                    mcq = eval(chatgpt_response)

                    if not isinstance(mcq, list) or not self.__is_correctly_formatted(mcq):
                        self.logger.warning("ChatGPTの回答フォーマットが正しくありません")
                        raise Exception

                    for i in range(len(mcq)):
                        mcq[i]["wiki_text"] = wiki_text
                        mcq[i]["page_id"] = page_id
                        mcq[i]["page_title"] = page_title
                        mcq[i]["category_label"] = category_label

                        if mcq[i]["answer"] in self.options_set:
                            continue
                        else:
                            # TODO: indexメソッドは、answerがリスト形式ではないとエラーを返すため注意
                            answ_indx = [v.lower() for v in mcq[i].values()].index(mcq[i]["answer"].lower())
                            mcq[i]["answer"] = list(mcq[i].keys())[answ_indx]

                    multiple_choice_questions += mcq
                    with open(f'../output/pickle/generate_question_{today}.pickle', 'wb') as f:
                        pickle.dump(multiple_choice_questions, f)
                    seen_pages.append(page_id)
                    generated_count += 1
                    self.logger.info(f"seed:{seed-1}")
                    # NOTE: 
                    # seed-1としているのは、while文の冒頭にiterate_count+=1をしているから。
                    # このイテレーションに関しての再現を行う時にgather_multible_choice_question_datasetに渡すseedは
                    # この行でのseedの値よりも1だけ小さくなくてはならない。
                    # iterate_count+=1をイテレートの最後に持ってくるのは同じWikipediaのページをループする可能性があり危険。
                    break
                except Exception:
                    attempts_count += 1
                    self.logger.warning("正しくChatGPTの回答が得られませんでした")
                    attempts_list.append(attempts_count)
                    if attempts_count > max_completion_attempts:
                        break

            with open(f'../output/pickle/seen_pages_{today}.pickle', 'wb') as f:
                pickle.dump(seen_pages, f)
        
        return multiple_choice_questions, seen_pages, attempts_list

    def __get_wiki_text(
        self, 
        seen_pages:list, 
        seed:int, 
        min_page_length:int=3, 
        sentences_include:int=3
        )->tuple[str,str,str,str]:
        """Wikipediaからランダムに記事を取得し、指定した分量の内容を抽出する
        
        Parameters
        ----------
        seen_pages: list
            すでに問題を作るのに参照したページIDを格納する
        seed: int
            ランダムにWikipediaの記事を取得するためのシード
        min_page_length: int
            Wikipediaの記事のうち、参照する最低限のページの分量(行)
        sentence_include: int
            取得したWikipediaの記事について、冒頭の何行を参照するか

        Returns
        ----------
        wiki_text: str
            取得したWikipediaの記事から抽出したテキスト
        wiki_page.pageid: str
            取得したWikipediaの記事のID
        wiki_page.title: str
            取得したWikipediaの記事のタイトル
        category_label: str
            取得したWikipediaの記事のジャンル(このジャンルは初めに自分で区分けしたジャンル)
            例：S→Science, M→Mathematics
        """
        while True:
            try:
                seed += 1# このseedの加算はエラーが起こりうる部分よりも先に行う必要がある
                wiki_page, category_label = self.__get_wiki_random_page(seed)
            except:
                self.logger.warning("get_wiki_random_page()に失敗")
                seed += 1
                continue

            if wiki_page.pageid in seen_pages:
                self.logger.warning("すでに見たページを参照")
                seed += 1
                continue

            page_sentences = wiki_page.text.split(". ")
            
            # 取得したWikipediaの記事が十分に長いかどうか
            if len(page_sentences) >= min_page_length:
                # トピックの主要な内容は、大抵はじめの(sentence_include)行内に書かれている
                wiki_text = ". ".join(page_sentences[:sentences_include]) + "."
                break
        
        return wiki_text, wiki_page.pageid, wiki_page.title, category_label
    
    def __get_wiki_random_page(self, seed:int, deep_subcategories=True)->tuple[str,str]:
        """
        Wikipediaから指定したカテゴリのページをランダムに拾ってくる

        Parameters
        ----------
        seed: int
            ランダムにWikipediaの記事を取得するためのシード
        deep_subcategories: bool
            カテゴリの中のサブカテゴリに進んでいく(True)か否(False)か
            
        Returns
        ----------
        selected_page: str
            選択されたWikipediaのページ情報
        category_label: str
            選択されたWikipediaの属する大カテゴリ
        """
        random.seed(seed)
        category_label, categories = random.choices(list(self.category.items()), weights=self.category_weight, k=1)[0]
        category = random.choice(categories)
        self.logger.info(f"category:{category}")
        category_page = self.wiki_wiki.page(category)
        for _ in range(100):
            # FIXME: while文ではなくfor文を使用しているのは、while文で処理が抜け出せなくなる場合があるため
            chosen_list = list(category_page.categorymembers.items())
            if deep_subcategories:
                category_list, page_list = self.__split_category_members(chosen_list)
                chosen_list = []
            else:
                category_list, page_list = [], []

            # NOTE: カテゴリまたはページリストのいずれかが空でない場合、
            # カテゴリとページリストのいずれをchosen_listとして選ぶのかは50%:50%でランダムに決定する。
            # Wikipediaのページには、カテゴリ(サブカテゴリ)のリストと記事のリストが両方とも羅列されている場合がある。
            # 例：https://en.wikipedia.org/wiki/Category:Applied_sciences
            if not (category_list or page_list) and not chosen_list:
                continue
            elif not category_list:
                chosen_list = page_list
            elif not page_list:
                chosen_list = category_list
            else:
                chosen_list = random.choice([category_list, page_list])

            # 作問するためのページはここで選択している
            selected_page_name, selected_page = random.choice(chosen_list)

            if not selected_page_name.startswith("Category"):
                break
            
            category_page = selected_page
        
        return selected_page, category_label
    
    def __split_category_members(self, members:list)->tuple[list,list]:
        """ Wikipediaのページに記載されているリンク(members)をカテゴリと記事に分け、それぞれをリストに格納して返す。

        Parameters
        ----------
        members:list
            Wikipediaのページに記載されているリンク

        Returns
        ----------
        category_list:list
            Wikipediaのページに記載されていたリンクのうち、カテゴリを集めたリスト
        page_list:list
            Wikipediaのページに記載されていたリンクのうち、記事を集めたリスト
        """
        category_list, page_list= [], []

        for member_name, member_page in members:
            if member_name.startswith('Category') and member_name not in self.exclude_category:
                category_list.append((member_name, member_page))
            else:
                page_list.append((member_name, member_page))
        
        return category_list, page_list
    
    def __is_correctly_formatted(self, mcq:list) -> bool:
        """ OpenAIから取得した回答のフォーマットが正しい(True)か否(False)か判定する

        Parameters
        ----------
        mcq: list
            OpenAIのAPIから取得した回答

        Returns
        ----------
        フォーマットが正しい(True)か否(False)か: bool
        """
        return all([
            len(el) == len(self.respons_keys_set) and self.respons_keys_set == set(list(el.keys()))
            for el in mcq
        ])

    def __get_completion_messages(self, wiki_text:str)-> str:
        """ CahtGPTに尋ねることができる形式のtextを返す

        Parameters
        ----------
        wiki_text: str
            Wikipediaの記事から抽出した文章

        Returns
        ----------
        OpenAIのAPIを利用するための形式のテキスト: str
        https://platform.openai.com/docs/api-reference/chat
        """
        return [  
            {
                'role':'system', 
                'content': self.prompt
            },    
            {
                'role':'user', 
                'content': f"{self.delimiter}{wiki_text}{self.delimiter}"
            },  
        ]

    def __get_completion_from_messages(
        self,
        messages: str, 
        model:str="gpt-3.5-turbo", 
        temperature:float=0.8, 
        max_tokens:int=3000
    )-> str:
        """ OpenAIのAPIを利用してChatGPTの回答を得る

        Parameters
        ----------
        messages: str
            ChatGPTに問いかける内容
        model: str
            どのGPTモデルを使うか
        temperature: float
            サンプリング温度。1に近いほどランダムで、0に近いほど決定論的。
        max_tokens: int
            評価に使うことのできる最大のトークン数

        Returns
        ----------
        ChatGPTの回答: str

        Note
        ----------
        このメソッドの引数についてはOpenAIのAPIを参照
        https://platform.openai.com/docs/api-reference/completions/create
        """

        dotenv_path = os.path.join("../config", '.env')
        load_dotenv(dotenv_path)

        openai.api_key = os.getenv("OPENAI_API_KEY")
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature, 
            max_tokens=max_tokens, 
        )
        # NOTE: 累計の課金額はOpenAIのページ(https://platform.openai.com/account/usage)で確認
        self.logger.info("課金されました")
        #NOTE: 少し前まではresponse.choices[0].message["content"]だったが、これだとエラーが出るようになった。
        # OpenAIのAPIのフォーマットが変わった？
        return response["choices"][0]["message"]["content"]
    
    @staticmethod
    def conver_to_compet_format_df(multiple_choice_questions: list)-> pd.DataFrame:
        """ ChatGPTから取得した回答群のリストをデータフレームに変換する

        Parameters
        ----------
        multiple_choice_questions: list
            ChatGPTから取得した回答群のリスト
        Returns
        ----------
        ChatGPTの回答のリストをデータフレーム化したもの: pd.DataFrame
        """
        df_mcq  = pd.DataFrame.from_records(multiple_choice_questions)

        df_compet = df_mcq.copy(deep=True)
        df_compet.insert(0, "id", list(range(len(df_compet))))
        df_compet.rename(
            columns = {
                'question': 'prompt', 
                'option_1': 'A', 
                'option_2': 'B', 
                'option_3': 'C', 
                'option_4': 'D', 
                'option_5': 'E'
            }, 
            inplace = True
        )

        answer_subjects = {
            'option_1': 'A', 
            'option_2': 'B', 
            'option_3': 'C', 
            'option_4': 'D', 
            'option_5': 'E'
        }
        df_compet["answer"] = df_compet["answer"].map(answer_subjects)
        df_compet = df_compet.drop(columns=["wiki_text", "page_id", "page_title", "category_label"])

        df_compet.to_csv(f"../output/dataset/stem_dataset_{get_str_today()}.csv", index=False)
        return df_compet
    