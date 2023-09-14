from dataclasses import dataclass, field
from typing import List, Dict, Set

@dataclass
class WikipediaCategory:
    CATEGORY_WEIGHTS: List[float] = field(default_factory=lambda: [1.25, 1, 1, 1])
    """カテゴリーの大分野を抽出する際の割合"""

    CATEGORY: Dict[str, List[str]] = field(default_factory=lambda: {
        "S": ["Category:Applied_sciences", "Category:Biotechnology", "Category:Biology", "Category:Natural_history"],
        "T": [
            "Category:Technology_strategy", "Category:Technical_specifications", "Category:Technology_assessment", 
            "Category:Technology_hazards", "Category:Technology_systems", "Category:Hypothetical_technology", 
        
            "Category:Technical_communication", "Category:Technological_comparisons"
        ],
        "E": ["Category:Engineering_disciplines", "Category:Engineering_concepts", "Category:Industrial_equipment", "Category:Manufacturing"],
        "M": ["Category:Fields_of_mathematics", "Category:Physical_sciences"]
        })
    """Wikipediaから記事を取得する分野のカテゴリ"""

    EXCLUDE_CATEGORIES: Set[str] = field(default_factory=lambda: set([
        "Category:Technology", "Category:Mathematics", "Category:Works about technology", 
        "Category:Technology evangelism", "Category:Artificial objects", "Category:Fictional physical scientists"
    ]))
    """Wikipediaから記事を取得する分野のサブカテゴリのうち除外するカテゴリ"""

    OPTIONS_SET: Set[str] = field(default_factory=lambda: set(("option_1", "option_2", "option_3", "option_4", "option_5")))
    """"""

    RESPONSE_KEYS_SET: Set[str] = field(default_factory=lambda: set(("question", "option_1", "option_2", "option_3", "option_4", "option_5", "answer")))
    """"""
