from __future__ import annotations

import re
import spacy

from typing import Final
from nltk.corpus import words
from pandas import DataFrame, Series
from spacy.language import Language
from tqdm import tqdm

from helpers.StopwordFilter import StopwordFilter

# run first
# python -m spacy download en_core_web_lg
ENGLISH_VOCAB = set(w.lower() for w in words.words())


class LyricsProcessor:
    _TOKEN_PATTERN: Final = re.compile(r"\b[\w']+\b")

    _MISSING_APOSTROPHE_MAP: Final[dict[str, str]] = {
        # 't cases
        r"\b(ain)\s*t\b": r"\1't",
        r"\b(can)\s*t\b": r"\1't",
        r"\b(couldn)\s*t\b": r"\1't",
        r"\b(didn)\s*t\b": r"\1't",
        r"\b(doesn)\s*t\b": r"\1't",
        r"\b(don)\s*t\b": r"\1't",
        r"\b(hadn)\s*t\b": r"\1't",
        r"\b(hasn)\s*t\b": r"\1't",
        r"\b(haven)\s*t\b": r"\1't",
        r"\b(isn)\s*t\b": r"\1't",
        r"\b(shouldn)\s*t\b": r"\1't",
        r"\b(wasn)\s*t\b": r"\1't",
        r"\b(weren)\s*t\b": r"\1't",
        r"\b(won)\s*t\b": r"\1't",
        r"\b(wouldn)\s*t\b": r"\1't",
        # 's cases
        r"\b(he)\s*s\b": r"\1's",
        r"\b(how)\s*s\b": r"\1's",
        r"\b(it)\s*s\b": r"\1's",
        r"\b(let)\s*s\b": r"\1's",
        r"\b(she)\s*s\b": r"\1's",
        r"\b(that)\s*s\b": r"\1's",
        r"\b(there)\s*s\b": r"\1's",
        r"\b(what)\s*s\b": r"\1's",
        r"\b(where)\s*s\b": r"\1's",
        r"\b(who)\s*s\b": r"\1's",
        # 're cases
        r"\b(you)\s*re\b": r"\1're",
        r"\b(they)\s*re\b": r"\1're",
        r"\b(we)\s*re\b": r"\1're",
        # 'll cases
        r"\b(i)\s*ll\b": r"\1'll",
        r"\b(you)\s*ll\b": r"\1'll",
        r"\b(he)\s*ll\b": r"\1'll",
        r"\b(she)\s*ll\b": r"\1'll",
        r"\b(we)\s*ll\b": r"\1'll",
        r"\b(they)\s*ll\b": r"\1'll",
        r"\b(who)\s*ll\b": r"\1'll",
        # 've cases
        r"\b(i)\s*ve\b": r"\1've",
        r"\b(you)\s*ve\b": r"\1've",
        r"\b(we)\s*ve\b": r"\1've",
        r"\b(they)\s*ve\b": r"\1've",
        r"\b(who)\s*ve\b": r"\1've",
        # 'd cases
        r"\b(i)\s*d\b": r"\1'd",
        r"\b(you)\s*d\b": r"\1'd",
        r"\b(he)\s*d\b": r"\1'd",
        r"\b(she)\s*d\b": r"\1'd",
        r"\b(we)\s*d\b": r"\1'd",
        r"\b(they)\s*d\b": r"\1'd",
        r"\b(who)\s*d\b": r"\1'd",
    }

    _CONTRACTION_MAP: Final[dict[str, str]] = {
        "ain't": "am not",
        "aren't": "are not",
        "ev'rybody": "everybody",
        "ev'ry": "every",
        "can't": "cannot",
        "can't've": "cannot have",
        "could've": "could have",
        "couldn't": "could not",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he would",
        "he'll": "he will",
        "he's": "he is",
        "she'd": "she would",
        "she'll": "she will",
        "she's": "she is",
        "they'd": "they would",
        "they'll": "they will",
        "they're": "they are",
        "they've": "they have",
        "how'd": "how did",
        "how'll": "how will",
        "how's": "how is",
        "i'd": "i would",
        "i'll": "i will",
        "i'm": "i am",
        "i've": "i have",
        "isn't": "is not",
        "it'd": "it would",
        "it'll": "it will",
        "it's": "it is",
        "let's": "let us",
        "prob'ly": "probably",
        "shouldn't": "should not",
        "somebody's": "somebody is",
        "someone's": "someone is",
        "that's": "that is",
        "there's": "there is",
        "wasn't": "was not",
        "we'd": "we would",
        "we'll": "we will",
        "we're": "we are",
        "we've": "we have",
        "weren't": "were not",
        "what's": "what is",
        "where's": "where is",
        "who'd": "who would",
        "who'll": "who will",
        "who're": "who are",
        "who's": "who is",
        "who've": "who have",
        "won't": "will not",
        "wouldn't": "would not",
        "why's": "why is",
        "y'all": "you all",
        "you'd": "you would",
        "you'll": "you will",
        "you're": "you are",
        "you've": "you have",
        "whaddya": "what you",
        "whassup": "what is up",
        "whatchu": "what you",
        "whatcha": "what you",
        "whatta": "what a",
        "wunna": "want to",
        "imma": "i am going to",
        "gonna": "going to",
        "wanna": "want to",
        "gotta": "got to",
        "i'ma": "i am going to",
        "cmon": "come on",
        "dyou": "do you",
        "yknow": "you know",
        "tryna": "trying to",
        "gimme": "give me",
        "o'er": "over",
    }

    _DOMAIN_LEXICON: Final[dict[str, str]] = {
        "amazin": "amazing",
        "actin": "acting",
        "ballin": "balling",
        "bangin": "banging",
        "beatin": "beating",
        "dawg": "dog",
        "beefin": "beefing",
        "beginnin": "beginning",
        "behaviour": "behavior",
        "believin": "believing",
        "betta": "better",
        "achin": "ache",
        "biz": "business",
        "bitchin": "bitching",
        "bitin": "biting",
        "blastin": "blasting",
        "bleedin": "bleeding",
        "blessin": "blessing",
        "blowin": "blowing",
        "bloomin": "blooming",
        "bummin": "bumming",
        "bussin": "bussing",
        "busta": "buster",
        "bustin": "busting",
        "boomin": "booming",
        "bluffin": "bluffing",
        "bouncnin": "bouncing",
        "braggin": "bragging",
        "breakin": "breaking",
        "breathin": "breathing",
        "bringin": "bringing",
        "buggin": "bugging",
        "bullshittin": "bullshitting",
        "buildin": "building",
        "bumpin": "bumping",
        "burnin": "burning",
        "buyin": "buying",
        "buzzin": "buzzing",
        "callin": "calling",
        "carryin": "carrying",
        "catchin": "catching",
        "causin": "causing",
        "cuz": "because",
        "coz": "because",
        "'cause": "because",
        "ceilin": "ceiling",
        "chasin": "chasing",
        "cheatin": "cheating",
        "checkin": "checking",
        "chokin": "choking",
        "clappin": "clapping",
        "cliché": "cliche",
        "climbin": "climbing",
        "clutchin": "clutching",
        "comin": "coming",
        "controllin": "controlling",
        "cookin": "cooking",
        "coolin": "cooling",
        "countin": "counting",
        "crackin": "cracking",
        "crashin": "crashing",
        "crawlin": "crawling",
        "creepin": "creeping",
        "cruisin": "cruising",
        "cryin": "crying",
        "cuttin": "cutting",
        "dancin": "dancing",
        "darlin": "darling",
        "dealin": "dealing",
        "diggin": "digging",
        "droppin": "dropping",
        "dippin": "dipping",
        "dissin": "dissing",
        "doin": "do",
        "draggin": "dragging",
        "dreamin": "dreaming",
        "drinkin": "drinking",
        "drippin": "dripping",
        "dropin": "dropping",
        "duckin": "ducking",
        "eatin": "eating",
        "evenin": "evening",
        "facin": "facing",
        "fakin": "faking",
        "fallin": "falling",
        "feedin": "feeding",
        "fiendin": "fiending",
        "fishin": "fishing",
        "flashin": "flashing",
        "flippin": "flipping",
        "floatin": "floating",
        "flossin": "flossing",
        "flowin": "flowing",
        "flyin": "flying",
        "foolin": "fooling",
        "freakin": "freaking",
        "fronin": "fronting",
        "fuckin": "fucking",
        "gettin": "getting",
        "givin": "giving",
        "goin": "going",
        "grabbin": "grabbing",
        "grindin": "grinding",
        "grippin": "gripping",
        "groovin": "grooving",
        "growin": "growing",
        "gunnin": "gunning",
        "hangin": "hanging",
        "happenin": "happening",
        "hatin": "hating",
        "havin": "having",
        "headin": "heading",
        "healin": "healing",
        "hearin": "hearing",
        "hidin": "hiding",
        "holdin": "holding",
        "homies": "homie",
        "hoppin": "hopping",
        "howlin": "howling",
        "hummin": "humming",
        "hunin": "hunting",
        "hurtin": "hurting",
        "hustlin": "hustling",
        "jamin": "jamming",
        "judgement": "judgment",
        "jumpin": "jumping",
        "keepin": "keeping",
        "kickin": "kicking",
        "kiddin": "kidding",
        "kissin": "kissing",
        "knockin": "knocking",
        "knowin": "knowing",
        "knew": "knowing",
        "laughin": "laughing",
        "blinkin": "blinking",
        "loungin": "lounging",
        "layin": "laying",
        "leanin": "leaning",
        "leavin": "leaving",
        "lettin": "letting",
        "lickin": "licking",
        "lov": "love",
        "liv": "live",
        "maama": "mama",
        "livin": "living",
        "lightnin": "lightning",
        "listenin": "listening",
        "lookin": "looking",
        "losin": "losing",
        "lyin": "lying",
        "marchin": "marching",
        "needin": "needing",
        "nothin": "nothing",
        "packin": "packing",
        "payin": "paying",
        "passin": "passing",
        "pimpin": "pimping",
        "poppin": "popping",
        "pourin": "pouring",
        "prayin": "praying",
        "preachin": "preaching",
        "pretendin": "pretending",
        "puffin": "puffing",
        "pullin": "pulling",
        "pummpin": "pumping",
        "puttin": "putting",
        "racin": "racing",
        "thumpin": "thumping",
        "throwin": "throwing",
        "rainin": "raining",
        "textin": "texting",
        "rappin": "rapping",
        "rhymin": "rhyming",
        "ridin": "riding",
        "risin": "rising",
        "robbin": "robbing",
        "rushin": "rushing",
        "schemin": "scheming",
        "screamin": "screaming",
        "searchin": "searching",
        "settin": "setting",
        "shinin": "shining",
        "shootin": "shooting",
        "sippin": "sipping",
        "slammin": "slamming",
        "slippin": "slipping",
        "sniffin": "sniffing",
        "speedin": "speeding",
        "spillin": "spilling",
        "suckin": "sucking",
        "swingin": "swinging",
        "taking": "taking",
        "tickin": "ticking",
        "trippin": "tripping",
        "tumblin": "tumbling",
        "wantin": "wanting",
        "whippin": "whipping",
        "wishin": "wishing",
        "wonderin": "wondering",
        "workin": "working",
        "worryin": "worrying",
        "writin": "writing",
        "yellin": "yelling",
        "seein": "seeing",
        "sellin": "selling",
        "plannin": "planning",
        "timin": "timing",
        "totin": "toting",
        "driftin": "drifting",
        "thuggin": "thugging",
        "touchin": "touching",
        "pushin": "pushing",
        "pumpin": "pumping",
        "plottin": "plotting",
        "aight": "alright",
        "lil": "little",
        "til": "until",
        "missin": "missing",
        "mixin": "mixing",
        "mornin": "morning",
        "movin": "moving",
        "walkin": "walking",
        "runnin": "running",
        "talkin": "talking",
        "tryin": "trying",
        "slangin": "slanging",
        "dyin": "dying",
        "beggin": "begging",
        "killa": "killer",
        "killas": "killer",
        "killin": "killing",
        "drivin": "driving",
        "makin": "making",
        "shiftin": "shifting",
        "supa": "super",
        "hittin": "hitting",
        "sittin": "sitting",
        "chillin": "chilling",
        "smokin": "smoking",
        "playin": "playing",
        "praying": "praying",
        "ramblin": "rambling",
        "sailin": "sailing",
        "savin": "saving",
        "shootin ": "shooting",
        "shoppin": "shopping",
        "sinkin": "sinking",
        "skippin": "skipping",
        "slidin": "sliding",
        "sleepin": "sleeping",
        "slowin": "slowing",
        "smilin": "smiling",
        "sneakin": "sneaking",
        "soakin": "soaking",
        "somethin": "something",
        "lovin": "loving",
        "speakin": "speaking",
        "spendin": "spending",
        "spreadin": "spreading",
        "stackin": "stacking",
        "standin": "standing",
        "starin": "staring",
        "stayin": "staying",
        "swimmin": "swimming",
        "stealin": "stealing",
        "sweatin": "sweating",
        "steppin": "stepping",
        "stickin": "sticking",
        "stinkin": "stinking",
        "stompin": "stomping",
        "stoppin": "stopping",
        "bein": "being",
        "wearin": "wearing",
        "wavin": "waving",
        "waitin": "waiting",
        "strugglin": "struggling",
        "stumblin": "stumbling",
        "stuntin": "stunting",
        "stylin": "styling",
        "suckas": "sucker",
        "switchin": "switching",
        "tellin": "telling",
        "thinkin": "thinking",
        "travlin": "traveling",
        "tricklin": "trickling",
        "feelin": "feeling",
        "winnin": "winning",
        "sayin": "saying",
        "wreckin": "wrecking",
        "nigger": "nigga",
        "niggers": "nigga",
        "niggas": "nigga",
        "niggaz": "nigga",
        "boyz": "boy",
        "caldonia": "caledonia",
        "muzik": "music",
        "nuthin": "nothing",
        "weighin": "weighing",
        "wetin": "weting",
        "oer": "over",
        "yappin": "yapping",
        "yougin": "younging",
        "gyal": "girl",
        "gyals": "girl",
    }

    _VOCALIZATIONS: Final[frozenset[str]] = frozenset(
        {
            "ahh",
            "ahhh",
            "aaaah",
            "aaaaah",
            "aaow",
            "aaah",
            "aah",
            "aye",
            "ohh",
            "ohhh",
            "ohhhh",
            "uhh",
            "mmh",
            "mmm",
            "mmmm",
            "hmm",
            "hmmm",
            "hoo",
            "ooh",
            "oooh",
            "ooooh",
            "ayy",
            "ayyyy",
            "yeah",
            "yea",
            "yeh",
            "jah",
            "yuh",
            "dah",
            "huh",
            "hah",
            "haha",
            "hahah",
            "hahahaha",
            "hmmm",
            "hmm",
            "aww",
            "bae",
            "bah",
            "bam",
            "boo",
            "buh",
            "baow",
            "boom",
            "brr",
            "brrt",
            "doo",
            "daa",
            "dada",
            "deh",
            "dem",
            "den",
            "dum",
            "doh",
            "duh",
            "eah",
            "eee",
            "eeh",
            "err",
            "grr",
            "grrah",
            "guh",
            "haaa",
            "hee",
            "hem",
            "hoh",
            "laa",
            "laaa",
            "lala",
            "lalala",
            "lalalalala",
            "mca",
            "mcs",
            "meh",
            "mhh",
            "mwah",
            "naa",
            "nuh",
            "mmmm",
            "mmmmm",
            "ohoh",
            "oohhh",
            "ooo",
            "oho",
            "ohooho",
            "ohoho",
            "ola",
            "ole",
            "ooo",
            "oooo",
            "ooooo",
            "oooooohhhh",
            "oww",
            "owww",
            "tlc",
            "uuuh",
            "rrr",
            "skkr",
            "skrrt",
            "skrt",
            "schalalalala",
            "soo",
            "tss",
            "uuh",
            "waah",
            "shhhh",
            "ugh",
            "wha",
            "weh",
            "wam",
            "whoa",
            "yah",
            "yap",
            "woa",
            "woah",
            "woof",
            "wooh",
            "woop",
            "wop",
            "wuh",
            "yaa",
            "yallah",
            "yeah",
            "yeahh",
            "yee",
            "yep",
            "yay",
            "yoh",
            "yon",
            "yow",
            "yoyo",
            "yup",
            "yum",
            "zee",
        }
    )

    _NOUN_TAGS: Final[frozenset[str]] = frozenset({"NOUN", "PROPN"})
    _SENTIMENT_TAGS: Final[frozenset[str]] = frozenset({"ADJ", "ADV"})

    def __init__(
        self,
        corpus: DataFrame,
        lyrics_column: str = "lyrics",
        model: str = "en_core_web_lg",
    ) -> None:
        self.corpus = corpus
        self.lyrics_column = lyrics_column
        self.nlp = self._load_spacy_model(model)
        self.stopword_filter = StopwordFilter()

    def process(self, batch_size: int = 500, n_process: int = -1) -> DataFrame:
        """Process lyrics to create topic and sentiment versions using parallel processing.

        Args:
            batch_size (int): The number of texts to buffer.
            n_process (int): The number of cores to use. -1 means all available cores.

        Returns:
            DataFrame with columns: lemmatized_pos_lyrics, topic_lyrics, sentiment_lyrics
        """
        out = self.corpus.copy()

        print("Fixing missing apostrophes in lyrics...")
        expressions_texts = (
            out[self.lyrics_column].astype(str).map(self._fix_missing_apostrophes)
        )

        print("Expanding contractions in lyrics...")
        texts = expressions_texts.map(self._expand_contractions)

        print("Applying domain lexicon...")
        texts = self._apply_domain_lexicon_to_series(texts)

        print(
            f"Lemmatizing lyrics in parallel (n_process={n_process}, batch_size={batch_size})..."
        )

        # Process texts in batches using nlp.pipe for performance
        docs = self.nlp.pipe(texts, batch_size=batch_size, n_process=n_process)

        # Use a list comprehension for a concise and readable way to process docs
        processed_lyrics = [
            self._process_doc(doc) for doc in tqdm(docs, total=len(out))
        ]

        # Unpack the list of tuples into three new columns
        (
            out["lemmatized_pos_lyrics"],
            out["topic_lyrics"],
            out["sentiment_lyrics"],
        ) = zip(*processed_lyrics)

        print("Removing stopwords from topic and sentiment lyrics...")
        out["topic_lyrics"] = out["topic_lyrics"].map(self._remove_stopwords)
        out["sentiment_lyrics"] = out["sentiment_lyrics"].map(self._remove_stopwords)
        out["expression_lyrics"] = expressions_texts.str.lower()

        return out

    def _fix_missing_apostrophes(self, text: str) -> str:
        """Fix common contractions missing apostrophes."""
        if not text or not isinstance(text, str):
            return text

        for pattern, replacement in self._MISSING_APOSTROPHE_MAP.items():
            suffix = replacement.split("'")[1]  # Extract "'t", "'s", or "'re"

            def replace_with_case(match):
                word = match.group(1)
                if word.isupper():
                    return word + "'" + suffix.upper()
                elif word and word[0].isupper():
                    return word + "'" + suffix
                else:
                    return word + "'" + suffix

            text = re.sub(pattern, replace_with_case, text, flags=re.IGNORECASE)

        return text

    def _expand_contractions(self, text: str) -> str:
        """Expand English contractions in text."""
        if not text or not isinstance(text, str):
            return text

        pattern = re.compile(
            r"\b("
            + "|".join(re.escape(key) for key in self._CONTRACTION_MAP.keys())
            + r")\b",
            re.IGNORECASE,
        )

        def replace(match):
            contraction = match.group(0)
            return self._CONTRACTION_MAP.get(
                contraction.lower(), self._CONTRACTION_MAP.get(contraction, contraction)
            )

        return pattern.sub(replace, text)

    def _process_doc(self, doc: spacy.tokens.Doc) -> tuple[str, str, str]:
        """Process a single spaCy Doc to lemmatize and extract POS-specific tokens."""
        lemmas_with_pos: list[str] = []
        topic_lemmas: list[str] = []
        sentiment_lemmas: list[str] = []

        for token in doc:
            lemma = token.lemma_.lower()

            if (
                lemma in ENGLISH_VOCAB
                and lemma not in self._VOCALIZATIONS
                and not token.is_punct
                and not token.is_space
                and not token.is_oov
                and not token.is_stop
                and lemma.isalpha()
                and len(lemma) > 1
                and not any(c.isdigit() for c in lemma)
            ):
                pos = token.pos_
                lemmas_with_pos.append(f"{lemma}/{pos}")

                if pos in self._NOUN_TAGS and len(lemma) > 2:
                    topic_lemmas.append(lemma)
                elif pos in self._SENTIMENT_TAGS and len(lemma) > 2:
                    sentiment_lemmas.append(lemma)

        return (
            " ".join(lemmas_with_pos),
            " ".join(topic_lemmas),
            " ".join(sentiment_lemmas),
        )

    def _remove_stopwords(self, text: str) -> str:
        """Remove stopwords from extracted text."""
        if not text or not isinstance(text, str):
            return ""

        tokens = text.split()
        # Logic is now fully consolidated in StopwordFilter
        filtered_tokens = [
            token for token in tokens if not self.stopword_filter.is_stopword(token)
        ]
        return " ".join(filtered_tokens)

    @staticmethod
    def _load_spacy_model(model: str) -> Language:
        """Load a spaCy model, downloading it if necessary."""
        try:
            return spacy.load(model)
        except OSError:
            print(f"Spacy model '{model}' not found. Downloading...")
            spacy.cli.download(model)
            return spacy.load(model)

    def _apply_domain_lexicon_to_series(self, texts: Series) -> Series:
        """
        Apply the domain lexicon to a pandas Series of texts, preserving case.
        If the matched word is capitalized, the replacement will be capitalized.
        """
        for original, replacement in self._DOMAIN_LEXICON.items():

            def case_sensitive_replace(match):
                matched_word = match.group(0)
                if matched_word[0].isupper():
                    return replacement.capitalize()
                return replacement

            texts = texts.str.replace(
                r"\b" + re.escape(original) + r"\b",
                case_sensitive_replace,
                regex=True,
                flags=re.IGNORECASE,
            )
        return texts
