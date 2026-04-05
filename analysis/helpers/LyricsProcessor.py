from __future__ import annotations

import re
from typing import Final

import spacy
from pandas import DataFrame
from spacy.language import Language
from tqdm import tqdm

from helpers.StopwordFilter import StopwordFilter

# run first
# python -m spacy download en_core_web_sm


class LyricsProcessor:
    _TOKEN_PATTERN: Final = re.compile(r"\b[\w']+\b")
    _CONTRACTION_MAP: Final[dict[str, str]] = {
        "ain't": "am not",
        "aren't": "are not",
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
        "they'd": "they would",
        "they'll": "they will",
        "they're": "they are",
        "they've": "they have",
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
    }

    _DOMAIN_LEXICON: Final[dict[str, str]] = {
        "amazin": "amazing",
        "actin": "act",
        "bitches": "bitch",
        "ballin": "ball",
        "bangin": "bang",
        "beatin": "beat",
        "becomes": "become",
        "beefin": "beef",
        "beginnin": "begin",
        "behaviour": "behavior",
        "being": "be",
        "believin": "believe",
        "betta": "better",
        "achin": "ache",
        "biz": "business",
        "bitchin": "bitch",
        "bitin": "bite",
        "blastin": "blast",
        "bleedin": "bleed",
        "bleeds": "bleed",
        "blessin": "bless",
        "blowin": "blow",
        "bloomin": "bloom",
        "bummin": "bum",
        "bussin": "buss",
        "busta": "buster",
        "boomin": "boom",
        "bluffin": "bluff",
        "bouncnin": "bounce",
        "braggin": "brag",
        "breakin": "break",
        "breathin": "breathe",
        "bringin": "bring",
        "brings": "bring",
        "buggin": "bug",
        "bullshittin": "bullshit",
        "buildin": "build",
        "bumpin": "bump",
        "burnin": "burn",
        "bustin": "bust",
        "buyin": "buy",
        "buzzin": "buzz",
        "callin": "call",
        "carryin": "carry",
        "catchin": "catch",
        "causin": "cause",
        "cuz": "because",
        "coz": "because",
        "'cause": "because",
        "ceilin": "ceil",
        "chasin": "chase",
        "cheatin": "cheat",
        "checkin": "check",
        "chokin": "choke",
        "clappin": "clap",
        "cliché": "cliche",
        "climbin": "climb",
        "clutchin": "clutch",
        "comin": "come",
        "controllin": "control",
        "cookin": "cook",
        "coolin": "cool",
        "countin": "count",
        "crackin": "crack",
        "crashin": "crash",
        "crawlin": "crawl",
        "creepin": "creep",
        "cruisin": "cruise",
        "cryin": "cry",
        "cuttin": "cut",
        "dancin": "dance",
        "darlin": "darling",
        "dealin": "deal",
        "diggin": "dig",
        "droppin": "drop",
        "dippin": "dip",
        "dissin": "diss",
        "doin": "do",
        "draggin": "drag",
        "dreamin": "dream",
        "drinkin": "drink",
        "drippin": "drip",
        "dropin": "drop",
        "duckin": "duck",
        "eatin": "eat",
        "evenin": "evening",
        "eyes": "eye",
        "facin": "face",
        "fakin": "fake",
        "fallin": "fall",
        "feedin": "feed",
        "fiendin": "fiend",
        "fishin": "fish",
        "flashin": "flash",
        "flippin": "flip",
        "floatin": "float",
        "flossin": "floss",
        "flowin": "flow",
        "flyin": "fly",
        "foolin": "fool",
        "freakin": "freak",
        "friends": "friend",
        "fronin": "front",
        "fuckin": "fucking",
        "gettin": "get",
        "girlies": "girlie",
        "givin": "give",
        "goin": "go",
        "grabbin": "grab",
        "grindin": "grind",
        "grippin": "grip",
        "groovin": "groove",
        "growin": "grow",
        "gunnin": "gun",
        "hangin": "hang",
        "happenin": "happen",
        "hatin": "hate",
        "havin": "have",
        "headin": "head",
        "healin": "heal",
        "hearin": "hear",
        "hidin": "hide",
        "hmm": "hm",
        "hmmm": "hm",
        "holdin": "hold",
        "homies": "homie",
        "hoppin": "hop",
        "howlin": "howl",
        "hummin": "humm",
        "hunin": "hunt",
        "hurtin": "hurt",
        "hustlin": "hustle",
        "jamin": "jam",
        "judgement": "judgment",
        "jumpin": "jump",
        "keepin": "keep",
        "kickin": "kick",
        "kiddin": "kid",
        "kissin": "kiss",
        "knockin": "knock",
        "knowin": "know",
        "knew": "know",
        "laughin": "laugh",
        "loungin": "lounge",
        "layin": "lay",
        "leanin": "lean",
        "leavin": "leave",
        "lettin": "let",
        "lickin": "lick",
        "lov": "love",
        "liv": "live",
        "maama": "mama",
        "livin": "live",
        "lightnin": "lightning",
        "listenin": "listen",
        "lookin": "look",
        "losin": "loose",
        "lyin": "lie",
        "marchin": "march",
        "needin": "need",
        "nothin": "nothing",
        "packin": "pack",
        "payin": "pay",
        "passin": "pass",
        "pimpin": "pimp",
        "poppin": "pop",
        "pourin": "pour",
        "prayin": "pray",
        "preachin": "preach",
        "pretendin": "pretend",
        "puffin": "puff",
        "pullin": "pull",
        "pummpin": "pump",
        "puttin": "put",
        "racin": "race",
        "rainin": "rain",
        "rappin": "rap",
        "rhymin": "rhyme",
        "ridin": "ride",
        "risin": "rise",
        "robbin": "rob",
        "rushin": "rush",
        "schemin": "scheme",
        "screamin": "scream",
        "searchin": "search",
        "settin": "set",
        "shinin": "shine",
        "shootin": "shoot",
        "sippin": "sip",
        "slammin": "slam",
        "slippin": "slip",
        "sniffin": "sniff",
        "speedin": "speed",
        "spillin": "spill",
        "suckin": "suck",
        "swingin": "swing",
        "taking": "take",
        "tickin": "tick",
        "trippin": "trip",
        "tumblin": "tumble",
        "wantin": "want",
        "whippin": "whip",
        "wishin": "wish",
        "wonderin": "wonder",
        "workin": "work",
        "worryin": "worry",
        "writin": "write",
        "yellin": "yell",
        "seein": "see",
        "sellin": "sell",
        "plannin": "plan",
        "timin": "time",
        "totin": "tot",
        "driftin": "drift",
        "disappears": "disappear",
        "thuggin": "thug",
        "touchin": "touch",
        "pushin": "push",
        "pumpin": "pump",
        "plottin": "plot",
        "aight": "alright",
        "lil": "little",
        "missin": "miss",
        "mixin": "mix",
        "mornin": "morning",
        "movin": "move",
        "walkin": "walk",
        "runnin": "run",
        "talkin": "talk",
        "tryin": "try",
        "slangin": "slang",
        "dyin": "die",
        "beggin": "beg",
        "killa": "killer",
        "killas": "killer",
        "killin": "kill",
        "drivin": "drive",
        "makin": "make",
        "hittin": "hit",
        "sittin": "sit",
        "chillin": "chill",
        "smokin": "smoke",
        "playin": "play",
        "praying": "pray",
        "ramblin": "ramble",
        "sailin": "sail",
        "savin": "sav",
        "shootin ": "shoot",
        "shoppin": "shop",
        "sinkin": "sink",
        "skippin": "skip",
        "slidin": "slide",
        "sleepin": "sleep",
        "slowin": "slow",
        "smilin": "smile",
        "sneakin": "sneak",
        "soakin": "soak",
        "somethin": "something",
        "lovin": "love",
        "speakin": "speak",
        "spendin": "spend",
        "spreadin": "spread",
        "stackin": "stack",
        "standin": "stand",
        "starin": "stare",
        "stayin": "stay",
        "swimmin": "swim",
        "stealin": "steal",
        "sweatin": "sweat",
        "steppin": "step",
        "stickin": "stick",
        "stinkin": "stink",
        "stompin": "stomp",
        "stoppin": "stop",
        "strugglin": "struggle",
        "stumblin": "stumble",
        "stuntin": "stunt",
        "stylin": "style",
        "suckas": "sucker",
        "switchin": "switch",
        "tellin": "tell",
        "tryna": "trying to",
        "thinkin": "think",
        "travlin": "travel",
        "tricklin": "trickle",
        "feelin": "feel",
        "winnin": "win",
        "sayin": "say",
        "wreckin": "wreck",
        "niggas": "nigga",
        "niggaz": "nigga",
        "boyz": "boy",
        "caldonia": "caledonia",
        "muzik": "music",
        "nuthin": "nothing",
        "weighin": "weighing",
        "wetin": "wet",
        "oer": "over",
        "yappin": "yapping",
        "yougin": "young",
    }

    _GERUND_RE = re.compile(r"^(.{3,}?)(?:ing)$")

    _VOCALIZATIONS: Final[frozenset[str]] = frozenset(
        {
            "ahh",
            "ahhh",
            "aaaah",
            "aaaaah",
            "aaow",
            "aaah",
            "aah",
            "ohh",
            "ohhh",
            "ohhhh",
            "uhh",
            "mmh",
            "mmm",
            "mmmm",
            "hmm",
            "hmmm",
            "ooh",
            "oooh",
            "ooooh",
            "ayy",
            "ayyyy",
            "yeah",
            "yea",
            "yeh",
            "yuh",
            "huh",
            "hah",
            "haha",
            "hahah",
            "hahahaha",
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
        model: str = "en_core_web_sm",
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

        print("Converting lyrics to lowercase...")
        out[self.lyrics_column] = out[self.lyrics_column].astype(str).str.lower()

        print("Expanding contractions in lyrics...")
        expanded = out[self.lyrics_column].astype(str).map(self._expand_contractions)

        print("Stripping apostrophes...")
        expanded = expanded.map(self._strip_apostrophes)

        print(
            f"Lemmatizing lyrics in parallel (n_process={n_process}, batch_size={batch_size})..."
        )

        # Process texts in batches using nlp.pipe for performance
        docs = self.nlp.pipe(expanded.astype(str))

        # Use a list comprehension for a concise and readable way to process docs
        processed_lyrics = [self._process_doc(doc) for doc in docs]

        # Unpack the list of tuples into three new columns
        (
            out["lemmatized_pos_lyrics"],
            out["topic_lyrics"],
            out["sentiment_lyrics"],
        ) = zip(*processed_lyrics)

        print("Removing stopwords from topic and sentiment lyrics...")
        out["topic_lyrics"] = out["topic_lyrics"].map(self._remove_stopwords)
        out["sentiment_lyrics"] = out["sentiment_lyrics"].map(self._remove_stopwords)
        out["expression_lyrics"] = out[self.lyrics_column]

        return out

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

    def _strip_apostrophes(self, text: str) -> str:
        """Remove apostrophes and trailing 's' from possessives (e.g., 'Tobi's' -> 'Tobi')."""
        if not text or not isinstance(text, str):
            return text
        return re.sub(r"'s\b", "", text).replace("'", "")

    def _process_doc(self, doc: spacy.tokens.Doc) -> tuple[str, str, str]:
        """Process a single spaCy Doc to lemmatize and extract POS-specific tokens."""
        lemmas_with_pos: list[str] = []
        topic_lemmas: list[str] = []
        sentiment_lemmas: list[str] = []

        for token in doc:
            # Apply custom lexicon first to handle slang before lemmatization
            token_text = self._apply_domain_lexicon(token.text)

            # Use spaCy's lemma if the custom lexicon didn't change the token,
            # otherwise, re-lemmatize the normalized token.
            if token_text == token.text:
                lemma = token.lemma_
            else:
                # Re-process the single normalized token to get its lemma
                # This is less efficient but necessary for custom lexicon items.
                # nlp.pipe has already lowercased the text.
                lemma = self.nlp(token_text)[0].lemma_

            # Fallback for gerunds not covered by spaCy or lexicon
            lemma = self._normalize_gerund(lemma)

            if (
                lemma not in self._VOCALIZATIONS
                and not token.is_punct
                and not token.is_space
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

    def _apply_domain_lexicon(self, token: str) -> str:
        return self._DOMAIN_LEXICON.get(token, token)

    def _normalize_gerund(self, token: str) -> str:
        """Strip -ing suffix as a fallback for uncovered gerunds/progressives."""
        m = self._GERUND_RE.match(token)
        if m:
            stem = m.group(1)
            if len(stem) >= 3 and stem[-1] == stem[-2]:
                stem = stem[:-1]
            return stem
        return token
