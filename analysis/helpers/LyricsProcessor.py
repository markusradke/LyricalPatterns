from __future__ import annotations

import re

from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag
from typing import Final

from pandas import DataFrame
from helpers.StopwordFilter import StopwordFilter


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
        "bitchin": "bitch",
        "bitin": "bite",
        "blastin": "blast",
        "bleedin": "bleed",
        "bleeds": "bleed",
        "blessin": "bless",
        "blowin": "blow",
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
        "imma": "i am going to",
        "gonna": "going to",
        "wanna": "want to",
        "gotta": "got to",
        "i'ma": "i am going to",
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
        "layin": "lay",
        "leanin": "lean",
        "leavin": "leave",
        "lettin": "let",
        "lickin": "lick",
        "lightnin": "lightning",
        "listenin": "listen",
        "lookin": "look",
        "losin": "loose",
        "lyin": "lie",
        "marchin": "march",
        "needin": "need",
        "nothin": "nothing",
        "packin": "pack",
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
        "livin": "live",
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
        "thinkin": "think",
        "feelin": "feel",
        "winnin": "win",
        "sayin": "say",
        "wreckin": "wreck",
    }

    _GERUND_RE = re.compile(r"^(.{3,}?)(?:ing)$")

    _FILLER_TOKENS: Final[frozenset[str]] = frozenset(
        {
            "ah",
            "ahh",
            "ahhh",
            "aaaah",
            "aaah",
            "aah",
            "oh",
            "ohh",
            "ohhh",
            "ohhhh",
            "uh",
            "uhh",
            "mm",
            "mmh",
            "mmm",
            "mmmm",
            "hm",
            "hmm",
            "hmmm",
            "oo",
            "ooh",
            "oooh",
            "ooooh",
            "ay",
            "ayy",
            "ayyyy",
            "yeah",
            "yea",
            "yeh",
            "yo",
            "yuh",
            "huh",
            "hah",
            "haha",
            "hahah",
            "hahahaha",
            "hmm",
            "aw",
            "aww",
        }
    )

    _NOUN_TAGS: Final[frozenset[str]] = frozenset({"NN", "NNS", "NNP", "NNPS"})
    _SENTIMENT_TAGS: Final[frozenset[str]] = frozenset(
        {"JJ", "JJR", "JJS", "RB", "RBR", "RBS"}
    )

    def __init__(self, corpus: DataFrame, lyrics_column: str = "lyrics") -> None:
        self.corpus = corpus
        self.lyrics_column = lyrics_column
        self.stopword_filter = StopwordFilter()

    def process(self) -> DataFrame:
        """Process lyrics to create topic and sentiment versions.

        Returns:
            DataFrame with columns: full_lyrics, expressions_lyrics, topic_lyrics, sentiment_lyrics
        """
        out = self.corpus.copy()
        out["expressions_lyrics"] = out[self.lyrics_column]

        print("Expanding contractions in lyrics...")
        expanded = out[self.lyrics_column].astype(str).map(self._expand_contractions)

        print("Stripping apostrophes...")
        expanded = expanded.map(self._strip_apostrophes)

        print(
            "Lemmatizing lyrics and extracting POS tags for topic modeling and sentiment modeling..."
        )
        lemmatized_with_pos = expanded.map(self._lemmatize_text)

        print(
            "Extracting nouns for topic modeling and adjectives/adverbs for sentiment modeling..."
        )
        out["topic_lyrics"] = lemmatized_with_pos.map(self._extract_nouns)
        out["sentiment_lyrics"] = lemmatized_with_pos.map(self._extract_sentiment_words)

        print("Removing stopwords from topic and sentiment lyrics...")
        out["topic_lyrics"] = out["topic_lyrics"].map(self._remove_stopwords)
        out["sentiment_lyrics"] = out["sentiment_lyrics"].map(self._remove_stopwords)

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

    def _lemmatize_text(self, text: str) -> str:
        """Lemmatize text and return tokens with POS tags formatted as 'lemma/POS'."""
        lemmatizer = WordNetLemmatizer()
        lines = text.split("\n")
        lemmatized_lines: list[str] = []

        for line in lines:
            tokens = self._tokenize_line(line)
            tagged = pos_tag(tokens)
            lemmas: list[str] = []

            for tok, pos in tagged:
                tok_norm = self._apply_domain_lexicon(tok)
                tok_norm = self._normalize_gerund(tok_norm)
                wn_pos = self._wordnet_pos(pos)

                if wn_pos is None:
                    lemma = lemmatizer.lemmatize(tok_norm)
                else:
                    lemma = lemmatizer.lemmatize(tok_norm, pos=wn_pos)

                if (
                    lemma not in self._FILLER_TOKENS
                    and len(lemma) > 1
                    and not any(c.isdigit() for c in lemma)
                ):
                    lemmas.append(f"{lemma}/{pos}")

            lemmatized_lines.append(self._join_tokens(lemmas))

        return "\n".join(lemmatized_lines)

    def _extract_nouns(self, text: str) -> str:
        """Extract only nouns from lemmatized text (for topic modeling)."""
        return self._extract_pos_tokens(text, self._NOUN_TAGS)

    def _extract_sentiment_words(self, text: str) -> str:
        """Extract adjectives and adverbs from lemmatized text (for sentiment modeling)."""
        return self._extract_pos_tokens(text, self._SENTIMENT_TAGS)

    def _remove_stopwords(self, text: str) -> str:
        """Remove stopwords from extracted text."""
        if not text or not isinstance(text, str):
            return ""

        tokens = text.split()
        filtered_tokens = [
            token for token in tokens if not self.stopword_filter.is_stopword(token)
        ]
        return " ".join(filtered_tokens)

    @staticmethod
    def _extract_pos_tokens(text: str, target_tags: frozenset[str]) -> str:
        """Extract tokens with specific POS tags from lemmatized text."""
        if not text or not isinstance(text, str):
            return ""

        tokens = []
        for token_pair in text.split():
            if "/" in token_pair:
                token, pos = token_pair.rsplit("/", 1)
                if pos in target_tags:
                    tokens.append(token)

        return " ".join(tokens)

    def _tokenize_line(self, line: str) -> list[str]:
        """Tokenize a line using the same pattern as the ngram feature extractors."""
        return re.compile(r"\b[\w']+\b").findall(line.lower())

    def _apply_domain_lexicon(self, token: str) -> str:
        return self._DOMAIN_LEXICON.get(token, token)

    @staticmethod
    def _wordnet_pos(treebank_pos: str) -> str | None:
        if not treebank_pos:
            return None

        tag = treebank_pos[0].upper()
        if tag == "J":
            return wordnet.ADJ
        if tag == "V":
            return wordnet.VERB
        if tag == "N":
            return wordnet.NOUN
        if tag == "R":
            return wordnet.ADV
        return None

    def _normalize_gerund(self, token: str) -> str:
        """Strip -ing suffix as a fallback for uncovered gerunds/progressives."""
        m = self._GERUND_RE.match(token)
        if m:
            stem = m.group(1)
            if len(stem) >= 3 and stem[-1] == stem[-2]:
                stem = stem[:-1]
            return stem
        return token

    @staticmethod
    def _join_tokens(tokens: list[str]) -> str:
        """Join tokens back into a readable string."""
        if not tokens:
            return ""

        no_space_before = {".", ",", ":", ";", "!", "?", ")", "]", "}", "'"}
        no_space_after = {"(", "[", "{"}

        out: list[str] = []
        for tok in tokens:
            if not out:
                out.append(tok)
                continue

            tok_text = tok.split("/")[0] if "/" in tok else tok

            if tok_text in no_space_before:
                out[-1] = f"{out[-1]}{tok_text}"
            elif (
                out[-1].split("/")[0] if "/" in out[-1] else out[-1]
            ) in no_space_after:
                out[-1] = f"{out[-1]}{tok}"
            else:
                out.append(f" {tok}")

        return "".join(out)
