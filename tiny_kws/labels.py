from typing import Final

TARGET_WORDS: Final[tuple[str, ...]] = (
    "down",
    "go",
    "left",
    "no",
    "off",
    "on",
    "right",
    "stop",
    "up",
    "yes",
)
UNKNOWN_WORDS: Final[tuple[str, ...]] = (
    "backward",
    "bed",
    "bird",
    "cat",
    "dog",
    "eight",
    "five",
    "follow",
    "forward",
    "four",
    "happy",
    "house",
    "learn",
    "marvin",
    "nine",
    "one",
    "seven",
    "sheila",
    "six",
    "three",
    "tree",
    "two",
    "visual",
    "wow",
    "zero",
)
CLASS_NAMES: Final[tuple[str, ...]] = (*TARGET_WORDS, "unknown", "silence")
UNKNOWN_CLASS_INDEX: Final = CLASS_NAMES.index("unknown")
SILENCE_CLASS_INDEX: Final = CLASS_NAMES.index("silence")
WORD_TO_CLASS_INDEX: Final[dict[str, int]] = {
    word: idx for idx, word in enumerate(TARGET_WORDS)
} | dict.fromkeys(UNKNOWN_WORDS, UNKNOWN_CLASS_INDEX)
