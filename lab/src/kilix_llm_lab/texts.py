"""Small original contrast sets for the 0.6B project.

These are enough to exercise difference of means. A larger corpus changes the
direction. Sad and happy lines are written so a lexicon scorer separates them.
"""

from __future__ import annotations

HAPPY = (
    "The morning was bright and cheerful. She felt glad and full of joy.",
    "Friends laugh in the warm sun. A smile makes the day happy.",
    "The garden was bright with joy. Everyone felt glad and cheerful.",
    "A warm laugh filled the room. He was happy and glad to be there.",
    "Children smile and laugh. The bright day felt cheerful and full of joy.",
    "She felt a glad warm smile. The news was happy and cheerful.",
    "Joy and laughter made the evening bright. He was glad and happy.",
    "A cheerful smile met a warm hello. The room was bright with joy.",
)

SAD = (
    "The house felt empty and lonely. Grief sat in the quiet rooms.",
    "She began to cry. Tears and sorrow made the night miserable.",
    "He felt sad and lonely. The empty chair brought grief.",
    "Miserable sorrow kept him awake. He would cry until the tears stopped.",
    "The lonely street was empty. Sad grief hung over the town.",
    "Tears marked her sorrow. She felt miserable, sad, and lonely.",
    "An empty room held only grief. The cry of the wind sounded sad.",
    "Sorrow and tears returned. The miserable day left him lonely.",
)

# Speech-act contrast for a refusal direction. These are the assistant's words,
# not a list of requests. A JSON file with the same two keys can replace them.
REFUSE = (
    "I must refuse that request.",
    "I can't help with that.",
    "No, I will not do that.",
    "I have to decline.",
    "That is not something I can do.",
    "I must say no.",
    "I won't assist with this.",
    "My answer is a refusal.",
)
COMPLY = (
    "Here is a direct answer.",
    "Yes, I can help with that.",
    "Sure, this is the reply.",
    "I will answer directly.",
    "Of course. Here it is.",
    "Yes. The answer follows.",
    "I can do that.",
    "Here is what you asked for.",
)

# One tomato sentence against everyday objects. The blank in the video is filled
# here so each line is a complete forward-pass example.
TOMATO = (
    "I put a tomato on the kitchen table.",
    "I set a tomato beside the plate.",
)
OTHER_OBJECTS = (
    "I put an apple on the kitchen table.",
    "I put a book on the kitchen table.",
    "I put a chair on the kitchen table.",
    "I put a cup on the kitchen table.",
    "I put a key on the kitchen table.",
    "I put a lamp on the kitchen table.",
    "I put a pencil on the kitchen table.",
    "I put a stone on the kitchen table.",
)

# Prompt, and a substring the unsteered answer is scored against.
FACTS = (
    ("The capital of Japan is", "tokyo"),
    ("Water freezes at 0 degrees", "celsius"),
    ("The planet closest to the sun is", "mercury"),
    ("Two plus two equals", "four"),
    ("The Pacific is an", "ocean"),
    ("A triangle has three", "sides"),
)

STORY_PROMPT = "Tell me a story."
NEUTRAL_PROMPT = "Tell me a story."
WHO_PROMPT = "Who are you?"

# The four Neuron Explorer prompts. The last one is the open write-whatever line.
EXPLORER_PROMPTS = (
    "Who are you?",
    "Write a snake game in Python.",
    "Describe the color orange.",
    "Write whatever you like.",
)
