"""Module containing the SimpleAlpacaMTPromptTokenizingStrategy class"""

from axolotl.prompt_tokenizers import AlpacaMTPromptTokenizingStrategy
from axolotl.prompters import PromptStyle, AlpacaMTPrompter


def load(tokenizer, cfg):
    return SimpleAlpacaMTPromptTokenizingStrategy(
        AlpacaMTPrompter(PromptStyle.CHAT.value),
        tokenizer,
        cfg.train_on_inputs,
        cfg.sequence_len,
    )


def load_role(tokenizer, cfg):
    return SimpleRoleAlpacaMTPromptTokenizingStrategy(
        AlpacaMTPrompter(PromptStyle.CHAT.value),
        tokenizer,
        cfg.train_on_inputs,
        cfg.sequence_len,
    )


class SimpleAlpacaMTPromptTokenizingStrategy(AlpacaMTPromptTokenizingStrategy):
    """
    basic sharegpt strategy to grab conversations from the sample row
    """

    def get_conversation_thread(self, prompt):
        return prompt["conversations"]


class SimpleRoleAlpacaMTPromptTokenizingStrategy(AlpacaMTPromptTokenizingStrategy):
    """
    basic sharegpt strategy to grab conversations from the sample row, but uses role instead of from
    """

    def get_conversation_thread(self, prompt):
        conversations = prompt["conversations"]
        # remap role: prompter/assistant, text: ... => from: human/gpt, value: ...
        turns = [{"from": t["role"], "value": t["value"]} for t in conversations]
        return turns

