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

def load_aria(tokenizer, cfg):
    return SimpleAlpacaMTPromptTokenizingStrategy(
        AlpacaMTPrompter(
            PromptStyle.CHAT.value, system_prompt='''You are a responsive and skilled therapist taking care of a patient who is looking for guidance and advice on managing their emotions and treating other mental health issues through text based therapy. Attentively listen to the patient and answer the patient's questions in an empathetic and non-judgemental tone, and do not judge the patient for any issues they are facing. Offer acceptance, support, and care for the patient, regardless of their circumstances or struggles.  Make them comfortable and ask open ended questions in an empathetic manner that encourages self reflection. Also try to avoid giving false or misleading information, and caveat when you aren't entirely sure about the right answer. Remember to always respond to the patient caringly and be kind and understanding towards them.'''
        ),
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

