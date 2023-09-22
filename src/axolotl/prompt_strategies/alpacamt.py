"""Module containing the SimpleAlpacaMTPromptTokenizingStrategy class"""

from axolotl.prompt_tokenizers import AlpacaMTPromptTokenizingStrategy
from axolotl.prompters import PromptStyle, AlpacaMTPrompter, SHAREGPT_ASSERTION_FAILED_ROLE
from typing import Generator, List, Optional, Tuple, Union
import dataclasses


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

def load_therapist(tokenizer, cfg):
    return SimpleAlpacaMTPromptTokenizingStrategy(
        AlpacaMTPrompter(
            PromptStyle.CHAT.value, system_prompt='''You are a responsive and skilled therapist taking care of a patient who is looking for guidance and advice on managing their emotions and treating other mental health issues through text based therapy. Attentively listen to the patient and answer the patient's questions in an empathetic and non-judgemental tone, and do not judge the patient for any issues they are facing. Offer acceptance, support, and care for the patient, regardless of their circumstances or struggles.  Make them comfortable and ask open ended questions in an empathetic manner that encourages self reflection. Also try to avoid giving false or misleading information, and caveat when you aren't entirely sure about the right answer. Remember to always respond to the patient caringly and be kind and understanding towards them.'''
        ),
        tokenizer,
        cfg.train_on_inputs,
        cfg.sequence_len,
    )

class TherapistMTPrompter(AlpacaMTPrompter):
    def __init__(self, prompt_style=None, system_prompt: Optional[str] = None):
        self.system_prompt = '''You are a responsive and skilled therapist taking care of a patient who is looking for guidance and advice on managing their emotions and treating other mental health issues through text based therapy. Attentively listen to the patient and answer the patient's questions in an empathetic and non-judgemental tone, and do not judge the patient for any issues they are facing. Offer acceptance, support, and care for the patient, regardless of their circumstances or struggles.  Make them comfortable and ask open ended questions in an empathetic manner that encourages self reflection. Also try to avoid giving false or misleading information, and caveat when you aren't entirely sure about the right answer. Remember to always respond to the patient caringly and be kind and understanding towards them.'''
        super().__init__(prompt_style=prompt_style, system_prompt=self.system_prompt)

    def build_prompt(self, source) -> Generator[str, None, None]:
        if len(source) < 2:
            # If there isn't a back and forth conversation, ignore it
            # also happens on the data splitting leaving empty conversations
            raise IndexError(
                f"A conversation entry has less than 2 messages :\n{source}"
            )

        conv = self._conversation.copy()

        # Add the conversation system prompt if provided, otherwise use the default one
        if source[0]["from"] == "system":
            conv.system = source[0]["value"]
            source.pop(0)

        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

        try:
            # Apply prompt templates
            if (
                source[0]["from"] not in roles
                or roles[source[0]["from"]] != conv.roles[0]
            ):
                # Skip the first one if it is not from human
                conv.system += f'\nTherapist\'s first response:\n{source[0]["value"]}'
                source = source[1:]
        except IndexError as err:
            # sometimes there is a bing or system chat
            raise err

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], SHAREGPT_ASSERTION_FAILED_ROLE
            conv.append_message(role, sentence["value"])

        for part in conv.get_prompt():
            yield part



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