"""charlm: a small character-level GPT library for experimenting with the LLM training pipeline:
pretraining -> supervised fine-tuning (SFT) -> preference tuning (DPO) or reinforcement learning (GRPO)."""
from .chat import chat, encode_chat_example, format_prompt, sample_replies
from .checkpoint import load_checkpoint, save_checkpoint
from .config import load_config
from .datasets import prepare
from .dpo import DPOConfig, dpo
from .evaluation import evaluate_tasks
from .grpo import GRPOConfig, grpo
from .model import CharTransformerLanguageModel, ModelConfig
from .pretrain import PretrainConfig, pretrain
from .reasoning import answer_of, format_response, split_reply
from .sft import SFTConfig, sft
from .tasks import (ALL_TASKS, CHECKABLE_TASKS, MATH_TASKS, TASKS, VERIFIABLE_TASKS, Example, TaskSuite,
                    score)
from .tokenizer import REASONING_TOKENS, CharTokenizer

__all__ = [
    'CharTokenizer', 'REASONING_TOKENS', 'CharTransformerLanguageModel', 'ModelConfig',
    'save_checkpoint', 'load_checkpoint', 'load_config', 'prepare',
    'PretrainConfig', 'pretrain', 'SFTConfig', 'sft', 'DPOConfig', 'dpo', 'GRPOConfig', 'grpo',
    'chat', 'sample_replies', 'format_prompt', 'encode_chat_example',
    'format_response', 'split_reply', 'answer_of',
    'TaskSuite', 'Example', 'score', 'ALL_TASKS', 'VERIFIABLE_TASKS', 'CHECKABLE_TASKS', 'MATH_TASKS', 'TASKS',
    'evaluate_tasks',
]
