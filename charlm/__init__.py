"""charlm: a small character-level GPT library for experimenting with the LLM training pipeline:
pretraining -> supervised fine-tuning (SFT) -> preference tuning (DPO) or reinforcement learning (GRPO)."""
from .chat import chat, encode_chat_example, format_prompt, sample_replies
from .checkpoint import load_checkpoint, save_checkpoint
from .config import load_config
from .dpo import DPOConfig, dpo
from .evaluation import evaluate_tasks
from .grpo import GRPOConfig, grpo
from .model import CharTransformerLanguageModel, ModelConfig
from .pretrain import PretrainConfig, pretrain
from .sft import SFTConfig, sft
from .tasks import ALL_TASKS, VERIFIABLE_TASKS, Example, TaskSuite, score
from .tokenizer import CharTokenizer

__all__ = [
    'CharTokenizer', 'CharTransformerLanguageModel', 'ModelConfig',
    'save_checkpoint', 'load_checkpoint', 'load_config',
    'PretrainConfig', 'pretrain', 'SFTConfig', 'sft', 'DPOConfig', 'dpo', 'GRPOConfig', 'grpo',
    'chat', 'sample_replies', 'format_prompt', 'encode_chat_example',
    'TaskSuite', 'Example', 'score', 'ALL_TASKS', 'VERIFIABLE_TASKS', 'evaluate_tasks',
]
