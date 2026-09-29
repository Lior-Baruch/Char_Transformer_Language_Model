import math
import os

import pytest
import torch
from torch.nn import functional as F

from charlm import (CharTokenizer, CharTransformerLanguageModel, DPOConfig, GRPOConfig, ModelConfig, PretrainConfig,
                    SFTConfig, TaskSuite, dpo, encode_chat_example, grpo, load_checkpoint, load_config, pretrain,
                    save_checkpoint, score, sft)
from charlm.cli import main as cli_main
from charlm.dpo import dpo_loss
from charlm.grpo import group_advantages, grpo_loss
from charlm.model import CausalSelfAttention

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS = os.path.join(ROOT, 'data', 'input.txt')


def tiny_model(vocab_size=100, block_size=32, dropout=0.0):
    torch.manual_seed(0)
    return CharTransformerLanguageModel(ModelConfig(vocab_size=vocab_size, block_size=block_size, n_embd=32,
                                                    n_head=4, n_layer=2, dropout=dropout))


def test_tokenizer_round_trip_and_special_tokens():
    tok = CharTokenizer()
    ids = tok.encode("<|user|>hi<|assistant|>")
    assert ids == [tok.user_id, tok.stoi['h'], tok.stoi['i'], tok.assistant_id]
    assert tok.decode(ids) == "<|user|>hi<|assistant|>"
    assert tok.decode(ids, skip_special=True) == "hi"
    # without allow_special the marker is just characters
    assert len(tok.encode("<|end|>", allow_special=False)) == len("<|end|>")
    with pytest.raises(ValueError):
        tok.encode("café")
    assert CharTokenizer.from_dict(tok.to_dict()).itos == tok.itos


def test_model_is_causal():
    model = tiny_model().eval()
    x = torch.randint(0, 100, (1, 16))
    x2 = x.clone()
    x2[0, 10] = (x2[0, 10] + 1) % 100  # change a token in the middle
    logits, _ = model(x)
    logits2, _ = model(x2)
    assert torch.allclose(logits[0, :10], logits2[0, :10], atol=1e-5)  # earlier positions can't see it
    assert not torch.allclose(logits[0, 10:], logits2[0, 10:])


def test_attention_is_scaled_by_head_size():
    config = ModelConfig(vocab_size=10, block_size=8, n_embd=16, n_head=4, dropout=0.0)
    attn = CausalSelfAttention(config).eval()
    x = torch.randn(2, 8, 16)
    q, k, v = attn.qkv(x).split(16, dim=2)
    q, k, v = (t.view(2, 8, 4, 4).transpose(1, 2) for t in (q, k, v))
    wei = q @ k.transpose(-2, -1) * 4 ** -0.5  # head_size = 16 / 4
    wei = wei.masked_fill(torch.tril(torch.ones(8, 8)) == 0, float('-inf')).softmax(-1)
    expected = attn.proj((wei @ v).transpose(1, 2).reshape(2, 8, 16))
    assert torch.allclose(attn(x), expected, atol=1e-5)


def test_generate_stops_at_stop_token():
    model = tiny_model()
    with torch.no_grad():  # make token 7 overwhelmingly likely
        model.lm_head.weight.zero_()
        model.lm_head.bias.zero_()
        model.lm_head.bias[7] = 100.0
    out = model.generate(torch.zeros((3, 4), dtype=torch.long), max_new_tokens=10, stop_token=7)
    assert out.shape == (3, 5) and (out[:, -1] == 7).all()
    out = model.generate(torch.zeros((1, 4), dtype=torch.long), max_new_tokens=5, temperature=0.0)
    assert out.shape == (1, 9)


def test_chat_example_only_trains_on_the_response():
    tok = CharTokenizer()
    inputs, targets = encode_chat_example(tok, "hi", "yo")
    # full sequence: <|user|> h i <|assistant|> y o <|end|>
    assert inputs == [tok.user_id, tok.stoi['h'], tok.stoi['i'], tok.assistant_id, tok.stoi['y'], tok.stoi['o']]
    assert targets == [-100, -100, -100, tok.stoi['y'], tok.stoi['o'], tok.end_id]


def test_checkpoint_round_trip(tmp_path):
    model, tok = tiny_model(dropout=0.2), CharTokenizer()
    path = str(tmp_path / 'm.pt')
    save_checkpoint(path, model, tok, {'stage': 'test'})
    loaded, loaded_tok, meta = load_checkpoint(path, dropout=0.0)
    x = torch.randint(0, 100, (2, 8))
    assert torch.allclose(model.eval()(x)[0], loaded.eval()(x)[0])
    assert loaded.config.dropout == 0.0 and meta == {'stage': 'test'} and loaded_tok.itos == tok.itos


def test_config_file_and_overrides(tmp_path):
    path = tmp_path / 'c.json'
    path.write_text('{"max_iters": 7, "model": {"n_layer": 3}}')
    cfg = load_config(PretrainConfig, str(path), ['model.n_embd=48', 'data_path=foo.txt'])
    assert (cfg.max_iters, cfg.model.n_layer, cfg.model.n_embd, cfg.data_path) == (7, 3, 48, 'foo.txt')
    with pytest.raises(ValueError, match='unknown'):
        load_config(PretrainConfig, None, ['max_iter=5'])


def test_dpo_loss():
    zeros = torch.zeros(4)
    loss, stats = dpo_loss(zeros, zeros, zeros, zeros, beta=0.1)
    assert math.isclose(loss.item(), math.log(2), rel_tol=1e-6)  # policy == reference
    better, _ = dpo_loss(zeros + 1, zeros - 1, zeros, zeros, beta=0.1)
    assert better < loss
    assert stats['reward_acc'] == 0.0


def test_group_advantages_and_grpo_loss():
    adv = group_advantages(torch.tensor([[1.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]]))
    assert torch.allclose(adv[0], torch.tensor([1.0, -1.0, -1.0, 1.0]), atol=1e-3)
    assert torch.allclose(adv[1], torch.zeros(4))  # no signal when every reply gets the same reward

    logps = torch.randn(2, 5)
    mask = torch.tensor([[1.0, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
    loss, kl = grpo_loss(logps, logps, logps, torch.tensor([2.0, -1.0]), mask, clip_eps=0.2, kl_coef=0.1)
    assert math.isclose(kl.item(), 0.0, abs_tol=1e-7)  # policy == reference
    assert math.isclose(loss.item(), (-2.0 + 1.0) / 2, rel_tol=1e-6)  # ratio 1: loss = -mean advantage


def test_tasks_are_correct_and_split():
    suite = TaskSuite(open(CORPUS).read())
    assert not set(suite.pools['words']['train']) & set(suite.pools['words']['eval'])
    for example in suite.eval_set(5):
        prompt, answer = example.prompt, example.answer
        if example.task == 'reverse':
            assert answer == prompt.split(': ')[1][::-1]
        elif example.task == 'add':
            a, b = prompt[len('What is '):-1].split(' + ')
            assert int(answer) == int(a) + int(b)
        assert score(example, f" {answer} ") == 1.0 and score(example, answer + 'x') == 0.0
    assert all(e.task == 'speak' for e in suite.sample(3, ['speak']))


def test_full_pipeline(tmp_path):
    """ pretrain -> SFT -> DPO and GRPO with tiny settings, then the CLI eval """
    corpus = tmp_path / 'corpus.txt'
    corpus.write_text(open(CORPUS).read()[:200_000])
    common = [f'corpus_path={corpus}', 'device=cpu', 'eval_per_task=2']
    base, sft_path = str(tmp_path / 'base.pt'), str(tmp_path / 'sft.pt')

    pretrain(load_config(PretrainConfig, None, [
        f'data_path={corpus}', f'out_path={base}', 'model.n_embd=32', 'model.n_head=2', 'model.n_layer=1',
        'model.block_size=64', 'batch_size=8', 'max_iters=6', 'eval_interval=3', 'eval_iters=2', 'device=cpu',
        'sample_tokens=10']))
    sft(load_config(SFTConfig, None, common + [
        f'init_from={base}', f'out_path={sft_path}', 'n_train=64', 'n_val=16', 'batch_size=8', 'max_iters=4',
        'eval_interval=2']))
    dpo(load_config(DPOConfig, None, common + [
        f'init_from={sft_path}', f'out_path={tmp_path / "dpo.pt"}', 'n_pairs=8', 'batch_size=4', 'max_iters=2',
        'eval_interval=1']))
    grpo(load_config(GRPOConfig, None, common + [
        f'init_from={sft_path}', f'out_path={tmp_path / "grpo.pt"}', 'batch_size=2', 'group_size=3',
        'max_new_tokens=8', 'max_iters=2', 'eval_interval=1']))

    for stage in ('base', 'sft', 'dpo', 'grpo'):
        path = tmp_path / f'{stage}.pt'
        assert path.exists() and (tmp_path / f'{stage}.metrics.jsonl').exists()
        assert load_checkpoint(str(path))[2]['stage'] == ('pretrain' if stage == 'base' else stage)
    assert (tmp_path / 'dpo.pairs.jsonl').exists()
    cli_main(['eval', '--model', sft_path, str(tmp_path / 'grpo.pt'), '--corpus', str(corpus), '--n-per-task', '2'])
    cli_main(['chat', '--model', sft_path, 'Reverse the word: love'])
