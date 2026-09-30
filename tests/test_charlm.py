import json
import math
import os

import pytest
import torch

from charlm import (MATH_TASKS, REASONING_TOKENS, VERIFIABLE_TASKS, CharTokenizer, CharTransformerLanguageModel,
                    DPOConfig, GRPOConfig, ModelConfig, PretrainConfig, SFTConfig, TaskSuite, answer_of, dpo,
                    encode_chat_example, format_response, grpo, load_checkpoint, load_config, pretrain,
                    sample_replies, save_checkpoint, score, sft, split_reply)
from charlm.cli import main as cli_main
from charlm.data import read_jsonl
from charlm.dpo import dpo_loss
from charlm.grpo import group_advantages, grpo_loss
from charlm.model import CausalSelfAttention
from charlm.reasoning import add_trace, div_trace, mul_trace, sub_trace
from charlm.tasks import eval_tasks
from charlm.tokenizer import END_THINK, THINK

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


@pytest.mark.parametrize('stage', ['pretrain', 'sft', 'dpo', 'grpo'])
def test_resumed_run_matches_uninterrupted_run(stage, tmp_path, monkeypatch):
    """ interrupt a run at step 5, resume it, and check it ends exactly like a run that was never interrupted """
    import importlib
    import json
    module = importlib.import_module(f'charlm.{stage}')  # the module, not the function of the same name
    corpus = tmp_path / 'corpus.txt'
    corpus.write_text(open(CORPUS).read()[:200_000])
    common = ['device=cpu', 'max_iters=8', 'eval_interval=3']
    tiny = ['model.n_embd=16', 'model.n_head=2', 'model.n_layer=1', 'model.block_size=64', 'batch_size=4',
            'eval_iters=2', 'sample_tokens=0', f'data_path={corpus}']
    finetune = [f'corpus_path={corpus}', 'eval_per_task=1', f'init_from={tmp_path / "base.pt"}']
    if stage != 'pretrain':
        pretrain(load_config(PretrainConfig, None, tiny + [f'out_path={tmp_path / "base.pt"}', 'max_iters=2']))
    options = {'pretrain': tiny + ['patience=0'],
               'sft': finetune + ['n_train=32', 'n_val=8', 'batch_size=4', 'patience=0'],
               'dpo': finetune + ['n_pairs=12', 'batch_size=4'],
               'grpo': finetune + ['batch_size=2', 'group_size=3', 'max_new_tokens=6']}[stage]
    config_cls = {'pretrain': PretrainConfig, 'sft': SFTConfig, 'dpo': DPOConfig, 'grpo': GRPOConfig}[stage]
    run = getattr(module, stage)

    def config(name, resume=False):
        return load_config(config_cls, None, common + options + [f'out_path={tmp_path / name}.pt',
                                                                  f'resume={json.dumps(resume)}'])

    full, _ = run(config('full'))

    real_step = module.optimizer_step

    def crash_at_step_5(model, optimizer, loss, it, *args, **kwargs):
        if it == 5:
            raise KeyboardInterrupt
        return real_step(model, optimizer, loss, it, *args, **kwargs)

    monkeypatch.setattr(module, 'optimizer_step', crash_at_step_5)
    with pytest.raises(KeyboardInterrupt):
        run(config('resumed'))
    assert (tmp_path / 'resumed.state.pt').exists()
    monkeypatch.setattr(module, 'optimizer_step', real_step)
    with pytest.raises(ValueError, match='different settings'):  # resuming with other settings would mix two runs
        run(load_config(config_cls, None, common + options + [f'out_path={tmp_path / "resumed.pt"}', 'resume=true',
                                                              'max_iters=9']))
    resumed, _ = run(config('resumed', resume=True))

    assert not (tmp_path / 'resumed.state.pt').exists()
    for a, b in zip(full.state_dict().values(), resumed.state_dict().values()):
        assert torch.equal(a, b)
    logs = [[{k: v for k, v in json.loads(line).items() if k != 'time'}
             for line in open(tmp_path / f'{name}.metrics.jsonl')] for name in ('full', 'resumed')]
    assert logs[0] == logs[1]
    assert [row['step'] for row in logs[0]] == [0, 3, 6, 8]  # 8 evaluates the final update


@pytest.mark.parametrize('stage', ['pretrain', 'sft'])
def test_final_update_is_evaluated_and_kept(stage, tmp_path):
    """ with a single training step, the returned model must include that step """
    import json
    corpus = tmp_path / 'corpus.txt'
    corpus.write_text(open(CORPUS).read()[:200_000])
    tiny = [f'data_path={corpus}', 'model.n_embd=16', 'model.n_head=2', 'model.n_layer=1', 'model.block_size=64',
            'batch_size=4', 'eval_iters=2', 'sample_tokens=0', 'device=cpu', 'patience=0']
    pretrain(load_config(PretrainConfig, None, tiny + [f'out_path={tmp_path / "base.pt"}', 'max_iters=2']))
    one_step = ['max_iters=1', 'eval_interval=1', 'learning_rate=0.1', 'warmup_iters=0',
                f'out_path={tmp_path / "out.pt"}']
    if stage == 'pretrain':
        config = load_checkpoint(str(tmp_path / 'base.pt'))[0].config
        torch.manual_seed(1337)  # pretrain seeds the same way before building its model, so this is its starting point
        before = CharTransformerLanguageModel(config)
        after, _ = pretrain(load_config(PretrainConfig, None, tiny + one_step))
    else:
        before = load_checkpoint(str(tmp_path / 'base.pt'))[0]
        after, _ = sft(load_config(SFTConfig, None, [f'corpus_path={corpus}', f'init_from={tmp_path / "base.pt"}',
                                                     'n_train=16', 'n_val=4', 'batch_size=4', 'eval_per_task=0',
                                                     'device=cpu', 'patience=0'] + one_step))
    assert [json.loads(line)['step'] for line in open(tmp_path / 'out.metrics.jsonl')] == [0, 1]
    assert not all(torch.equal(a, b) for a, b in zip(before.state_dict().values(), after.state_dict().values()))


def test_prompt_without_room_for_a_reply_is_rejected():
    model = tiny_model(block_size=16)
    with pytest.raises(ValueError, match='no room'):
        sample_replies(model, CharTokenizer(), ['x' * 20])


@pytest.mark.parametrize('stage', ['dpo', 'grpo'])
def test_reference_model_with_another_tokenizer_is_rejected(stage, tmp_path):
    tok = CharTokenizer()
    reordered = CharTokenizer(special_tokens=list(reversed(tok.special_tokens)))  # same size, different ids
    save_checkpoint(str(tmp_path / 'policy.pt'), tiny_model(), tok)
    save_checkpoint(str(tmp_path / 'ref.pt'), tiny_model(), reordered)
    config_cls, run = {'dpo': (DPOConfig, dpo), 'grpo': (GRPOConfig, grpo)}[stage]
    with pytest.raises(ValueError, match='tokenizer'):
        run(load_config(config_cls, None, [f'init_from={tmp_path / "policy.pt"}', f'ref_from={tmp_path / "ref.pt"}',
                                           f'out_path={tmp_path / "out.pt"}', f'corpus_path={CORPUS}', 'device=cpu']))


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
        'eval_interval=1', 'nll_coef=1.0']))
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


# --- reasoning, new tasks, precision, data ---------------------------------------------------------------------

def test_reasoning_tokens_are_appended_without_changing_ids():
    tok = CharTokenizer()
    assert 'chars' in tok.to_dict() and not tok.has_reasoning_tokens  # the original checkpoint format
    before = dict(tok.stoi)
    assert tok.add_special_tokens(REASONING_TOKENS) == 2 and tok.add_special_tokens(REASONING_TOKENS) == 0
    assert all(tok.stoi[t] == i for t, i in before.items()) and tok.vocab_size == len(before) + 2
    restored = CharTokenizer.from_dict(tok.to_dict())
    assert restored.itos == tok.itos and restored.has_reasoning_tokens
    reply = format_response('132', '7+5+0=12 A=2')
    ids = tok.encode(reply, allow_special=REASONING_TOKENS)
    assert ids[0] == tok.stoi[THINK] and tok.stoi[END_THINK] in ids and tok.decode(ids) == reply
    assert tok.decode(ids, skip_special=True) == '7+5+0=12 A=2132'
    # a user's message is never turned into special tokens
    assert len(tok.encode(THINK, allow_special=False)) == len(THINK)
    with pytest.raises(ValueError, match='reasoning tokens'):
        encode_chat_example(CharTokenizer(), 'What is 1 + 2?', format_response('3', 'x'))


def test_resize_vocab_keeps_the_old_predictions():
    model = tiny_model().eval()
    x = torch.randint(0, 100, (2, 10))
    before = model(x)[0]
    model.resize_vocab(102)
    after = model(x)[0]
    assert after.shape[-1] == 102 and torch.allclose(before, after[..., :100], atol=1e-6)
    assert model.config.vocab_size == 102 and model(torch.tensor([[101, 100]]))[0].shape == (1, 2, 102)


def test_kv_cache_generates_the_same_tokens():
    model = tiny_model(block_size=32)
    x = torch.randint(0, 100, (3, 5))
    cached = model.generate(x, 20, temperature=0.0)
    assert torch.equal(cached, model.generate(x, 20, temperature=0.0, use_cache=False))
    torch.manual_seed(3)
    sampled = model.generate(x, 20, temperature=1.0)
    torch.manual_seed(3)
    assert torch.equal(sampled, model.generate(x, 20, temperature=1.0, use_cache=False))
    assert model.generate(x, 40, temperature=0.0).shape == (3, 45)  # longer than block_size: a sliding window


def test_reasoning_traces_are_correct():
    import random
    rng = random.Random(0)
    for _ in range(300):
        a, b = rng.randint(0, 9999), rng.randint(0, 9999)
        trace, answer = add_trace(a, b)
        assert answer == str(a + b) and trace.endswith(f'=> {a + b}')
        a, b = max(a, b), min(a, b)
        assert sub_trace(a, b)[1] == str(a - b)
        d = rng.randint(1, 9)
        assert mul_trace(a, d)[1] == str(a * d)
        assert div_trace(a * d, d)[1] == str(a)
    assert add_trace(47, 85)[0] == '7+5+0=12 A=2, 4+8+1=13 A=132 => 132'
    assert sub_trace(82, 47)[0] == '2-7-0=5 b1 A=5, 8-4-1=3 b0 A=35 => 35'
    assert mul_trace(47, 6)[0] == '7*6=42+0=42 A=2, 4*6=24+4=28 A=282 => 282'
    assert div_trace(84, 6)[0] == '08/6=1 r2 A=1, 24/6=4 r0 A=14 => 14'
    assert split_reply(format_response('5', 'x')) == ('x', '5') and split_reply('<|think|>cut') == ('cut', '')
    assert split_reply('plain') == ('', 'plain') and answer_of('<|think|>x<|/think|> 7') == ' 7'


def test_new_tasks_are_correct_split_and_fit_the_example_context():
    suite = TaskSuite(open(CORPUS).read())
    tok = CharTokenizer()
    tok.add_special_tokens(REASONING_TOKENS)
    ops = {'sub': lambda a, b: a - b, 'mul': lambda a, b: a * b, 'div': lambda a, b: a // b}
    for split in ('train', 'eval'):
        examples = suite.sample(2000, MATH_TASKS, split, seed=5)
        for e in examples:
            if e.task in ops:
                a, b = map(int, e.prompt[len('What is '):-1].split(f" {dict(sub='-', mul='*', div='/')[e.task]} "))
                assert int(e.answer) == ops[e.task](a, b) and (e.task != 'div' or a % b == 0)
            assert e.reasoning.endswith(f'=> {e.answer}') and score(e, format_response(e.answer, e.reasoning))
            inputs, _ = encode_chat_example(tok, e.prompt, format_response(e.answer, e.reasoning))
            assert len(inputs) + 1 <= 128, e  # the example models have block_size 128
    train = {e.prompt for e in suite.sample(3000, ['sub', 'mul', 'div'], 'train', seed=1)}
    held_out = {e.prompt for e in suite.eval_set(300, ['sub', 'mul', 'div'])}
    assert not train & held_out
    # word problems are evaluated on a phrasing that is never trained on
    import re
    from charlm.tasks import WORD_TEMPLATES
    pattern = lambda t: re.compile(re.sub(r'\\{[a-z]\\}', '.+', re.escape(t)) + '$')
    held_out = [pattern(ts[-1]) for ts in WORD_TEMPLATES.values()]
    trained = [pattern(t) for ts in WORD_TEMPLATES.values() for t in ts[:-1]]
    assert all(any(p.match(e.prompt) for p in held_out) for e in suite.eval_set(100, ['word']))
    assert not any(p.match(e.prompt) for p in held_out for e in suite.sample(300, ['word']))
    assert all(any(p.match(e.prompt) for p in trained) for e in suite.sample(300, ['word']))
    assert eval_tasks(['reverse', 'sub', 'word']) == VERIFIABLE_TASKS + ('sub', 'word')


def test_precision_options_on_cpu(tmp_path):
    from charlm.training import resolve_precision, setup_precision
    assert resolve_precision('auto', 'cpu') == 'fp32'
    with pytest.raises(ValueError):
        resolve_precision('fp8', 'cpu')
    model = tiny_model()
    cfg = SFTConfig(precision='bf16')
    scaler = setup_precision(cfg, 'cpu', model)
    assert not scaler.is_enabled()
    logits, loss = model(torch.randint(0, 100, (2, 8)), torch.randint(0, 100, (2, 8)))
    assert logits.dtype == torch.bfloat16 and loss.dtype == torch.float32
    corpus = tmp_path / 'corpus.txt'
    corpus.write_text(open(CORPUS).read()[:100_000])
    for precision in ('bf16', 'fp16'):
        model, _ = pretrain(load_config(PretrainConfig, None, [
            f'data_path={corpus}', f'out_path={tmp_path / precision}.pt', 'model.n_embd=16', 'model.n_head=2',
            'model.n_layer=1', 'model.block_size=32', 'batch_size=4', 'max_iters=3', 'eval_interval=3',
            'eval_iters=1', 'sample_tokens=0', 'device=cpu', f'precision={precision}']))
        assert all(torch.isfinite(p).all() for p in model.parameters())


def test_sft_with_reasoning_then_dpo_and_grpo(tmp_path):
    corpus = tmp_path / 'corpus.txt'
    corpus.write_text(open(CORPUS).read()[:200_000])
    base, sft_path = str(tmp_path / 'base.pt'), str(tmp_path / 'sft.pt')
    pretrain(load_config(PretrainConfig, None, [
        f'data_path={corpus}', f'out_path={base}', 'model.n_embd=32', 'model.n_head=2', 'model.n_layer=1',
        'model.block_size=128', 'batch_size=4', 'max_iters=2', 'eval_iters=1', 'device=cpu', 'sample_tokens=0']))
    common = [f'corpus_path={corpus}', 'device=cpu', 'eval_per_task=2', 'max_new_tokens=100']
    model, tok = sft(load_config(SFTConfig, None, common + [
        f'init_from={base}', f'out_path={sft_path}', 'n_train=32', 'n_val=8', 'batch_size=4', 'max_iters=2',
        'eval_interval=1', 'reasoning=true', 'tasks=["reverse","sub","word"]']))
    assert tok.has_reasoning_tokens and model.config.vocab_size == tok.vocab_size == 102
    metrics = [json.loads(line) for line in open(tmp_path / 'sft.metrics.jsonl')]
    assert {'acc/sub', 'acc/word', 'acc/add'} <= set(metrics[-1])  # the original tasks are always evaluated
    assert load_checkpoint(sft_path)[2]['finished']
    dpo(load_config(DPOConfig, None, common + [
        f'init_from={sft_path}', f'out_path={tmp_path / "dpo.pt"}', 'n_pairs=6', 'batch_size=2', 'max_iters=1',
        'tasks=["sub"]']))
    assert all(THINK in r['chosen'] for r in read_jsonl(str(tmp_path / 'dpo.pairs.jsonl')))
    grpo(load_config(GRPOConfig, None, common + [
        f'init_from={sft_path}', f'out_path={tmp_path / "grpo.pt"}', 'batch_size=2', 'group_size=2', 'max_iters=1',
        'tasks=["sub"]']))
    assert {'reasoned', 'closed_think'} <= set(json.loads(open(tmp_path / 'grpo.metrics.jsonl').readlines()[-1]))
    cli_main(['eval', '--model', sft_path, '--corpus', str(corpus), '--n-per-task', '2', '--tasks', 'sub', 'word',
              '--show', '1', '--device', 'cpu'])
    cli_main(['chat', '--model', sft_path, 'What is 5 - 3?', '--device', 'cpu'])


def test_finished_stage_is_skipped_on_resume(tmp_path):
    corpus = tmp_path / 'corpus.txt'
    corpus.write_text(open(CORPUS).read()[:100_000])
    options = [f'data_path={corpus}', f'out_path={tmp_path / "base.pt"}', 'model.n_embd=16', 'model.n_head=2',
               'model.n_layer=1', 'model.block_size=32', 'batch_size=4', 'max_iters=2', 'eval_iters=1',
               'sample_tokens=0', 'device=cpu', 'resume=true']
    pretrain(load_config(PretrainConfig, None, options))
    mtime = os.path.getmtime(tmp_path / 'base.pt')
    model, _ = pretrain(load_config(PretrainConfig, None, options))
    assert os.path.getmtime(tmp_path / 'base.pt') == mtime and isinstance(model, CharTransformerLanguageModel)


def test_state_from_before_a_new_option_still_resumes(tmp_path):
    from charlm.training import load_state, make_optimizer, save_state, state_path
    model, cfg = tiny_model(), SFTConfig(out_path=str(tmp_path / 'm.pt'))
    optimizer = make_optimizer(model, cfg)
    save_state(cfg.out_path, model, optimizer, 4, cfg)
    state = torch.load(state_path(cfg.out_path), weights_only=True)
    del state['config']['reasoning'], state['precision']  # saved by an older charlm
    torch.save(state, state_path(cfg.out_path))
    assert load_state(cfg.out_path, model, optimizer, 'cpu', cfg)[0] == 4
    with pytest.raises(ValueError, match='reasoning'):
        load_state(cfg.out_path, model, optimizer, 'cpu', SFTConfig(out_path=cfg.out_path, reasoning=True))


def test_data_files_directories_and_fast_encoding(tmp_path):
    from charlm.data import encode_text_bytes, expand_paths, read_text_bytes, text_chars
    (tmp_path / 'b.txt').write_bytes(b'second file\r\nwith CRLF\r\n')
    (tmp_path / 'a.txt').write_text('first file\n' * 50)
    (tmp_path / 'notes.json').write_text('{}')
    assert expand_paths(str(tmp_path)) == [str(tmp_path / 'a.txt'), str(tmp_path / 'b.txt')]
    assert expand_paths([str(tmp_path / '*.txt'), str(tmp_path / 'a.txt')]) == expand_paths(str(tmp_path))
    with pytest.raises(FileNotFoundError):
        expand_paths(str(tmp_path / 'missing.txt'))
    tok = CharTokenizer()
    buf = read_text_bytes(str(tmp_path / 'b.txt'))
    assert bytes(buf) == b'second file\nwith CRLF\n' and text_chars(buf) == set('second filewthCRLF\n')
    ids = encode_text_bytes(tok, buf)
    assert ids.dtype == torch.uint8 and ids.tolist() == tok.encode(buf.decode(), allow_special=False)
    unicode = bytearray('café\n'.encode())
    with pytest.raises(ValueError, match='vocabulary'):
        encode_text_bytes(tok, unicode)
    assert encode_text_bytes(CharTokenizer.from_text('café'), unicode).tolist() == \
        CharTokenizer.from_text('café').encode('café\n')
    with pytest.raises(ValueError, match='vocabulary'):
        encode_text_bytes(tok, bytearray(b'tab\there'))
    # pretraining on a directory: each file's end is its validation part, capped by max_val_chars
    from charlm.pretrain import load_splits
    splits = load_splits(expand_paths(str(tmp_path)), tok, 0.5, 10)
    assert len(splits['val']) == 20 and len(splits['train']) == 550 + 22 - 20


def test_prepare_data_normalizes_and_skips_prepared_files(tmp_path):
    from charlm.datasets import normalize, prepare
    assert normalize('Café “yes” — ½\tcup…\x07') == 'Cafe "yes" - 1/2 cup...'
    book = tmp_path / 'book.txt'
    book.write_text('Title\n*** START OF THE PROJECT GUTENBERG EBOOK X ***\n\n\nTo be ’tis\n'
                    '*** END OF THE PROJECT GUTENBERG EBOOK X ***\nlicense\n', encoding='utf-8')
    out = str(tmp_path / 'shakespeare.txt')
    prepare('shakespeare', out, url=book.as_uri())
    assert open(out).read() == "To be 'tis\n"
    stories = tmp_path / 'stories.txt'
    stories.write_text('One.\n<|endoftext|>\nTwo.\n<|endoftext|>\nThree.\n')
    out = str(tmp_path / 'tinystories.txt')
    prepare('tinystories', out, url=stories.as_uri(), max_chars=10)
    assert open(out).read() == 'One.\n\n'
    mtime = os.path.getmtime(out)
    prepare('tinystories', out, url=stories.as_uri(), max_chars=10)  # already prepared: skipped
    assert os.path.getmtime(out) == mtime
    prepare('tinystories', out, url=stories.as_uri())  # other options: prepared again
    assert open(out).read() == 'One.\n\nTwo.\n\nThree.\n'
    joined = str(tmp_path / 'joined.txt')
    prepare('files', joined, files=[str(book), str(stories)])
    assert open(joined).read().startswith('Title\n') and '\n\nOne.\n' in open(joined).read()
    cli_main(['prepare-data', 'files', '--files', str(stories), '--out', str(tmp_path / 'cli.txt')])
    assert os.path.exists(tmp_path / 'cli.json')


# --- review follow-ups --------------------------------------------------------------------------------------

def _problem(e):
    """ the arithmetic problem (op, a, b) behind a math example """
    import re
    if e.task == 'word':
        a, op, b = re.match(r'(\d+)([-+*/])(\d+):', e.reasoning).groups()
    else:
        a, op, b = re.match(r'What is (\d+) ([-+*/]) (\d+)\?', e.prompt).groups()
    return op, int(a), int(b)


def test_held_out_problems_are_never_trained_in_any_task():
    suite = TaskSuite(open(CORPUS).read())
    trained = {_problem(e) for e in suite.sample(30000, MATH_TASKS, 'train', seed=1)}
    held_out = [_problem(e) for e in suite.eval_set(200, MATH_TASKS)]
    assert not [p for p in held_out if p in trained]
    with pytest.raises(ValueError, match='eval_fraction'):
        TaskSuite('some words here', eval_fraction=0)
    import random
    with pytest.raises(ValueError, match='split'):  # a split with no problems raises instead of looping forever
        TaskSuite(open(CORPUS).read(), eval_fraction=0.0001).make('word', random.Random(0), 'eval')


def test_finished_stage_with_other_settings_is_not_silently_reused(tmp_path):
    corpus = tmp_path / 'corpus.txt'
    corpus.write_text(open(CORPUS).read()[:100_000])
    options = [f'data_path={corpus}', f'out_path={tmp_path / "base.pt"}', 'model.n_embd=16', 'model.n_head=2',
               'model.n_layer=1', 'model.block_size=32', 'batch_size=4', 'max_iters=2', 'eval_iters=1',
               'sample_tokens=0', 'device=cpu', 'resume=true']
    pretrain(load_config(PretrainConfig, None, options))
    pretrain(load_config(PretrainConfig, None, options + ['device=auto', 'state_every=3']))  # same run: skipped
    with pytest.raises(ValueError, match='max_iters'):
        pretrain(load_config(PretrainConfig, None, options + ['max_iters=3']))


def test_resume_with_sparse_states_keeps_one_row_per_evaluation(tmp_path, monkeypatch):
    import importlib
    module = importlib.import_module('charlm.pretrain')  # the module, not the function of the same name
    corpus = tmp_path / 'corpus.txt'
    corpus.write_text(open(CORPUS).read()[:100_000])
    options = [f'data_path={corpus}', 'model.n_embd=16', 'model.n_head=2', 'model.n_layer=1', 'model.block_size=32',
               'batch_size=4', 'max_iters=8', 'eval_interval=2', 'eval_iters=1', 'sample_tokens=0', 'device=cpu',
               'patience=0', 'state_every=2']
    full, _ = pretrain(load_config(PretrainConfig, None, options + [f'out_path={tmp_path / "full.pt"}']))
    real_step = module.optimizer_step

    def crash_at_step_7(model, optimizer, loss, it, *args, **kwargs):
        if it == 7:
            raise KeyboardInterrupt
        return real_step(model, optimizer, loss, it, *args, **kwargs)
    monkeypatch.setattr(module, 'optimizer_step', crash_at_step_7)
    resumed_options = options + [f'out_path={tmp_path / "resumed.pt"}', 'resume=true']
    with pytest.raises(KeyboardInterrupt):  # the last state is from step 4, so step 6 is evaluated again on resume
        pretrain(load_config(PretrainConfig, None, resumed_options))
    monkeypatch.setattr(module, 'optimizer_step', real_step)
    resumed, _ = pretrain(load_config(PretrainConfig, None, resumed_options))
    steps = [json.loads(line)['step'] for line in open(tmp_path / 'resumed.metrics.jsonl')]
    assert steps == [0, 2, 4, 6, 8]
    assert all(torch.equal(a, b) for a, b in zip(full.state_dict().values(), resumed.state_dict().values()))


def test_old_state_without_a_new_nested_option_still_resumes(tmp_path):
    from charlm.training import load_state, make_optimizer, save_state, state_path
    cfg = PretrainConfig(out_path=str(tmp_path / 'm.pt'))
    model = tiny_model()
    optimizer = make_optimizer(model, cfg)
    save_state(cfg.out_path, model, optimizer, 3, cfg)
    state = torch.load(state_path(cfg.out_path), weights_only=True)
    del state['config']['model']['dropout']  # as if ModelConfig.dropout had been added later
    torch.save(state, state_path(cfg.out_path))
    assert load_state(cfg.out_path, model, optimizer, 'cpu', cfg)[0] == 3


@pytest.mark.parametrize('stage', ['dpo', 'grpo'])
def test_resuming_from_the_last_state_keeps_the_final_metrics(stage, tmp_path, monkeypatch):
    import importlib
    module = importlib.import_module(f'charlm.{stage}')
    corpus = tmp_path / 'corpus.txt'
    corpus.write_text(open(CORPUS).read()[:200_000])
    pretrain(load_config(PretrainConfig, None, [
        f'data_path={corpus}', f'out_path={tmp_path / "base.pt"}', 'model.n_embd=16', 'model.n_head=2',
        'model.n_layer=1', 'model.block_size=64', 'batch_size=4', 'max_iters=2', 'eval_iters=1', 'device=cpu',
        'sample_tokens=0']))
    config_cls = {'dpo': DPOConfig, 'grpo': GRPOConfig}[stage]
    options = [f'corpus_path={corpus}', f'init_from={tmp_path / "base.pt"}', f'out_path={tmp_path / "out.pt"}',
               'device=cpu', 'eval_per_task=1', 'max_iters=2', 'eval_interval=1', 'resume=true', 'batch_size=2']
    options += {'dpo': ['n_pairs=6'], 'grpo': ['group_size=2', 'max_new_tokens=6']}[stage]
    real_save = module.save_checkpoint

    def crash(*args, **kwargs):  # interrupted after the last evaluation, before the model is saved
        raise KeyboardInterrupt
    monkeypatch.setattr(module, 'save_checkpoint', crash)
    with pytest.raises(KeyboardInterrupt):
        module.__dict__[stage](load_config(config_cls, None, options))
    monkeypatch.setattr(module, 'save_checkpoint', real_save)
    module.__dict__[stage](load_config(config_cls, None, options))
    meta = load_checkpoint(str(tmp_path / 'out.pt'))[2]
    assert meta['finished'] and 'acc' in meta and meta['iter'] == 2


def test_grpo_micro_batches_give_the_same_update(tmp_path):
    corpus = tmp_path / 'corpus.txt'
    corpus.write_text(open(CORPUS).read()[:200_000])
    pretrain(load_config(PretrainConfig, None, [
        f'data_path={corpus}', f'out_path={tmp_path / "base.pt"}', 'model.n_embd=16', 'model.n_head=2',
        'model.n_layer=1', 'model.block_size=64', 'batch_size=4', 'max_iters=2', 'eval_iters=1', 'device=cpu',
        'sample_tokens=0']))
    models = []
    for micro_batch in (0, 4):
        model, _ = grpo(load_config(GRPOConfig, None, [
            f'corpus_path={corpus}', f'init_from={tmp_path / "base.pt"}', f'out_path={tmp_path / f"g{micro_batch}.pt"}',
            'device=cpu', 'eval_per_task=0', 'max_iters=2', 'batch_size=3', 'group_size=3', 'max_new_tokens=6',
            f'micro_batch={micro_batch}', 'learning_rate=0.01']))
        models.append(model)
    for a, b in zip(models[0].state_dict().values(), models[1].state_dict().values()):
        assert torch.allclose(a, b, atol=1e-6)


def test_prepare_data_rejects_truncated_downloads_and_redoes_edited_files(tmp_path):
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    from charlm.datasets import normalize, prepare

    class Truncated(BaseHTTPRequestHandler):
        def do_GET(self):  # announces 1000 bytes but sends 12, as when a connection drops
            self.send_response(200)
            self.send_header('Content-Length', '1000')
            self.end_headers()
            self.wfile.write(b'Once upon a ')

        def log_message(self, *args):
            pass
    server = HTTPServer(('127.0.0.1', 0), Truncated)
    threading.Thread(target=server.handle_request, daemon=True).start()
    out = tmp_path / 'stories.txt'
    with pytest.raises(IOError, match='ended early'):
        prepare('tinystories', str(out), url=f'http://127.0.0.1:{server.server_port}/x.txt')
    server.server_close()
    assert not out.exists() and not (tmp_path / 'stories.txt.tmp').exists()

    text = tmp_path / 'mine.txt'
    text.write_text('first version\n')
    joined = str(tmp_path / 'joined.txt')
    prepare('files', joined, files=[str(text)])
    text.write_text('second, longer version\n')
    prepare('files', joined, files=[str(text)])  # the input changed, so it is prepared again
    assert open(joined).read() == 'second, longer version\n'
    with pytest.raises(ValueError, match='only for'):
        prepare('tinystories', files=[str(text)])
    assert normalize('⅛ cup and 3⁄4') == '1/8 cup and 3/4'


def test_paths_with_pattern_characters_are_read_as_files(tmp_path):
    from charlm.data import expand_paths
    odd = tmp_path / 'draft[1].txt'
    odd.write_text('x')
    folder = tmp_path / 'set[a]'
    folder.mkdir()
    (folder / 'f.txt').write_text('y')
    assert expand_paths(str(odd)) == [str(odd)] and expand_paths(str(folder)) == [str(folder / 'f.txt')]


def test_cli_shows_reasoning_and_writes_reasoning_data(tmp_path, capsys):
    from charlm.cli import format_reply
    assert format_reply('evol') == 'evol'
    assert format_reply('<|think|>7+5+0=12 A=2 => 12<|/think|>12') == '[thinking: 7+5+0=12 A=2 => 12] 12'
    assert format_reply('<|think|>7+5+0=1') == '[thinking, unfinished: 7+5+0=1]'
    for flag in ([], ['--reasoning']):
        out = tmp_path / f'data{len(flag)}.jsonl'
        cli_main(['make-sft-data', '--out', str(out), '--n', '50', '--tasks', 'reverse', 'sub', '--device', 'cpu']
                 + flag)
        rows = read_jsonl(str(out))
        assert any(r['task'] == 'sub' for r in rows)
        assert all((THINK in r['response']) == (bool(flag) and r['task'] == 'sub') for r in rows)
    model, tok = tiny_model(vocab_size=102, block_size=64), CharTokenizer()
    tok.add_special_tokens(REASONING_TOKENS)
    with torch.no_grad():  # a model that always starts to reason and never finishes
        model.lm_head.bias[tok.stoi[THINK]] = 100.0
    save_checkpoint(str(tmp_path / 'm.pt'), model, tok)
    capsys.readouterr()
    cli_main(['chat', '--model', str(tmp_path / 'm.pt'), 'What is 1 + 2?', '--device', 'cpu', '--max-new-tokens', '5'])
    assert capsys.readouterr().out.startswith('[thinking, unfinished: ')
