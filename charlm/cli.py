"""Command-line interface: python -m charlm <command> --help"""
import argparse
import json
from collections import Counter

import torch

from .chat import sample_replies
from .checkpoint import load_checkpoint
from .config import load_config, to_dict
from .data import load_text, write_jsonl
from .datasets import SOURCES, prepare
from .dpo import DPOConfig, dpo
from .evaluation import evaluate_tasks
from .grpo import GRPOConfig, grpo
from .pretrain import PretrainConfig, pretrain
from .reasoning import format_response, split_reply
from .sft import SFTConfig, sft
from .tasks import ALL_TASKS, CHECKABLE_TASKS, TASKS, VERIFIABLE_TASKS, TaskSuite
from .tokenizer import END_THINK, THINK
from .training import resolve_device

STAGES = {
    'pretrain': (PretrainConfig, pretrain, 'pretrain a base model on a text file'),
    'sft': (SFTConfig, sft, 'supervised fine-tuning into an instruction-following model'),
    'dpo': (DPOConfig, dpo, 'Direct Preference Optimization'),
    'grpo': (GRPOConfig, grpo, 'Group Relative Policy Optimization (RL with a verifiable reward)'),
}


def run_stage(args):
    config_cls, train, _ = STAGES[args.command]
    cfg = load_config(config_cls, args.config, args.set)
    if args.print_config:
        print(json.dumps(to_dict(cfg), indent=2))
        return
    train(cfg)


def run_generate(args):
    torch.manual_seed(args.seed)
    model, tokenizer, _ = load_checkpoint(args.model, args.device)
    ids = tokenizer.encode(args.prompt) or [tokenizer.stoi['\n']]
    out = model.generate(torch.tensor([ids], device=args.device), args.max_new_tokens,
                         temperature=args.temperature, top_k=args.top_k)
    print(tokenizer.decode(out[0].tolist()))


def format_reply(reply):
    """ a reply for printing: a reasoning model's scratchpad in brackets before its answer. A scratchpad without
    <|/think|> is unfinished: the model stopped early, or ran into --max-new-tokens or the end of its context """
    if THINK not in reply:
        return reply
    reasoning, answer = split_reply(reply)
    if END_THINK not in reply.split(THINK, 1)[1]:
        return f"[thinking, unfinished: {reasoning}]"
    return f"[thinking: {reasoning}] {answer}"


def run_chat(args):
    torch.manual_seed(args.seed)
    model, tokenizer, _ = load_checkpoint(args.model, args.device)

    def respond(message):
        reply = sample_replies(model, tokenizer, [message], 1, args.max_new_tokens, args.temperature,
                               args.top_k)[0][0][1]
        print(format_reply(reply))

    if args.message:
        respond(' '.join(args.message))
        return
    print("type a message (empty line to quit), e.g. 'Reverse the word: love' or 'What is 47 + 85?'")
    while True:
        try:
            message = input('> ')
        except EOFError:
            break
        if not message.strip():
            break
        respond(message)


def run_eval(args):
    suite = TaskSuite(load_text(args.corpus), eval_phrasing=args.phrasing)
    examples = suite.eval_set(args.n_per_task, args.tasks)
    distinct = Counter(e.task for e in {(e.task, e.prompt): e for e in examples}.values())
    print(f"{args.n_per_task} held-out prompts per task (distinct: "
          + ', '.join(f'{t} {distinct[t]}' for t in args.tasks) + ')')
    print(f"{'model':<40} " + ' '.join(f'{t:>9}' for t in args.tasks) + f" {'overall':>9}")
    for path in args.model:
        model, tokenizer, _ = load_checkpoint(path, args.device)
        results, replies = evaluate_tasks(model, tokenizer, examples, args.max_new_tokens, return_replies=True)
        print(f"{path:<40} " + ' '.join(f"{results[f'acc/{t}']:>9.1%}" for t in args.tasks)
              + f" {results['acc']:>9.1%}")
        if any(k.startswith('trace/') for k in results):  # how often the reasoning is exactly the taught method
            print(f"{'  reasoning matches the method':<40} "
                  + ' '.join(f"{results[f'trace/{t}']:>9.1%}" if f'trace/{t}' in results else f"{'':>9}"
                             for t in args.tasks))
        for example, reply in list(zip(examples, replies))[::args.n_per_task][:args.show]:
            expected = format_response(example.answer, example.reasoning) if THINK in reply else example.answer
            print(f"    {example.prompt!r}\n      -> {format_reply(reply)!r}"
                  f"\n      expected {format_reply(expected)!r}")


def run_make_sft_data(args):
    suite = TaskSuite(load_text(args.corpus))
    rows = [{'task': e.task, 'prompt': e.prompt,
             'response': format_response(e.answer, e.reasoning if args.reasoning else '')}
            for e in suite.sample(args.n, args.tasks, args.split, args.seed)]
    write_jsonl(args.out, rows)
    print(f"wrote {len(rows)} examples to {args.out}")


def run_prepare_data(args):
    try:
        prepare(args.source, args.out, args.files, args.url, args.max_chars, args.force)
    except ValueError as e:  # wrong arguments
        raise SystemExit(f"error: {e}")


def main(argv=None):
    parser = argparse.ArgumentParser(prog='charlm', description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)

    for name, (_, _, help_text) in STAGES.items():
        p = sub.add_parser(name, help=help_text)
        p.add_argument('--config', help='JSON config file (defaults are used for missing options)')
        p.add_argument('--set', nargs='*', default=[], metavar='KEY=VALUE',
                       help='override config options, e.g. --set max_iters=100 model.n_layer=2')
        p.add_argument('--print-config', action='store_true', help='print the resolved config and exit')
        p.set_defaults(func=run_stage)

    p = sub.add_parser('generate', help='continue a text prompt with any model')
    p.add_argument('--model', required=True)
    p.add_argument('--prompt', default='', help='text to continue (default: start from a newline)')
    p.add_argument('--max-new-tokens', type=int, default=500)
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--top-k', type=int, default=None)
    p.set_defaults(func=run_generate)

    p = sub.add_parser('chat', help='send a message to a fine-tuned model (interactive without a message)')
    p.add_argument('--model', required=True)
    p.add_argument('message', nargs='*')
    p.add_argument('--max-new-tokens', type=int, default=128)
    p.add_argument('--temperature', type=float, default=0.0, help='0 = greedy')
    p.add_argument('--top-k', type=int, default=None)
    p.set_defaults(func=run_chat)

    p = sub.add_parser('eval', help='accuracy of one or more models on held-out task prompts')
    p.add_argument('--model', required=True, nargs='+')
    p.add_argument('--corpus', default='data/input.txt')
    p.add_argument('--tasks', nargs='+', default=list(VERIFIABLE_TASKS), choices=CHECKABLE_TASKS)
    p.add_argument('--n-per-task', type=int, default=200)
    p.add_argument('--max-new-tokens', type=int, default=128, help='reply length limit (reasoning needs ~100)')
    p.add_argument('--show', type=int, default=0, help='print this many example replies per model')
    p.add_argument('--phrasing', type=int, choices=range(4), default=None,
                   help='ask word problems in this phrasing (0-2 are trained on; default: the held-out 3)')
    p.set_defaults(func=run_eval)

    p = sub.add_parser('make-sft-data', help='write synthetic task examples to a JSONL file')
    p.add_argument('--out', required=True)
    p.add_argument('--corpus', default='data/input.txt')
    p.add_argument('--n', type=int, default=20000)
    p.add_argument('--tasks', nargs='+', default=list(ALL_TASKS), choices=TASKS)
    p.add_argument('--split', default='train', choices=['train', 'eval'])
    p.add_argument('--reasoning', action='store_true', help='math replies include the <|think|> scratchpad')
    p.set_defaults(func=run_make_sft_data)

    p = sub.add_parser('prepare-data', help='download and clean a larger pretraining corpus',
                       description='download and clean a pretraining corpus: ' + ', '.join(SOURCES) +
                                   ', or your own --files (see charlm/datasets.py)')
    p.add_argument('source', choices=list(SOURCES) + ['files'])
    p.add_argument('--out', help='output file (default: data/<source>.txt; required for files)')
    p.add_argument('--files', nargs='+', default=[], help="the .txt files to join, for the 'files' source")
    p.add_argument('--url', help='download from this address instead of the default')
    p.add_argument('--max-chars', type=int, default=None, help='stop after this many characters')
    p.add_argument('--force', action='store_true', help='prepare again even if the output is up to date')
    p.set_defaults(func=run_prepare_data)

    for name, p in sub.choices.items():
        if p.get_default('func') not in (run_stage, run_prepare_data):
            p.add_argument('--device', default='auto', help='auto, cpu, cuda or mps')
            p.add_argument('--seed', type=int, default=0)

    args = parser.parse_args(argv)
    if hasattr(args, 'device'):
        args.device = resolve_device(args.device)
    args.func(args)


if __name__ == '__main__':
    main()
