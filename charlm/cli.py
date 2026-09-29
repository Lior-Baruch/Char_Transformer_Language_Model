"""Command-line interface: python -m charlm <command> --help"""
import argparse
import json

import torch

from .chat import chat
from .checkpoint import load_checkpoint
from .config import load_config, to_dict
from .data import load_text, write_jsonl
from .dpo import DPOConfig, dpo
from .evaluation import evaluate_tasks
from .grpo import GRPOConfig, grpo
from .pretrain import PretrainConfig, pretrain
from .sft import SFTConfig, sft
from .tasks import ALL_TASKS, VERIFIABLE_TASKS, TaskSuite

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


def run_chat(args):
    torch.manual_seed(args.seed)
    model, tokenizer, _ = load_checkpoint(args.model, args.device)
    reply = lambda m: chat(model, tokenizer, m, args.max_new_tokens, args.temperature, args.top_k)
    if args.message:
        print(reply(' '.join(args.message)))
        return
    print("type a message (empty line to quit), e.g. 'Reverse the word: love'")
    while True:
        try:
            message = input('> ')
        except EOFError:
            break
        if not message.strip():
            break
        print(reply(message))


def run_eval(args):
    suite = TaskSuite(load_text(args.corpus))
    examples = suite.eval_set(args.n_per_task, args.tasks)
    print(f"{'model':<40} " + ' '.join(f'{t:>9}' for t in args.tasks) + f" {'overall':>9}")
    for path in args.model:
        model, tokenizer, _ = load_checkpoint(path, args.device)
        results, replies = evaluate_tasks(model, tokenizer, examples, return_replies=True)
        print(f"{path:<40} " + ' '.join(f"{results[f'acc/{t}']:>9.1%}" for t in args.tasks)
              + f" {results['acc']:>9.1%}")
        for example, reply in list(zip(examples, replies))[::args.n_per_task][:args.show]:
            print(f"    {example.prompt!r} -> {reply!r}  (expected {example.answer!r})")


def run_make_sft_data(args):
    suite = TaskSuite(load_text(args.corpus))
    rows = [{'task': e.task, 'prompt': e.prompt, 'response': e.answer}
            for e in suite.sample(args.n, args.tasks, args.split, args.seed)]
    write_jsonl(args.out, rows)
    print(f"wrote {len(rows)} examples to {args.out}")


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
    p.add_argument('--max-new-tokens', type=int, default=64)
    p.add_argument('--temperature', type=float, default=0.0, help='0 = greedy')
    p.add_argument('--top-k', type=int, default=None)
    p.set_defaults(func=run_chat)

    p = sub.add_parser('eval', help='accuracy of one or more models on held-out task prompts')
    p.add_argument('--model', required=True, nargs='+')
    p.add_argument('--corpus', default='data/input.txt')
    p.add_argument('--tasks', nargs='+', default=list(VERIFIABLE_TASKS), choices=VERIFIABLE_TASKS)
    p.add_argument('--n-per-task', type=int, default=200)
    p.add_argument('--show', type=int, default=0, help='print this many example replies per model')
    p.set_defaults(func=run_eval)

    p = sub.add_parser('make-sft-data', help='write synthetic task examples to a JSONL file')
    p.add_argument('--out', required=True)
    p.add_argument('--corpus', default='data/input.txt')
    p.add_argument('--n', type=int, default=20000)
    p.add_argument('--tasks', nargs='+', default=list(ALL_TASKS), choices=ALL_TASKS)
    p.add_argument('--split', default='train', choices=['train', 'eval'])
    p.set_defaults(func=run_make_sft_data)

    for p in sub.choices.values():
        if p.get_default('func') is not run_stage:
            p.add_argument('--device', default='cpu')
            p.add_argument('--seed', type=int, default=0)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == '__main__':
    main()
