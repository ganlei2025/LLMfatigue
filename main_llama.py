import datetime
import functools
import json
import os
import re
import sys
import numpy as np
import openai
import argparse
from time import time
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Inner
from optimzier_utils import call_optimizer, organize
from evaluation import *
import prompt_utils
from utils import initilization, test_accept
from logger import StatsLogger


def main():
    parser = argparse.ArgumentParser(description='LLM for knowledge discovery')

    # —— LLM 相关参数 ——
    parser.add_argument('--LLM_name', type=str, default="llama-7b",
                        help='choose LLM: gpt-3.5-turbo / gpt-4 / llama-7b / llama-13b')
    parser.add_argument('--openai-api-key', type=str, default="", help='OpenAI API Key')
    parser.add_argument('--new-add', type=int, default=0)
    parser.add_argument('--gpt_temperature', type=float, default=1.0)
    parser.add_argument('--max_decode_length', type=int, default=1024)

    # —— Symbolic Regression 相关参数 ——
    parser.add_argument('--N', type=int, default=4)
    parser.add_argument('--init-num', type=int, default=20)
    parser.add_argument('--operators', type=str, default="{+, -, *, /, ^2}")
    parser.add_argument('--operands', type=str, default="{u, u_x, u_xx, u_xxx, x}")
    parser.add_argument('--optimize-type', type=str, default='evolution_optimize')
    parser.add_argument('--evo-type', type=str, default='term', choices=['term', 'equation'])
    parser.add_argument('--reward_limit', type=float, default=0.5)
    parser.add_argument('--sort', type=str, default="not_reverse",
                        choices=['not_reverse', 'reverse', 'Not sorted'])

    parser.add_argument('--max_epoch', type=int, default=50)
    parser.add_argument('--threshold', type=float, default=0.995)

    # —— Dataset ——
    parser.add_argument('--data-name', type=str, default='chafee-infante')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--max-terms', type=int, default=5)
    parser.add_argument('--metric', type=str, default="sparse_reward")
    parser.add_argument('--mode', type=str, default="sparse_regression",
                        choices=['sparse_regression', 'regression', 'nonlinear'])
    parser.add_argument('--metric_params', nargs='+', type=float, default=[0.01])
    parser.add_argument('--job-name', type=str, default='exp')
    parser.add_argument('--logdir', type=str, default="./log")
    parser.add_argument('--use-pqt', type=int, default=1)
    parser.add_argument('--add-const', type=int, default=0)
    parser.add_argument('--noise', type=float, default=0)

    args = parser.parse_args()
    print(args)

    # —— 设置 OpenAI API（仅云模型有效） ——
    if args.new_add:
        openai.api_base = "https://oneapi.xty.app/v1"
    openai.api_key = args.openai_api_key

    # —— 决定使用本地模型还是 API ——
    llm_name = args.LLM_name

    assert llm_name in {"gpt-3.5-turbo", "gpt-4", "llama-7b", "llama-13b"}

    # 🎯 —— 加载本地 Llama 模型 ——
    if "llama" in llm_name:
        model_pretrained, tokenizer = prompt_utils.load_model(llm_name)
    else:
        model_pretrained, tokenizer = None, None

    # —— LLM 配置字典 ——
    llm_dict = dict()
    llm_dict['name'] = llm_name
    llm_dict["max_decode_steps"] = args.max_decode_length
    llm_dict["temperature"] = args.gpt_temperature
    llm_dict["batch_size"] = 1
    llm_dict["model_pretrained"] = model_pretrained
    llm_dict["tokenizer"] = tokenizer

    # —— 数据评估器 ——
    evaluator = Evaluator(
        args.data_name,
        metric=args.metric,
        metric_params=args.metric_params,
        max_terms=args.max_terms,
        mode=args.mode,
        add_const=args.add_const,
        noise=args.noise
    )

    logger = StatsLogger(args)

    print("\n===== Initialization =====")
    result_init, best_sample, eqs, dur_time = call_optimizer(
        llm_dict, args, evaluator, call_type='initialization'
    )

    optimizer_input = result_init
    total_request_time = dur_time

    print("\n===== Optimization Loop =====")
    optimize_type = args.optimize_type.split("_")

    for i in range(args.max_epoch):
        begin_time = time()
        cur_opt = optimize_type[i % 2]

        logger.save_stats(optimizer_input, i)

        info_str, info_score = organize(
            optimizer_input,
            cur_opt,
            args.sort,
            evaluator.pq,
            use_pqt=args.use_pqt
        )

        evaluator.pq.push(optimizer_input)
        print(f"Iteration {i}, Score = {info_score}")

        optimizer_input, best_sample_cur, all_eqs, dur_time = call_optimizer(
            llm_dict, args, evaluator,
            info=info_str,
            call_type=cur_opt
        )

        total_request_time += dur_time
        best_sample = best_sample_cur

        print(f"  Best: {best_sample.exp_str}, Reward={best_sample.score}")

        if best_sample.score >= args.threshold and test_accept(best_sample):
            print("Early stopping!")
            break

    print("\n===== Finished =====")
    logger.save_results()

    print(f"Total request time: {total_request_time}")
    print(f"Total invalid count: {sum(list(evaluator.invalid.values()))}")


if __name__ == "__main__":
    main()
