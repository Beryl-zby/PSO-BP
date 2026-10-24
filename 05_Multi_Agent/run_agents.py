"""Local Qwen role agents for evidence-based water-demand model combination.

Only training_evidence.json is passed to inference. Read evaluation targets after
all five decisions have been saved. Existing numerical models do the arithmetic.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent
EXPERTS = ('pso', 'gpr', 'ols')
ROLES = {
    'accuracy': '你是需水预测精度审查员，重点比较训练期末3年内验证的平均绝对百分比误差。',
    'stability': '你是小样本稳定性审查员，重点比较3年误差波动和最坏年份误差，避免只看均值。',
    'scenario': '你是节水情景审查员，重点检查农业、工业驱动负响应；人口相关性不作因果解释。兼顾内验证误差。',
}
COORDINATOR = ('你是水资源预测协调者，综合平均误差、最坏误差及农业工业响应方向，'
               '在三个已有数值模型间分配预测权重。证据只有少量年份，勿宣称统计显著或因果关系。')
DECISION_SCHEMA = {
    'type': 'object', 'required': ['weights', 'reason'], 'additionalProperties': False,
    'properties': {
        'weights': {'type': 'object', 'required': list(EXPERTS), 'additionalProperties': False,
                    'properties': {key: {'type': 'number', 'minimum': 0, 'maximum': 1} for key in EXPERTS}},
        'reason': {'type': 'string'},
    },
}
OUTPUT_RULE = ('只根据所给训练期证据决策，不能猜测测试年份真实值。'
               '输出且仅输出符合以下JSON Schema的JSON对象：' + json.dumps(DECISION_SCHEMA) +
               '三个权重必须非负且总和为1，允许权重为0。reason用不超过60字说明证据。不要输出预测需水量。')



def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def normalize(values):
    if len(values) != 3 or any(isinstance(v, bool) or not isinstance(v, (int, float))
                               or not math.isfinite(v) or v < 0 for v in values):
        raise ValueError('Expected three finite nonnegative numerical weights')
    total = math.fsum(values)
    if total <= 0:
        raise ValueError('All-zero weights are invalid')
    return [v / total for v in values]


def parse_decision(text):
    text = text.strip()
    if text.startswith('```') and text.endswith('```'):
        text = '\n'.join(text.splitlines()[1:-1])
    response = json.loads(text)
    if set(response['weights']) != set(EXPERTS) or not isinstance(response['reason'], str):
        raise ValueError('Invalid decision schema')
    weights = normalize([response['weights'][key] for key in EXPERTS])
    return {'weights': dict(zip(EXPERTS, weights)), 'reason': response['reason']}


def make_prompt(packet, role, reviews=None):
    instruction = ROLES[role] if role in ROLES else COORDINATOR
    prompt = instruction + '\n' + OUTPUT_RULE + '\n'
    prompt += 'cv误差越低越好。负响应计数越高表示反常点越多，0为未见反常；顺序[农业,工业,人口]。人口不做因果解释。这是训练输入的1%扰动诊断，不是未来保证。\n'
    prompt += '训练期证据：' + json.dumps(packet, ensure_ascii=False, separators=(',', ':'))
    if reviews is not None:
        prompt += '\n三位审查员意见：' + json.dumps(reviews, ensure_ascii=False, separators=(',', ':'))
        prompt += '\n审查意见可能有错，请核对上面的数值证据后独立协调。'
    return prompt


class LocalQwen:
    def __init__(self, model_path):
        from mlx_vlm import load, generate
        from mlx_vlm.prompt_utils import apply_chat_template
        from mlx_vlm.structured import build_json_schema_logits_processor
        self.generate = generate
        self.template = apply_chat_template
        self.schema_processor = build_json_schema_logits_processor
        model_path = Path(model_path).expanduser().resolve()
        if not model_path.is_dir():
            raise FileNotFoundError('Set --model-path to the existing local MLX Qwen model directory')
        self.model, self.processor = load(str(model_path), trust_remote_code=False, local_files_only=True)
        self.metadata = {
            'model_id': 'mlx-community/Qwen3-VL-4B-Instruct-4bit',
            'model_directory_name': model_path.name,
            'config_sha256': sha256(model_path / 'config.json'),
            'weights_sha256': sha256(model_path / 'model.safetensors'),
            'mlx_vlm_version': importlib.metadata.version('mlx-vlm'),
            'mlx_version': importlib.metadata.version('mlx'),
            'input_mode': 'text_only', 'temperature': 0.0, 'max_tokens': 256,
            'fine_tuned_for_water': False, 'json_schema_constrained': True,
        }

    def __call__(self, prompt):
        formatted = self.template(self.processor, self.model.config, prompt=prompt, num_images=0)
        tokenizer = self.processor.tokenizer if hasattr(self.processor, 'tokenizer') else self.processor
        constraint = self.schema_processor(tokenizer, DECISION_SCHEMA)
        started = time.perf_counter()
        generated = self.generate(self.model, self.processor, formatted, max_tokens=256,
                                  temperature=0.0, verbose=False, logits_processors=[constraint])
        return {
            'raw_response': generated.text,
            'prompt_tokens': int(generated.prompt_tokens),
            'generation_tokens': int(generated.generation_tokens),
            'finish_reason': generated.finish_reason,
            'seconds': time.perf_counter() - started,
        }


def decide_fold(packet, infer, trace, trace_file=None):
    """Explicit DAG: three independent reviewers -> coordinator; single-agent ablation."""
    def ask(role, reviews=None):
        prompt = make_prompt(packet, role, reviews)
        response = infer(prompt)
        record = {'training_end_year': packet['training_end_year'], 'role': role,
                  'prompt': prompt, **response}
        # Keep failed outputs reviewable; never silently replace an LLM with a rule.
        trace.append(record)
        if trace_file:
            write_json(trace_file, trace)
        decision = parse_decision(response['raw_response'])
        record['decision'] = decision
        if trace_file:
            write_json(trace_file, trace)
        return decision

    reviews = {role: ask(role) for role in ROLES}
    multi = ask('coordinator', reviews)
    single = ask('single_coordinator')
    inverse = normalize([1 / max(expert['cv_mape'], 1e-8) for expert in packet['experts']])
    return {
        'training_end_year': packet['training_end_year'], 'training_rows': packet['training_rows'],
        'multi_agent': multi, 'single_agent': single, 'reviews': reviews,
        'inverse_cv_weights': dict(zip(EXPERTS, inverse)),
        'equal_weights': dict(zip(EXPERTS, [1 / 3] * 3)),
    }


def get_weights(decision, strategy):
    obj = decision[strategy]['weights'] if strategy in ('multi_agent', 'single_agent') else decision[strategy + '_weights']
    return [obj[key] for key in EXPERTS]


def dot(values, weights):
    return sum(value * weight for value, weight in zip(values, weights))


def metrics(actual, prediction):
    errors = [p - y for p, y in zip(prediction, actual)]
    mse = sum(e * e for e in errors) / len(errors)
    return [100 * sum(abs(e / y) for e, y in zip(errors, actual)) / len(errors),
            sum(abs(e) for e in errors) / len(errors), math.sqrt(mse), mse]


def evaluate(bank, evidence, decisions):
    results = []
    labels = {'multi_agent': '千问多智能体协作', 'single_agent': '千问单智能体',
              'inverse_cv': '内验证误差倒数加权', 'equal': '三个专家等权平均'}
    for strategy, name in labels.items():
        weights = [get_weights(d, strategy) for d in decisions]
        fixed = [dot(row, weights[0]) for row in bank['fixed_prediction']]
        rolling = [dot(row, weights[i]) for i, row in enumerate(bank['rolling_prediction'])]
        future = []
        negative = [0] * 3
        for point in bank['future']:
            value = dot(point['expert_prediction'], weights[-1])
            for c, row in enumerate(point['down_expert_prediction']):
                negative[c] += value - dot(row, weights[-1]) < -1e-7
            future.append({key: point[key] for key in ('year', 'scenario', 'path')})
            future[-1]['prediction'] = value
        lookup = {(p['year'], p['scenario'], p['path']): p['prediction'] for p in future}
        path_reversals = 0
        for year in bank['future_years']:
            for scenario in bank['scenarios']:
                p = [lookup[(year, scenario, path)] for path in bank['paths']]
                path_reversals += p[0] < p[1] - 1e-7 or p[1] < p[2] - 1e-7
        fm, rm = metrics(bank['actual'], fixed), metrics(bank['actual'], rolling)
        results.append({'id': strategy, 'method': name, 'weights_by_training_end_year': weights,
                        'fixed_prediction': fixed, 'rolling_prediction': rolling,
                        'fixed_metrics': fm, 'rolling_metrics': rm, 'score': (fm[0] + rm[0]) / 2,
                        'future_prediction': future, 'direction_counts': negative + [sum(p['prediction'] <= 0 for p in future), path_reversals]})
    comparison = [
        {'method': name, 'fixed_metrics': fm, 'rolling_metrics': rm, 'score': (fm[0] + rm[0]) / 2}
        for name, fm, rm in zip(bank['baseline_methods'], bank['baseline_fixed_metrics'], bank['baseline_rolling_metrics'])
    ]
    comparison.extend({k: r[k] for k in ('method', 'fixed_metrics', 'rolling_metrics', 'score')} for r in results)
    return {'input_sha256': evidence['input_sha256'], 'validation_years': bank['validation_years'],
            'actual': bank['actual'], 'expert_order': list(EXPERTS),
            'metric_columns': ['MAPE_percent', 'MAE', 'RMSE', 'MSE'],
            'direction_columns': ['农业负响应', '工业负响应', '人口负响应', '非正需水', '路径反序'],
            'validation_scope': '2020—2023已知数据开发回测；两种检验复用相同四年，无新增独立测试集',
            'results': results, 'ranking': sorted(comparison, key=lambda row: row['score'])}


def write_report(result, output):
    lines = ['# 本地千问多智能体：实测结果', '',
             '三个角色分别审查预测精度、误差稳定性、农业工业节水响应，再由协调者分配PSO-BP/GPR/OLS权重。',
             '同一个本地Qwen模型执行不同角色；每折4次调用完成多智能体，另1次为单智能体对照。无需训练或微调千问。',
             '数值模型仍负责预测，千问只返回非负权重；代码归一化后计算总需水量。不是农业、工业、生活分部门需水的物理仿真。', '',
             '**开发回测：2020—2023已参与模型探索；不是新独立测试。只运行一种千问模型、一次确定性解码，未证明统计显著性。**', '',
             '| 方法 | 固定MAPE (%) | 滚动MAPE (%) | 两项均值 (%) |', '|---|---:|---:|---:|']
    for row in result['ranking']:
        lines.append(f"| {row['method']} | {row['fixed_metrics'][0]:.6f} | {row['rolling_metrics'][0]:.6f} | {row['score']:.6f} |")
    lines += ['', '## 多智能体逐折决策', '', '| 训练截止年 | PSO-BP权重 | GPR权重 | OLS权重 | 协调者解释 |', '|---|---:|---:|---:|---|']
    for decision in result['decisions']:
        values = list(decision['multi_agent']['weights'].values())
        reason = decision['multi_agent']['reason'].replace('|', '/').replace('\n', ' ')
        lines.append(f"| {decision['training_end_year']} | {values[0]:.6f} | {values[1]:.6f} | {values[2]:.6f} | {reason} |")
    lines += ['', '解释为模型原文，可能有判断错误；以training_evidence.json中的数值为准。', '',
              '## 逐年需水预测', '', '| 年份 | 实际 | 多智能体固定 | 多智能体滚动 | 单智能体固定 | 单智能体滚动 |', '|---|---:|---:|---:|---:|---:|']
    multi, single = result['results'][:2]
    for i, year in enumerate(result['validation_years']):
        values = [result['actual'][i], multi['fixed_prediction'][i], multi['rolling_prediction'][i], single['fixed_prediction'][i], single['rolling_prediction'][i]]
        lines.append('| ' + str(year) + ' | ' + ' | '.join(f'{v:.6f}' for v in values) + ' |')
    lines += ['', '## 数据边界与方向检查', '',
              '每折只将当时训练前缀的3年内验证误差、已选配置和训练输入方向诊断交给千问。固定起点四年共用2019年的决策；滚动分别用2019/2020/2021/2022年的决策；未来情景用2023年决策。',
              '所有决策先完成并落盘，之后才读取evaluation_bank.json中的检验真实值与数值模型预测。训练前缀中包含较早的检验年份，符合逐年扩展历史协议。',
              '前缀模型配置继承此前数值实验，整个工作仍处于开发回测阶段。农业/工业方向检查是诊断，并非正权重就能消除每个专家的非单调响应。', '',
              '| 方法 | 农业反向 | 工业反向 | 人口反向 | 非正需水 | 路径反序 |', '|---|---:|---:|---:|---:|---:|']
    for row in result['results']:
        lines.append('| ' + row['method'] + ' | ' + ' | '.join(str(v) for v in row['direction_counts']) + ' |')
    lines += ['', '每种方案检查105个未来输入点；没有用此结果回调权重。', '',
              'trace.json保存25次真实推理的完整输入和原始输出，decisions.json保存5折决策；result.json保存指标、预测和105点未来情景。',
              '单智能体与多智能体使用相同证据、数值专家和temperature=0；简单等权/误差倒数加权不调用大模型。',
              '权重和模型原文解释不等于物理贡献率；本实验不提供供水能力约束、区域博弈或实际调度优化。']
    (output / '结果报告.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    default = Path.home() / 'Desktop/羽毛球论文/04_原型/models/vision/Qwen3-VL-4B-Instruct-4bit'
    parser.add_argument('--model-path', default=os.environ.get('WATER_QWEN_MODEL', str(default)))
    parser.add_argument('--output', type=Path, default=HERE / 'results')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    evidence_file = HERE / 'results/training_evidence.json'
    evidence = read_json(evidence_file)
    qwen = LocalQwen(args.model_path)
    write_json(args.output / 'runtime.json', qwen.metadata)
    trace, decisions = [], []
    started = time.perf_counter()
    for packet in evidence['folds']:
        decision = decide_fold(packet, qwen, trace, args.output / 'trace.json')
        decisions.append(decision)
        write_json(args.output / 'decisions.json', decisions)
        print(f"DECISION {packet['training_end_year']}: {decision['multi_agent']['weights']}", flush=True)
    # Evaluation labels are opened only AFTER the decisions have been frozen.
    bank = read_json(HERE / 'results/evaluation_bank.json')
    result = evaluate(bank, evidence, decisions)
    result.update(model=qwen.metadata, decisions=decisions, inference_calls=len(trace),
                  inference_seconds=time.perf_counter() - started,
                  evidence_file_sha256=sha256(evidence_file), decisions_sha256=sha256(args.output / 'decisions.json'))
    write_json(args.output / 'result.json', result)
    write_report(result, args.output)
    for row in result['results']:
        print(row['method'], row['fixed_metrics'][0], row['rolling_metrics'][0], flush=True)


if __name__ == '__main__':
    main()
