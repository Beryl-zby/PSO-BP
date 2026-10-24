"""Verify real saved Qwen traces, evidence boundaries and computed forecasts.

No model download or inference required: replay the recorded real outputs.
"""
import copy
import importlib.util
import json
import math
from pathlib import Path
import statistics
import unittest

ROOT = Path(__file__).resolve().parents[1]
METHOD = ROOT / '05_Multi_Agent'
spec = importlib.util.spec_from_file_location('water_agents', METHOD / 'run_agents.py')
agent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agent)


def load(name):
    return json.loads((METHOD / 'results' / name).read_text(encoding='utf-8'))


class MultiAgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evidence = load('training_evidence.json')
        cls.bank = load('evaluation_bank.json')
        cls.result = load('result.json')
        cls.trace = load('trace.json')
        cls.decisions = load('decisions.json')

    def assert_nested_close(self, actual, expected):
        if isinstance(expected, dict):
            self.assertEqual(set(actual), set(expected))
            for key in expected:
                self.assert_nested_close(actual[key], expected[key])
        elif isinstance(expected, list):
            self.assertEqual(len(actual), len(expected))
            for x, y in zip(actual, expected):
                self.assert_nested_close(x, y)
        elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
            self.assertTrue(math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-10), (actual, expected))
        else:
            self.assertEqual(actual, expected)

    def test_time_prefix_and_source_evidence(self):
        source = json.loads((ROOT / 'results/three_methods/comparison.json').read_text())
        self.assertEqual(self.evidence['input_sha256'], source['input_sha256'])
        self.assertEqual(self.evidence['input_sha256'], agent.sha256(ROOT / 'data/需水预测模型对照.xlsx'))
        self.assertEqual(self.result['evidence_file_sha256'], agent.sha256(METHOD / 'results/training_evidence.json'))
        self.assertEqual(self.result['decisions_sha256'], agent.sha256(METHOD / 'results/decisions.json'))
        for fold, packet in enumerate(self.evidence['folds']):
            year = 2019 + fold
            self.assertEqual(packet['training_end_year'], year)
            self.assertEqual(packet['training_rows'], 10 + fold)
            self.assertEqual(set(packet), {'training_end_year', 'training_rows', 'experts'})
            for expert, index in zip(packet['experts'], [0, 2, 5]):
                self.assertEqual(expert['cv_years'], list(range(year - 2, year + 1)))
                original = source['results'][index]
                flat = original['tuning_rows']  # MATLAB cell arrays serialize column-major
                count = len(flat) // 7
                all_rows = [[flat[c * count + i] for c in range(7)] for i in range(count)]
                rows = [row for row in all_rows if row[0] == year and row[1] == original['selected'][fold]['label']]
                self.assertEqual(expert['cv_absolute_percent_errors'], [row[6] for row in rows])
                self.assertAlmostEqual(expert['cv_mape'], statistics.mean(row[6] for row in rows))
                self.assertTrue(all(row[3] < row[2] <= year for row in rows))
        self.assertEqual(self.bank['fixed_prediction'], [[r[i] for i in [0, 2, 5]] for r in source['fixed_prediction']])
        self.assertEqual(self.bank['rolling_prediction'], [[r[i] for i in [0, 2, 5]] for r in source['rolling_prediction']])

    def test_replay_real_calls_and_no_evaluation_in_prompts(self):
        self.assertEqual(len(self.trace), 25)
        cursor, replayed = 0, []

        def recorded_inference(prompt):
            nonlocal cursor
            saved = self.trace[cursor]
            self.assertEqual(prompt, saved['prompt'])
            self.assertGreater(saved['generation_tokens'], 0)
            self.assertNotIn('evaluation_bank', prompt)
            self.assertNotIn('fixed_prediction', prompt)
            self.assertNotIn('rolling_prediction', prompt)
            self.assertNotIn('"actual"', prompt)
            cursor += 1
            return {k: saved[k] for k in ['raw_response', 'prompt_tokens', 'generation_tokens', 'finish_reason', 'seconds']}

        for packet in self.evidence['folds']:
            replayed.append(agent.decide_fold(packet, recorded_inference, []))
        self.assert_nested_close(replayed, self.decisions)
        self.assertEqual(cursor, 25)
        # A changed evaluator is a different world, but the inference DAG has
        # no argument or file read through which these labels can reach it.
        changed = copy.deepcopy(self.bank)
        changed['actual'] = [x + 100 for x in changed['actual']]
        rerun = agent.evaluate(changed, self.evidence, replayed)
        self.assertEqual(rerun['results'][0]['fixed_prediction'], self.result['results'][0]['fixed_prediction'])
        self.assertNotEqual(rerun['results'][0]['fixed_metrics'], self.result['results'][0]['fixed_metrics'])

    def test_predictions_metrics_future_and_ablation_reproduce(self):
        reproduced = agent.evaluate(self.bank, self.evidence, self.decisions)
        self.assert_nested_close(reproduced['results'], self.result['results'])
        self.assert_nested_close(reproduced['ranking'], self.result['ranking'])
        for result in self.result['results']:
            weights = result['weights_by_training_end_year']
            for w in weights:
                self.assertAlmostEqual(sum(w), 1)
                self.assertTrue(all(math.isfinite(v) and v >= 0 for v in w))
            for mode in ['fixed', 'rolling']:
                prediction = result[mode + '_prediction']
                bank = self.bank[mode + '_prediction']
                for i, value in enumerate(prediction):
                    w = weights[0 if mode == 'fixed' else i]
                    self.assertAlmostEqual(value, sum(w[j] * bank[i][j] for j in range(3)))
                    self.assertTrue(min(bank[i]) - 1e-9 <= value <= max(bank[i]) + 1e-9)
                absolute = [abs(p - y) for p, y in zip(prediction, self.bank['actual'])]
                expected = [100 * statistics.mean(e / y for e, y in zip(absolute, self.bank['actual'])),
                            statistics.mean(absolute), math.sqrt(statistics.mean(e * e for e in absolute)),
                            statistics.mean(e * e for e in absolute)]
                for x, y in zip(expected, result[mode + '_metrics']):
                    self.assertAlmostEqual(x, y, places=10)
            self.assertEqual(len(result['future_prediction']), 105)
            for future, row in zip(result['future_prediction'], self.bank['future']):
                self.assertAlmostEqual(future['prediction'], sum(w * p for w, p in zip(weights[-1], row['expert_prediction'])))

    def test_invalid_llm_output_does_not_silently_become_rule_prediction(self):
        bad = ['not json', '{}', '{"weights":{"pso":-1,"gpr":1,"ols":1},"reason":"bad"}',
               '{"weights":{"pso":0,"gpr":0,"ols":0},"reason":"bad"}',
               '{"weights":{"pso":true,"gpr":0,"ols":0},"reason":"bad"}',
               '{"weights":{"pso":1,"gpr":0},"reason":"bad"}']
        for text in bad:
            with self.subTest(text=text), self.assertRaises((ValueError, KeyError, TypeError)):
                agent.parse_decision(text)


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(MultiAgentTests)
    run = unittest.TextTestRunner(verbosity=2).run(suite)
    agent.write_json(METHOD / 'results/verification.json', {
        'passed': run.wasSuccessful(), 'tests_run': run.testsRun,
        'real_inference_trace_calls': len(load('trace.json')),
        'training_prefix_evidence_matches_saved_numeric_experiments': run.wasSuccessful(),
        'saved_llm_decisions_replayed': run.wasSuccessful(),
        'evaluation_labels_absent_from_prompts': run.wasSuccessful(),
        'four_strategies_predictions_and_metrics_recomputed': run.wasSuccessful(),
        'independent_holdout_available': False,
    })
    raise SystemExit(not run.wasSuccessful())
