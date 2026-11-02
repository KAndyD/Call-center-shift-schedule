import sys
import copy
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generate_data import generate
from model import load_data, solve, status, build
from verify import check, conflict
from greedy import schedule


class ScheduleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_data()
        cls.result = solve(cls.data)

    def test_reproducible(self):
        self.assertEqual(self.data, generate())
        self.assertEqual(len(self.data['staff']), 10)

    def test_integer_feasibility(self):
        self.assertEqual(status(self.result), 'optimal')
        audit = check(self.data, self.result.x)
        self.assertTrue(audit['ok'])
        self.assertAlmostEqual(audit['objective'], self.result.fun, places=5)

    def test_relaxation_lower_bound(self):
        relaxed = solve(self.data, relax=True)
        self.assertTrue(relaxed.success)
        self.assertLessEqual(relaxed.fun, self.result.fun + 1e-5)

    def test_bad_schedule_rejected(self):
        self.assertFalse(check(self.data, np.zeros_like(self.result.x))['ok'])

    def test_oversized_big_m_weakens_bound(self):
        tight = solve(self.data, relax=True)
        huge = solve(self.data, relax=True, big_m=500)
        self.assertLessEqual(huge.fun, tight.fun + 1e-6)

    def test_stress_conflict(self):
        d = generate()
        d['coverage_per_day'] = [3, 3, 3]
        d['staff'] = d['staff'][:-2]
        self.assertTrue(conflict(d)['proven_infeasible_by_capacity'])
        self.assertEqual(status(solve(d)), 'infeasible')

    def test_missing_operator_scenario(self):
        data = copy.deepcopy(self.data)
        data['staff'] = data['staff'][:-1]
        res = solve(data)
        self.assertEqual(status(res), 'optimal')
        self.assertTrue(check(data, res.x)['ok'])
        self.assertGreaterEqual(res.fun + 1e-6, self.result.fun)

    def test_greedy_if_found(self):
        heuristic, _ = schedule(self.data)
        if heuristic is not None:
            self.assertTrue(check(self.data, heuristic)['ok'])
            self.assertGreaterEqual(check(self.data, heuristic)['objective'] + 1e-5, self.result.fun)


if __name__ == '__main__':
    unittest.main()
