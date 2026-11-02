"""MILP расписания с фиксированными затратами на привлечение оператора."""
import json
from pathlib import Path
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
import numpy as np


def load_data(path=None):
    path = Path(path) if path else Path(__file__).parent/'data'/'shifts.json'
    return json.loads(path.read_text(encoding='utf-8'))


def x_index(i, d, s):
    return i*21 + d*3 + s


def y_index(i, count):
    return count*21+i


def build(data, big_m=None):
    staff = data['staff']
    count = len(staff)
    total = 22*count
    shift_limit = data['max_shifts_per_week']
    m = float(shift_limit if big_m is None else big_m)
    if m < shift_limit:
        raise ValueError('big-M не может быть меньше допустимого недельного объёма')
    costs = np.zeros(total)
    upper = np.ones(total)
    for i, person in enumerate(staff):
        for day in range(7):
            for s in range(3):
                idx = x_index(i, day, s)
                costs[idx] = person['hourly_cost'] * data['hours_per_shift'] + (person['night_extra'] if s == 2 else 0)
                if person['day_off'] == day:
                    upper[idx] = 0
        costs[y_index(i, count)] = person['activation_cost']
    row_id, col_id, values, rhs = [], [], [], []

    def append(coeffs, bound):
        r = len(rhs)
        for index, coef in coeffs:
            row_id.append(r)
            col_id.append(index)
            values.append(float(coef))
        rhs.append(float(bound))

    # Достаточное количество операторов на каждой смене.
    for d in range(7):
        for s in range(3):
            append([(x_index(i, d, s), -1) for i in range(count)], -data['coverage_per_day'][s])
    for i in range(count):
        for d in range(7):
            append([(x_index(i, d, s), 1) for s in range(3)], 1)
        append([(x_index(i, d, s), 1) for d in range(7) for s in range(3)], shift_limit)
        append([(x_index(i, d, 2), 1) for d in range(7)], data['max_nights_per_week'])
        for d in range(6):
            append([(x_index(i, d, 2), 1), (x_index(i, d+1, 0), 1)], 1)
        # Сумма отработанных смен отключается бинарной переменной y_i.
        append([(x_index(i, d, s), 1) for d in range(7) for s in range(3)] +
               [(y_index(i, count), -m)], 0)
    a = sparse.coo_matrix((values, (row_id, col_id)), shape=(len(rhs), total)).tocsr()
    return costs, a, np.asarray(rhs), upper


def solve(data, relax=False, big_m=None, time_limit=12):
    costs, a, upper_rhs, variable_upper = build(data, big_m)
    if relax:
        return linprog(costs, A_ub=a, b_ub=upper_rhs,
                       bounds=list(zip(np.zeros_like(variable_upper), variable_upper)),
                       method='highs-ds', options={'presolve': True})
    return milp(costs, integrality=np.ones(len(costs)),
                bounds=Bounds(np.zeros_like(variable_upper), variable_upper),
                constraints=LinearConstraint(a, -np.inf, upper_rhs),
                options={'presolve': True, 'mip_rel_gap': 0.0001, 'time_limit': time_limit})


def status(result):
    return {0: 'optimal', 1: 'time_or_iteration_limit', 2: 'infeasible',
            3: 'unbounded', 4: 'numerical_failure'}.get(result.status, 'unknown')
