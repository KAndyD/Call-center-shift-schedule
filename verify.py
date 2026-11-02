"""Поштучная независимая проверка смен, отдыха, доступности и big-M."""
import numpy as np

TOL = 1e-7


def check(data, solution, big_m=None, integer=True):
    staff = data['staff']
    n = len(staff)
    v = np.asarray(solution, dtype=float)
    if v.size != 22*n or not np.all(np.isfinite(v)):
        raise ValueError('Некорректный вектор решений')
    x = v[:n*21].reshape(n, 7, 3)
    y = v[n*21:]
    max_shifts = data['max_shifts_per_week']
    m = max_shifts if big_m is None else big_m
    breaches = {'binary': float(np.max(np.abs(v - np.rint(v)))) if integer else 0.0,
                'bounds': float(max(0, -v.min(), v.max()-1)),
                'coverage': float(max(0, np.max(np.array(data['coverage_per_day'])[None, :] - x.sum(axis=0)))),
                'daily_limit': float(max(0, (x.sum(axis=2)-1).max())),
                'weekly_limit': float(max(0, (x.sum(axis=(1, 2))-max_shifts).max())),
                'night_limit': float(max(0, (x[:, :, 2].sum(axis=1)-data['max_nights_per_week']).max())),
                'night_rest': float(max(0, (x[:, :-1, 2]+x[:, 1:, 0]-1).max())),
                'activation_big_m': float(max(0, (x.sum(axis=(1, 2))-m*y).max()))}
    breaches['unavailability'] = float(max(x[i, person['day_off'], :].max() for i, person in enumerate(staff)))
    cost = 0.0
    for i, person in enumerate(staff):
        cost += person['activation_cost']*y[i]
        cost += data['hours_per_shift']*person['hourly_cost']*x[i].sum()
        cost += person['night_extra']*x[i, :, 2].sum()
    return {'ok': all(val <= TOL for val in breaches.values()),
            'violations': breaches, 'objective': float(cost),
            'working_operators': [staff[i]['id'] for i in range(n) if y[i] > 0.5],
            'covered': x.sum(axis=0).tolist(), 'total_assignments': float(x.sum())}


def conflict(data):
    demand = 7 * sum(data['coverage_per_day'])
    maximum = len(data['staff']) * data['max_shifts_per_week']
    return {'required_shifts': demand, 'aggregate_upper_limit': maximum,
            'proven_infeasible_by_capacity': demand > maximum}
