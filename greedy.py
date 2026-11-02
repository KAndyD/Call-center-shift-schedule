"""Детерминированная эвристика: сначала ночи, затем остальные смены."""
from time import perf_counter
import numpy as np
from verify import check


def schedule(data):
    start = perf_counter()
    staff = data['staff']
    n = len(staff)
    best = None
    best_cost = float('inf')
    # Разные порядки заполнения: цель — найти допустимый вариант без MILP.
    shift_orders = [(2, 1, 0), (0, 2, 1), (2, 0, 1), (1, 2, 0)]
    for rotation in range(8):
        for order in shift_orders:
            x = np.zeros((n, 7, 3))
            working = np.zeros(n, dtype=bool)
            assignments = np.zeros(n)
            nights = np.zeros(n)
            valid = True
            days = list(range(7))
            days = days[rotation % 7:] + days[:rotation % 7]
            for s in order:
                for d in days:
                    for _ in range(data['coverage_per_day'][s]):
                        candidates = []
                        for i, person in enumerate(staff):
                            if person['day_off'] == d or x[i, d, :].sum() or assignments[i] >= data['max_shifts_per_week']:
                                continue
                            if s == 2 and nights[i] >= data['max_nights_per_week']:
                                continue
                            if s == 0 and d > 0 and x[i, d-1, 2] > 0:
                                continue
                            if s == 2 and d < 6 and x[i, d+1, 0] > 0:
                                continue
                            added_cost = person['hourly_cost']*data['hours_per_shift'] + (person['night_extra'] if s == 2 else 0)
                            if not working[i]:
                                added_cost += person['activation_cost'] * (0.65 + rotation*0.03)
                            # Сберегаем небольшую часть доступности для дальнейших дней.
                            score = added_cost + 2*assignments[i] + ((i+rotation) % n)*0.001
                            candidates.append((score, i))
                        if not candidates:
                            valid = False
                            break
                        _, i = min(candidates)
                        x[i, d, s] = 1
                        working[i] = True
                        assignments[i] += 1
                        nights[i] += s == 2
                    if not valid:
                        break
                if not valid:
                    break
            if valid:
                vector = np.r_[x.ravel(), working.astype(float)]
                audit = check(data, vector)
                if audit['ok'] and audit['objective'] < best_cost:
                    best = vector
                    best_cost = audit['objective']
    return best, perf_counter()-start
