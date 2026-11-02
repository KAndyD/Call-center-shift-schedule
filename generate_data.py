"""Состав операторов и требования к дежурствам, вариант 5."""
import json
from pathlib import Path
import numpy as np

SEED = 10014
PATH = Path(__file__).parent / 'data' / 'shifts.json'


def generate(size=10):
    if size < 5:
        raise ValueError('Требуется не менее пяти операторов')
    rng = np.random.default_rng(SEED)
    staff = []
    for i in range(size):
        staff.append({'id': f'О{i+1:02d}', 'hourly_cost': int(rng.integers(98, 155)),
                      'activation_cost': int(rng.integers(220, 420)),
                      'day_off': (i * 3 + 1) % 7,
                      'night_extra': int(rng.integers(15, 46))})
    scale = size / 10
    coverage = [max(1, int(round(b * scale))) for b in (2, 2, 2)]
    return {'variant': 5, 'seed': SEED, 'group': 'Б24-507', 'student_number': 14,
            'days': ['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс'],
            'shifts': ['Утро', 'Вечер', 'Ночь'], 'hours_per_shift': 8,
            'coverage_per_day': coverage, 'max_shifts_per_week': 5,
            'max_nights_per_week': 2, 'staff': staff}


if __name__ == '__main__':
    PATH.write_text(json.dumps(generate(), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(PATH)
