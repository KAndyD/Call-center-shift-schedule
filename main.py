"""Расписание смен: MILP, LP-релаксация, сценарии, эксперименты масштаба."""
import copy
import csv
import json
import platform
from pathlib import Path
from time import perf_counter
import numpy as np
import scipy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from generate_data import generate
from greedy import schedule
from model import load_data, solve, status
from verify import check, conflict

ROOT = Path(__file__).parent
OUT = ROOT/'results'


def csv_write(name, columns, rows):
    with (OUT/name).open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(rows)


def run(data, relax=False, big_m=None, time_limit=12):
    start = perf_counter()
    result = solve(data, relax=relax, big_m=big_m, time_limit=time_limit)
    elapsed = perf_counter()-start
    response = {'status': status(result), 'message': result.message,
                'seconds': elapsed, 'objective': None, 'check': None}
    if not relax:
        response['mip_gap'] = float(getattr(result, 'mip_gap', float('nan')) or 0) if getattr(result, 'mip_gap', None) is not None else None
        response['mip_nodes'] = int(getattr(result, 'mip_node_count', 0) or 0)
        response['dual_bound'] = float(getattr(result, 'mip_dual_bound', float('nan'))) if getattr(result, 'mip_dual_bound', None) is not None else None
    if result.x is not None:
        audit = check(data, result.x, big_m=big_m, integer=not relax)
        response['objective'] = audit['objective']
        response['check'] = audit
        if not audit['ok']:
            raise RuntimeError('Решение не проходит независимую проверку')
        response['selected_operators'] = audit['working_operators']
    if result.x is None and response['status'] == 'infeasible':
        response['conflict'] = conflict(data)
    return response, result


def plot_schedule(data, result):
    n = len(data['staff'])
    assignments = result.x[:21*n].reshape(n, 7, 3)
    labels = np.full((n, 7), '-')
    for i in range(n):
        for d in range(7):
            for s in range(3):
                if assignments[i, d, s] > 0.5:
                    labels[i, d] = ('У', 'В', 'Н')[s]
    fig, ax = plt.subplots(figsize=(9, max(5, n*0.42)))
    picture = np.array([[0 if labels[i, d]=='-' else ('У','В','Н').index(labels[i, d])+1
                         for d in range(7)] for i in range(n)])
    ax.imshow(picture, cmap='Blues', vmin=0, vmax=3, aspect='auto')
    ax.set_xticks(range(7), data['days'])
    ax.set_yticks(range(n), [x['id'] for x in data['staff']])
    for i in range(n):
        for d in range(7):
            ax.text(d, i, labels[i, d], ha='center', va='center', fontsize=10)
    ax.set_title('Расписание смен: У — утро, В — вечер, Н — ночь, – — выходной')
    fig.tight_layout()
    fig.savefig(OUT/'schedule.png', dpi=170)
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    original = load_data()
    scenarios = {}
    runs = {}
    for title in ('Базовый', 'Дефицитный', 'Стрессовый'):
        data = copy.deepcopy(original)
        if title == 'Дефицитный':
            data['staff'] = data['staff'][:-1]
        if title == 'Стрессовый':
            data['coverage_per_day'] = [int(np.ceil(x*1.2)) for x in data['coverage_per_day']]
            data['staff'] = data['staff'][:-2]
        response, solver = run(data)
        scenarios[title] = response
        runs[title] = (data, solver)
    base = scenarios['Базовый']
    if base['status'] != 'optimal':
        raise RuntimeError('Не удалось найти доказанный MILP-оптимум')
    relaxation, relaxed_solver = run(original, relax=True)
    if relaxation['status'] != 'optimal':
        raise RuntimeError('LP-релаксация не решена')
    base['lp_relaxation_objective'] = relaxation['objective']
    base['integrality_gap_pct'] = 100*(base['objective']-relaxation['objective'])/base['objective']
    fractional = np.where((relaxed_solver.x > 1e-6) & (relaxed_solver.x < 1 - 1e-6))[0]
    base['fractional_variables_in_relaxation'] = int(len(fractional))
    base['fractional_example'] = {'index': int(fractional[0]), 'value': float(relaxed_solver.x[fractional[0]])} if len(fractional) else None
    greedy_x, greedy_sec = schedule(original)
    greedy = {'status': 'feasible' if greedy_x is not None else 'no_feasible_plan', 'seconds': greedy_sec}
    if greedy_x is not None:
        greedy_audit = check(original, greedy_x)
        greedy.update({'objective': greedy_audit['objective'], 'check': greedy_audit,
                       'gap_pct': 100*(greedy_audit['objective']-base['objective'])/base['objective']})
    # В обязательном эксперименте оценивается ослабление LP при слишком большом M.
    large_lp, _ = run(original, relax=True, big_m=500)
    large_m = {'M_tight': original['max_shifts_per_week'], 'M_excessive': 500,
               'relaxation_tight': relaxation['objective'],
               'relaxation_excessive': large_lp['objective'],
               'loss_of_lower_bound': relaxation['objective']-large_lp['objective']}
    scale_rows = []
    for size in (10, 20, 40):
        data = original if size == 10 else generate(size)
        if size == 10:
            exact = base
            lp = relaxation
        else:
            exact, _ = run(data, time_limit=12)
            lp, _ = run(data, relax=True)
        gap = None
        if exact['objective'] is not None and lp['objective'] is not None:
            gap = 100*(exact['objective']-lp['objective'])/exact['objective']
        scale_rows.append({'operators': size, 'status': exact['status'],
                           'seconds': exact['seconds'], 'mip_gap': exact.get('mip_gap'),
                           'lp_gap_pct': gap, 'objective': exact['objective'],
                           'lp_bound': lp['objective']})
    n = len(original['staff'])
    assignments = runs['Базовый'][1].x[:n*21].reshape(n, 7, 3)
    csv_write('schedule.csv', ['Оператор', 'День', 'Смена', 'Стоимость смены'],
              [[person['id'], original['days'][d], original['shifts'][s],
                original['hours_per_shift']*person['hourly_cost']+(person['night_extra'] if s == 2 else 0)]
               for i, person in enumerate(original['staff']) for d in range(7) for s in range(3)
               if assignments[i, d, s] > 0.5])
    csv_write('scenarios.csv', ['Сценарий', 'Статус', 'Стоимость', 'Работающие операторы'],
              [[name, r['status'], r['objective'], ', '.join(r.get('selected_operators', []))]
               for name, r in scenarios.items()])
    csv_write('scaling.csv', ['Операторов', 'Статус', 'Время (с)', 'MIP gap', 'Зазор LP (%)'],
              [[row['operators'], row['status'], row['seconds'], row['mip_gap'], row['lp_gap_pct']]
               for row in scale_rows])
    plot_schedule(original, runs['Базовый'][1])
    fig, ax = plt.subplots(figsize=(8, 4.7))
    sizes = [x['operators'] for x in scale_rows]
    ax.plot(sizes, [x['seconds'] for x in scale_rows], 'o-', label='Время MILP, с')
    ax.set_xlabel('Количество операторов')
    ax.set_ylabel('Время решения, с')
    other = ax.twinx()
    other.plot(sizes, [float('nan') if x['lp_gap_pct'] is None else x['lp_gap_pct'] for x in scale_rows], 's--', color='tab:orange', label='Зазор LP, %')
    other.set_ylabel('Интегральный зазор, %')
    ax.set_title('Масштаб: вычислительное время и LP–MILP зазор')
    fig.tight_layout()
    fig.savefig(OUT/'scaling.png', dpi=170)
    plt.close(fig)
    (OUT/'scenarios.json').write_text(json.dumps(scenarios, ensure_ascii=False, indent=2), encoding='utf-8')
    (OUT/'verification.json').write_text(json.dumps(base['check'], ensure_ascii=False, indent=2), encoding='utf-8')
    (OUT/'relaxation.json').write_text(json.dumps(relaxation, ensure_ascii=False, indent=2), encoding='utf-8')
    (OUT/'heuristic.json').write_text(json.dumps(greedy, ensure_ascii=False, indent=2), encoding='utf-8')
    (OUT/'big_m.json').write_text(json.dumps(large_m, ensure_ascii=False, indent=2), encoding='utf-8')
    (OUT/'scaling.json').write_text(json.dumps(scale_rows, ensure_ascii=False, indent=2), encoding='utf-8')
    (OUT/'solver.log').write_text('Python: '+platform.python_version()+'\nSciPy: '+scipy.__version__+'\n'
                                   'HiGHS MIP branch-and-bound; presolve=true; mip_rel_gap=1e-4; time_limit=12s\n'
                                   + '\n'.join(f'{name}: {s["message"]}; objective={s["objective"]}; '
                                               f'MIP gap={s.get("mip_gap")}; B&B nodes={s.get("mip_nodes")}'
                                               for name, s in scenarios.items())
                                   + '\nLP relaxation: ' + relaxation['message']+'\n', encoding='utf-8')
    report = ['# Отчёт: расписание операторов колл-центра', '',
              'Б24-507, № 14, вариант 5, seed 10014.', '',
              '## Постановка задачи', '',
              '10 операторов, 7 дней, 3 восьмичасовые смены каждый день.',
              'Переменная xᵢdₛ ∈ {0,1} означает назначение оператора i в день d на смену s.',
              'Переменная yᵢ ∈ {0,1} означает привлечение оператора на неделю (фиксированная оплата).',
              'Цель: минимальная сумма оплаты смен и затрат на привлечение операторов (условные рубли за неделю).',
              'Ограничения: покрытие каждой смены, не более 1 смены в день, не более 5 смен '
              'и 2 ночных смен в неделю, запрет утренней смены после ночной, '
              'личный выходной, связка Σd,s xᵢdₛ ≤ M yᵢ.',
              'При базовом лимите пяти смен M=5 — точная верхняя оценка количества недельных назначений.',
              'Всё выражено линейно; бинарность не заменяется простым непрерывным ЛП без изменения задачи.', '',
              '## LP-релаксация и MILP', '',
              f'MILP: статус {base["status"]}, стоимость {base["objective"]:.2f}, '
              f'сертифицированный MIP gap {base.get("mip_gap")}.',
              f'LP-релаксация: статус {relaxation["status"]}, нижняя оценка {relaxation["objective"]:.2f}.',
              f'Интегральный зазор: {base["integrality_gap_pct"]:.4f}% от MILP-стоимости.',
              f'Фактически в решении релаксации дробных переменных: {base["fractional_variables_in_relaxation"]}; '
              f'пример: индекс {base["fractional_example"]["index"]}, значение {base["fractional_example"]["value"]:.6f}.'
              if base['fractional_example'] else 'В данной релаксации все переменные получились целыми.',
              'В релаксации можно назначить дробную часть смены или частично активировать оператора — '
              'для реального расписания это недопустимо. MILP требует x и y из {0,1}.',
              f'Проверка NumPy: {base["check"]["ok"]}, максимальное отклонение '
              f'{max(base["check"]["violations"].values()):.3g}.', '',
              '## Сценарии', '', '| Сценарий | Статус | Стоимость | Активные операторы |',
              '|---|---|---:|---|']
    for name, r in scenarios.items():
        report.append(f'| {name} | {r["status"]} | {r["objective"] if r["objective"] is not None else "—"} | '
                      f'{", ".join(r.get("selected_operators", [])) or "—"} |')
    report += ['', 'Дефицит: один оператор отсутствует всю неделю (9 вместо 10).',
               'Стресс: потребность каждой смены повышена на 20% с округлением вверх; '
               'два сотрудника недоступны. Из-за округления 2 → 3 фактический прирост по каждой смене составляет 50%.',
               'Это сценарий проверки совместимости.',
               f'Стресс требует {conflict(runs["Стрессовый"][0])["required_shifts"]} назначений; '
               f'операторы могут выполнить не более {conflict(runs["Стрессовый"][0])["aggregate_upper_limit"]}.', '',
               '## Жадное сравнение', '',
               f'Эвристика: {greedy["status"]}, время {greedy["seconds"]:.6f} с.']
    if greedy.get('objective') is not None:
        report.append(f'Стоимость {greedy["objective"]:.2f}; отставание от MILP {greedy["gap_pct"]:.4f}%.')
    else:
        report.append('Допустимый жадный график не найден — сравнивать его цену с MILP нельзя.')
    report += ['', '## Большая константа big-M', '',
               f'Точное M={large_m["M_tight"]}: LP нижняя оценка {large_m["relaxation_tight"]:.2f}. '
               f'Завышенное M={large_m["M_excessive"]}: оценка {large_m["relaxation_excessive"]:.2f}.',
               f'Потеря силы релаксации: {large_m["loss_of_lower_bound"]:.2f} условных рублей.',
               'Завышенное M позволяет в релаксации активировать сотрудника на доли процента '
               'и при этом полностью отработать доступные смены. Для целочисленного решения смысл модели не меняется, '
               'но численная нижняя граница слабеет.', '',
               '## Рост масштаба', '',
               '| Операторов | Время (с) | Статус | Зазор LP–MILP |', '|---:|---:|---|---:|']
    report += [f'| {r["operators"]} | {r["seconds"]:.4f} | {r["status"]} | '
               f'{r["lp_gap_pct"] if r["lp_gap_pct"] is not None else "—"} |' for r in scale_rows]
    report += ['', 'Время зависит от условий запуска и не сравнивается с другим компьютером напрямую. '
               'При time limit найденный допустимый план не обязательно доказан оптимальным: '
               'в этом случае интерпретируют MIP gap и двойственную границу.', '',
               '## Вывод', '',
               'Расписание пригодно как учебный недельный план при принятых ограничениях. '
               'Для рабочего применения понадобятся требования законодательства о труде, '
               'реальные перерывы, отпуска, квалификации и прогноз потока обращений. '
               'Решатель — HiGHS через scipy.optimize.milp; метод ветвей и границ, presolve включён.']
    (ROOT/'REPORT.md').write_text('\n'.join(report)+'\n', encoding='utf-8')
    print('ЛР3:',base['status'],base['objective'],'LP:',relaxation['objective'],
          'greedy:',greedy['status'], 'scale:', [(x['operators'],x['status']) for x in scale_rows])


if __name__ == '__main__':
    main()
