# ЛР 3. Расписание смен колл-центра (MILP)

Группа Б24-507 · студент № 14 · вариант 5 · seed 10014.

## Windows 11

```powershell
python -m pip install -r requirements.txt
python main.py
python -m unittest discover -s tests -v
```

Можно запускать `main.py` кнопкой Run. Требования к ресурсам: несколько минут и несколько сотен МБ памяти на максимальном эксперименте; для задач 20/40 операторов установлен лимит времени MILP 12 секунд **на каждый запуск**. Если HiGHS не доказал оптимум к этому времени, будет записан статус `time_or_iteration_limit` и фактический MIP gap. Это не ошибка установки.

`python generate_data.py` воспроизводимо создаёт `data/shifts.json`. Для первого запуска делать это не требуется.

## Состав

- `model.py`: сборка MILP и LP-релаксации, HiGHS; `verify.py`: независимый аудит всех ограничений.
- `greedy.py`: упрощённая эвристика без солвера; `main.py`: сценарии, исследование big-M и масштаба.
- `results/schedule.csv`: базовый план; `schedule.png`: визуализация расписания.
- `results/scenarios.csv`: базовый, дефицитный, стрессовый сценарии.
- `results/scaling.csv`, `scaling.png`: время и зазор для 10, 20, 40 операторов.
- `results/verification.json`, `relaxation.json`, `big_m.json`, `heuristic.json`, `solver.log`: результаты проверок.
- `REPORT.md`: численные выводы; `THEORY.md`: теория и пояснение к защите.

`optimal` — глобальный оптимум MILP подтверждён с заданным допуском; `infeasible` — конфликт ограничений; `time_or_iteration_limit` — нет доказательства оптимальности, даже если имеется допустимый график.
