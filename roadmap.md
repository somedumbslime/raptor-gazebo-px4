# RAPTOR AI — Дорожня карта

Версія: 2026-04-10
Поточний статус: **Фази 0-4 завершено, активна Фаза 5**.

## 1. Мета проєкту

Побудувати модульну систему супроводу цілі в Gazebo/PX4, яка проходить шлях:

`synthetic target + gimbal -> actor target + YOLO/PTS -> PX4/MAVLink follow (XY -> XYZ)`.

Кінцева ціль Фази 5:

`зліт із землі -> пошук -> захоплення цілі -> стабільний follow -> посадка`.

## 2. Поточний baseline (що вже є)

- Runtime V2 з модульним конвеєром:
  `camera -> detector -> tracker -> primary selector -> guidance -> platform -> metrics/events`.
- Перемикання detector: `red` / `yolo_onnx`.
- Інтеграція зовнішнього PTS через адаптер (`selector.backend=external`) + fallback.
- Інтеграція зовнішнього guidance-модуля `target-guidance`.
- Платформний шар: `gimbal`, `iris`, `px4`.
- Сценарний runner + артефакти запусків у `runs/...`.
- Набір unit/integration тестів у `tests/`.

## 3. Статус фаз

- Фаза 0 — **Done** (baseline стабілізовано, артефакти зафіксовано).
- Фаза 1 — **Done** (PTS інтегровано через external adapter).
- Фаза 2 — **Done** (actor-сцени додано й верифіковано).
- Фаза 3 — **Done** (YOLO ONNX + tracking + PTS e2e).
- Фаза 4 — **Done** (платформний перехід на дрон, abstraction завершено).
- Фаза 5 — **In Progress** (PX4/MAVLink follow behavior).
- Фаза 6 — **Planned** (hardening і регресійні гейти).
- Фаза 7+ — **Planned** (масштабування сценаріїв та можливий hardware/ROS2 напрям).

---

## 4. Фаза 5 (активна): PX4/MAVLink Follow Behavior

### 5.1 PX4 SITL baseline і базові команди польоту — **Done**

Результат:
- Є smoke-контур `connect -> arm -> takeoff -> offboard -> land`.
- Базові параметри PX4 винесено в конфіг.

### 5.2 MAVLink bridge (`px4_bridge`) — **Done**

Результат:
- Реалізовано канал телеметрії/команд.
- API команд і watchdog/failsafe працюють через платформний адаптер.

### 5.3 Камера дрона і live-view — **Done**

Результат:
- Стабільний camera stream + smoke-перевірка контракту (`fps`, `size`, `gap`).
- Topic для PX4-сцени зафіксовано (`/front_camera/image_raw`).

### 5.4 Actor у світі PX4 у полі зору камери — **Done (з operational caveats)**

Результат:
- Actor спавниться/репозиціонується відносно `x500_0`.
- Є retry-механізм, але інколи потрібні повторні спроби (нестабільність Gazebo/Fuel).

### 5.5 YOLO/ONNX + PTS на камері дрона (e2e CV) — **Done (CPU/CUDA fallback path)**

Результат:
- E2E контур детекту/трекінгу/вибору primary працює в PX4-сцені.
- Для Pascal GPU підтверджено робочий шлях через сумісний стек ORT/CUDA.

### 5.6 Follow mode v1 (TRACKING_XY) — **In Progress**

Що вже зроблено:
- Guidance переведено на зовнішній модуль `target-guidance`.
- Єдина стратегія follow: `zone_track`.
- Профілі тюнінгу: `safe / balanced / aggressive`.
- Контракт guidance стабільний: на вході трек/помилки, на виході платформа-команди.

Що залишилось довести:
- Менше осциляцій під час підльоту.
- Стабільніше утримання висоти під час активного зближення.
- Передбачуваність поведінки при втраті/повторному захопленні цілі.

### 5.7 Політна state machine (повний автоконтур) — **Planned**

Станова модель:
- `DISARMED -> TAKEOFF -> SEARCHING -> TRACKING_XY -> (опц. TRACKING_XYZ) -> LANDING`.

Потрібно:
- Формалізувати переходи за подіями та таймаутами.
- Логувати переходи окремими event-сигналами.

### 5.8 Search behavior у польоті — **Planned**

Потрібно:
- Безпечний airborne-search (yaw sweep / bounded pattern).
- Межі швидкості/висоти + timeout search.
- Повернення в `LANDING`, якщо ціль довго не знайдена.

### 5.9 Follow XY -> XYZ + closure Фази 5 — **Planned**

Потрібно:
- Дотюнити XY follow за метриками стабільності.
- Додати/закріпити контроль висоти для XYZ.
- Підготувати `phase5_closure_report.md` з DoD-критеріями.

---

## 5. Найближчі 2-4 ітерації

1. Стабілізувати `zone_track + balanced` під реальні FPS/latency умови.
2. Формалізувати state machine переходи й перевести автоконтур у повністю детермінований режим.
3. Доробити search behavior як окрему policy-підсистему.
4. Підготувати acceptance-сценарії та закрити Фазу 5 через closure-report.

## 6. Ризики та контроль

- Ризик: просідання FPS/латентність рве контур follow.
  - Дія: профілювання, профілі тюнінгу, контроль частоти інференсу/команд.

- Ризик: нестабільність actor spawn у Gazebo/Fuel.
  - Дія: retry + strict-режим для CI/приймальних прогонів.

- Ризик: неузгодженість логіки між guidance і платформою.
  - Дія: жорсткий I/O контракт + тести інтерфейсного рівня.

- Ризик: небезпечні команди під час втрати треку.
  - Дія: safety limits, watchdog, передбачувані переходи в `SEARCHING/LANDING`.

## 7. Правило оновлення roadmap

- Спочатку оновлюємо roadmap, потім реалізуємо фічу.
- Після закриття кожного великого кроку — фіксуємо статус + артефакти.
- У roadmap тримаємо тільки актуальні кроки (legacy-ланцюги прибираємо).
