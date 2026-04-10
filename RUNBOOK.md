# RUNBOOK — збірка, запуск і діагностика

## 1. Передумови

- Ubuntu Linux.
- Gazebo Harmonic CLI (`gz ...`).
- PX4 репозиторій: `~/PX4-Autopilot` (або змінна `PX4_REPO`).
- Docker image: `px4-ubuntu24-sim`.
- Python-оточення з залежностями (рекомендовано: `.venv_gz`).

## 2. Швидкий e2e запуск (рекомендовано)

```bash
cd /home/sanek/raptor_ai
export GZ_PARTITION=raptor_px4
./run_px4_autostart.sh \
  --python /home/sanek/raptor_ai/.venv_gz/bin/python \
  --follow-profile balanced \
  --px4-gpu on \
  --viz --record
```

Артефакти запуску зберігаються в `runs/autostart/<timestamp>/`.

## 3. Альтернативні режими запуску

### Runtime без orchestration

```bash
cd /home/sanek/raptor_ai
PYTHONNOUSERSITE=1 /usr/bin/python3 scripts/run_runtime_v2.py \
  --config raptor_ai/config/default_config.yaml \
  --viz
```

### Лише вікно камери

```bash
cd /home/sanek/raptor_ai
PYTHONNOUSERSITE=1 /usr/bin/python3 scripts/camera_viewer.py --platform-type px4
```

### Тести

```bash
cd /home/sanek/raptor_ai
PYTHONNOUSERSITE=1 /usr/bin/python3 -m pytest tests -q
```

## 4. Важливі конфігураційні точки

Файл: `raptor_ai/config/default_config.yaml`.

Ключові блоки:

- `camera` — маршрутизація camera topic + frame contract.
- `platform` — параметри платформи (`gimbal`, `iris`, `px4`).
- `detector` — `red` / `yolo_onnx` + providers.
- `selector` — інтеграція PTS (`external`) або fallback.
- `guidance` — зовнішній модуль `target-guidance`.
- `follow` — профіль і тюнінг поведінки супроводу.

## 5. CUDA / ONNX Runtime

### Базова перевірка providers

```bash
/home/sanek/raptor_ai/.venv_gz/bin/python - <<'PY'
import onnxruntime as ort
print(ort.__version__)
print(ort.get_available_providers())
PY
```

### Примітка для GTX 10xx (Pascal, напр. GTX 1080)

Для Pascal-карт стабільний стек:

- `onnxruntime-gpu==1.18.1`
- `nvidia-cudnn-cu11==8.9.6.50`
- `nvidia-cublas-cu11==11.11.3.6`
- `nvidia-cuda-nvrtc-cu11==11.8.89`
- `nvidia-cuda-runtime-cu11==11.8.89`
- `nvidia-cufft-cu11==10.9.0.58`
- `nvidia-curand-cu11==10.3.0.86`

Приклад встановлення:

```bash
/home/sanek/raptor_ai/.venv_gz/bin/python -m pip install --force-reinstall \
  "onnxruntime-gpu==1.18.1" \
  "nvidia-cudnn-cu11==8.9.6.50" \
  "nvidia-cublas-cu11==11.11.3.6" \
  "nvidia-cuda-nvrtc-cu11==11.8.89" \
  "nvidia-cuda-runtime-cu11==11.8.89" \
  "nvidia-cufft-cu11==10.9.0.58" \
  "nvidia-curand-cu11==10.3.0.86"
```

## 6. Типові проблеми та швидкі рішення

### 6.1 `No usable temporary directory found`

Перевір права й наявність `/tmp`:

```bash
ls -ld /tmp
```

Нормальний режим для `/tmp`: `drwxrwxrwt` (тобто `1777`).

### 6.2 `mavsdk_server` займає порт 14540

```bash
pgrep -af mavsdk_server
```

Якщо процес сторонній (не ваш запуск), його треба зупинити, інакше PX4 bridge буде підключатися нестабільно.

### 6.3 Actor не спавниться з першої спроби

У `run_px4_autostart.sh` вже є retry-механізм actor setup. Якщо потрібно жорстко фейлити запуск при проблемі спавну, використовуйте:

```bash
./run_px4_autostart.sh --strict-actor-setup ...
```

### 6.4 Protobuf-попередження `File already exists in database`

Для поточного пайплайну ці повідомлення часто шумові. Критично, коли падає сам процес або немає кадрів/telemetry.

## 7. Корисні змінні середовища

- `GZ_PARTITION` — ізоляція Gazebo topic namespace.
- `PX4_REPO` — шлях до PX4-Autopilot.
- `RAPTOR_PYTHON` — інтерпретатор Python для host-скриптів.

