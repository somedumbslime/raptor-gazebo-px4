#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

CONFIG_PATH="raptor_ai/config/default_config.yaml"
SCENARIO_NAME="slow_circle"
DURATION_S="120"
TAKEOFF_ALT_M="2.0"
RUNS_ROOT=""
ENABLE_VIZ=1
ENABLE_RECORD=0
ENABLE_CAMERA_VIEWER=0
FOLLOW_MODE="xy"
FOLLOW_ENABLED=1
FOLLOW_XY_STRATEGY="zone_track"
FOLLOW_PROFILE="balanced"
PX4_TAKEOFF_CONFIRM_ALT=""
PX4_OFFBOARD_MIN_ALT=""
PX4_OFFBOARD_DELAY_AFTER_LIFTOFF=""
PX4_AUTO_ARM_REQUIRE_ARMABLE=""
PX4_AUTO_ARM_REQUIRE_LOCAL_POSITION=""
STATE_LOST_FRAME_THRESHOLD=""
STATE_REACQUIRE_THRESHOLD=""
SKIP_PX4=0
PX4_HEADLESS=0
PX4_GPU_MODE="off"
STRICT_ACTOR_SETUP=0
ACTOR_SETUP_RETRIES=3

PX4_REPO="${PX4_REPO:-$HOME/PX4-Autopilot}"
PX4_IMAGE="${PX4_IMAGE:-px4-ubuntu24-sim}"
GZ_PARTITION="${GZ_PARTITION:-raptor_px4}"
PYTHON_BIN="${RAPTOR_PYTHON:-}"

if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x "$ROOT_DIR/.venv_gz/bin/python" ]]; then
    PYTHON_BIN="$ROOT_DIR/.venv_gz/bin/python"
  else
    PYTHON_BIN="/usr/bin/python3"
  fi
fi

usage() {
  cat <<'EOF'
Usage: ./run_px4_autostart.sh [options]

Options:
  --config PATH              Runtime config (default: raptor_ai/config/default_config.yaml)
  --scenario NAME            Scenario profile (default: slow_circle)
  --duration SEC             Scenario duration (default: 120)
  --takeoff-alt METERS       PX4 auto-takeoff altitude (default: 2.0)
  --runs-root DIR            Output root (default: runs/autostart/<timestamp>)
  --python PATH              Python interpreter for host scripts
  --follow-mode MODE         off|xy (default: xy)
  --follow-enabled           Enable follow controller (default)
  --no-follow-enabled        Disable follow controller
  --follow-profile NAME      safe|balanced|aggressive preset for follow tuning (default: balanced)
  --follow-xy-strategy MODE  zone_track (default: zone_track)
  --follow-zone-track        Shortcut: --follow-mode xy --follow-enabled --follow-xy-strategy zone_track
  --follow-safe              Shortcut: --follow-profile safe
  --follow-balanced          Shortcut: --follow-profile balanced
  --follow-aggressive        Shortcut: --follow-profile aggressive
  --px4-takeoff-confirm-alt M
                             Override platform.px4.takeoff_confirm_alt_m
  --px4-offboard-min-alt M   Override platform.px4.offboard_min_relative_alt_m
  --px4-offboard-delay-after-liftoff SEC
                             Override platform.px4.offboard_start_delay_after_liftoff_s
  --px4-auto-arm-require-armable / --no-px4-auto-arm-require-armable
                             Override platform.px4.auto_arm_require_armable
  --px4-auto-arm-require-local-position / --no-px4-auto-arm-require-local-position
                             Override platform.px4.auto_arm_require_local_position
  --state-lost-frame-threshold N
                             Override state_machine.lost_frame_threshold
  --state-reacquire-threshold N
                             Override state_machine.reacquire_threshold
  --viz                      Enable runtime OpenCV window (default)
  --no-viz                   Disable runtime OpenCV window
  --record                   Enable runtime video recording
  --camera-viewer            Open separate camera viewer window in parallel
  --skip-px4                 Do not start PX4 Docker (assume already running)
  --px4-gui                  Run PX4 Gazebo with gz gui (default)
  --px4-headless             Run PX4 Gazebo without gz gui (fallback if GUI crashes)
  --px4-gpu MODE             Docker GPU mode: off|auto|on (default: off)
                             auto = try GPU, fallback to CPU if unavailable
  --strict-actor-setup       Fail autostart if actor setup fails after retries
  --actor-setup-retries N    Retries for actor setup (default: 3)
  -h, --help                 Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG_PATH="$2"; shift 2 ;;
    --scenario) SCENARIO_NAME="$2"; shift 2 ;;
    --duration) DURATION_S="$2"; shift 2 ;;
    --takeoff-alt) TAKEOFF_ALT_M="$2"; shift 2 ;;
    --runs-root) RUNS_ROOT="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --follow-mode) FOLLOW_MODE="$2"; shift 2 ;;
    --follow-enabled) FOLLOW_ENABLED=1; shift ;;
    --no-follow-enabled) FOLLOW_ENABLED=0; shift ;;
    --follow-profile) FOLLOW_PROFILE="$2"; shift 2 ;;
    --follow-xy-strategy) FOLLOW_XY_STRATEGY="$2"; shift 2 ;;
    --follow-zone-track) FOLLOW_MODE="xy"; FOLLOW_ENABLED=1; FOLLOW_XY_STRATEGY="zone_track"; shift ;;
    --follow-safe) FOLLOW_PROFILE="safe"; shift ;;
    --follow-balanced) FOLLOW_PROFILE="balanced"; shift ;;
    --follow-aggressive) FOLLOW_PROFILE="aggressive"; shift ;;
    --px4-takeoff-confirm-alt) PX4_TAKEOFF_CONFIRM_ALT="$2"; shift 2 ;;
    --px4-offboard-min-alt) PX4_OFFBOARD_MIN_ALT="$2"; shift 2 ;;
    --px4-offboard-delay-after-liftoff) PX4_OFFBOARD_DELAY_AFTER_LIFTOFF="$2"; shift 2 ;;
    --px4-auto-arm-require-armable) PX4_AUTO_ARM_REQUIRE_ARMABLE="true"; shift ;;
    --no-px4-auto-arm-require-armable) PX4_AUTO_ARM_REQUIRE_ARMABLE="false"; shift ;;
    --px4-auto-arm-require-local-position) PX4_AUTO_ARM_REQUIRE_LOCAL_POSITION="true"; shift ;;
    --no-px4-auto-arm-require-local-position) PX4_AUTO_ARM_REQUIRE_LOCAL_POSITION="false"; shift ;;
    --state-lost-frame-threshold) STATE_LOST_FRAME_THRESHOLD="$2"; shift 2 ;;
    --state-reacquire-threshold) STATE_REACQUIRE_THRESHOLD="$2"; shift 2 ;;
    --viz) ENABLE_VIZ=1; shift ;;
    --no-viz) ENABLE_VIZ=0; shift ;;
    --record) ENABLE_RECORD=1; shift ;;
    --camera-viewer) ENABLE_CAMERA_VIEWER=1; shift ;;
    --skip-px4) SKIP_PX4=1; shift ;;
    --px4-headless) PX4_HEADLESS=1; shift ;;
    --px4-gui) PX4_HEADLESS=0; shift ;;
    --px4-gpu) PX4_GPU_MODE="$2"; shift 2 ;;
    --strict-actor-setup) STRICT_ACTOR_SETUP=1; shift ;;
    --actor-setup-retries) ACTOR_SETUP_RETRIES="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "[AUTOSTART] unknown option: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "${DISPLAY:-}" ]]; then
  echo "[AUTOSTART] DISPLAY is not set. GUI windows (Gazebo/OpenCV) will not open." >&2
  exit 2
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "[AUTOSTART] config not found: $CONFIG_PATH" >&2
  exit 2
fi

if [[ "$FOLLOW_MODE" != "off" && "$FOLLOW_MODE" != "xy" ]]; then
  echo "[AUTOSTART] unsupported --follow-mode: $FOLLOW_MODE (supported: off|xy)" >&2
  exit 2
fi

if [[ "$FOLLOW_XY_STRATEGY" != "zone_track" ]]; then
  echo "[AUTOSTART] unsupported --follow-xy-strategy: $FOLLOW_XY_STRATEGY (supported: zone_track)" >&2
  exit 2
fi

PX4_GPU_MODE="$(echo "$PX4_GPU_MODE" | tr '[:upper:]' '[:lower:]')"
if [[ "$PX4_GPU_MODE" != "off" && "$PX4_GPU_MODE" != "auto" && "$PX4_GPU_MODE" != "on" ]]; then
  echo "[AUTOSTART] unsupported --px4-gpu: $PX4_GPU_MODE (supported: off|auto|on)" >&2
  exit 2
fi

if [[ ! -d "$PX4_REPO" ]]; then
  echo "[AUTOSTART] PX4 repo not found: $PX4_REPO" >&2
  exit 2
fi

if [[ -z "$RUNS_ROOT" ]]; then
  TS="$(date +%Y%m%d_%H%M%S)"
  RUNS_ROOT="runs/autostart/$TS"
fi
mkdir -p "$RUNS_ROOT"

PX4_LOG="$RUNS_ROOT/px4_sitl.log"
CAM_SMOKE_JSON="$RUNS_ROOT/px4_camera_smoke.json"
ACTOR_SETUP_JSON="$RUNS_ROOT/px4_actor_setup.json"

PX4_CONTAINER_NAME="raptor_px4_autostart_$(date +%s)"
PX4_DOCKER_PID=""
CAM_VIEWER_PID=""
XHOST_OPENED=0

cleanup() {
  local ec=$?
  set +e
  if [[ -n "$CAM_VIEWER_PID" ]] && kill -0 "$CAM_VIEWER_PID" 2>/dev/null; then
    kill "$CAM_VIEWER_PID" 2>/dev/null || true
  fi
  if [[ -n "$PX4_DOCKER_PID" ]] && kill -0 "$PX4_DOCKER_PID" 2>/dev/null; then
    kill "$PX4_DOCKER_PID" 2>/dev/null || true
  fi
  docker --context default rm -f "$PX4_CONTAINER_NAME" >/dev/null 2>&1 || true
  if [[ "$XHOST_OPENED" -eq 1 ]]; then
    xhost -local:docker >/dev/null 2>&1 || true
  fi
  exit "$ec"
}
trap cleanup EXIT INT TERM

run_host_py() {
  local -a env_cmd=(env -u PYTHONPATH -u PYTHONHOME)
  if [[ -n "$HOST_CUDA_LD_LIBRARY_PATH" ]]; then
    env_cmd+=(LD_LIBRARY_PATH="$HOST_CUDA_LD_LIBRARY_PATH")
  else
    env_cmd+=(-u LD_LIBRARY_PATH)
  fi
  env_cmd+=(
    GZ_PARTITION="$GZ_PARTITION"
    PYTHONNOUSERSITE=1
    "$PYTHON_BIN"
  )
  "${env_cmd[@]}" "$@"
}

echo "[AUTOSTART] root=$ROOT_DIR"
echo "[AUTOSTART] runs_root=$RUNS_ROOT"
echo "[AUTOSTART] python=$PYTHON_BIN"
echo "[AUTOSTART] gz_partition=$GZ_PARTITION"
echo "[AUTOSTART] px4_headless=$PX4_HEADLESS"
echo "[AUTOSTART] px4_gpu_mode=$PX4_GPU_MODE"

HOST_CUDA_LD_LIBRARY_PATH="$(
  env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 "$PYTHON_BIN" - <<'PY' 2>/dev/null || true
import os
import site

paths = []
for root in site.getsitepackages():
    for rel in (
        "nvidia/cudnn/lib",
        "nvidia/cublas/lib",
        "nvidia/cuda_nvrtc/lib",
        "nvidia/cuda_runtime/lib",
        "nvidia/cufft/lib",
        "nvidia/curand/lib",
    ):
        p = os.path.join(root, rel)
        if os.path.isdir(p):
            paths.append(p)

seen = set()
ordered = []
for p in paths:
    if p not in seen:
        seen.add(p)
        ordered.append(p)

print(":".join(ordered))
PY
)"
if [[ -n "$HOST_CUDA_LD_LIBRARY_PATH" ]]; then
  echo "[AUTOSTART] host_cuda_ld_path=detected"
else
  echo "[AUTOSTART] host_cuda_ld_path=not_detected"
fi

if pgrep -af '/usr/local/bin/mavsdk_server udp://:14540' >/dev/null 2>&1; then
  echo "[AUTOSTART] ERROR: rogue mavsdk_server is already running on udp://:14540." >&2
  echo "[AUTOSTART] Stop it first (usually via systemd mavsdk.service), then rerun." >&2
  exit 2
fi

DEPS_LOG="$RUNS_ROOT/deps_check.log"
set +e
run_host_py - <<'PY' >"$DEPS_LOG" 2>&1
import importlib
mods = ("onnxruntime", "yaml", "cv2", "mavsdk")
for name in mods:
    m = importlib.import_module(name)
    print(f"[DEPS] ok {name}: {getattr(m, '__file__', '')}")
print("[DEPS] all_ok")
PY
deps_ec=$?
set -e
if [[ "$deps_ec" -ne 0 ]]; then
  echo "[AUTOSTART] ERROR: python env check failed. Details:" >&2
  sed -n '1,120p' "$DEPS_LOG" >&2 || true
  if grep -q "compiled using NumPy 1.x" "$DEPS_LOG" 2>/dev/null; then
    echo "[AUTOSTART] Detected NumPy/OpenCV ABI mismatch (NumPy 2.x with OpenCV built for NumPy 1.x)." >&2
    echo "[AUTOSTART] Fix:" >&2
    echo "  $PYTHON_BIN -m pip install --force-reinstall 'numpy<2'" >&2
  else
    echo "[AUTOSTART] Install into selected interpreter:" >&2
    echo "  $PYTHON_BIN -m pip install --ignore-installed --no-deps PyYAML" >&2
    echo "  $PYTHON_BIN -m pip install --ignore-installed \"typing_extensions>=4.12\" \"protobuf>=6.32.1,<8\" mavsdk onnxruntime-gpu" >&2
    echo "  # For GTX 10xx / Pascal: use CUDA11 stack:" >&2
    echo "  $PYTHON_BIN -m pip install --force-reinstall \"onnxruntime-gpu==1.18.1\" \"nvidia-cudnn-cu11==8.9.6.50\" \"nvidia-cublas-cu11==11.11.3.6\" \"nvidia-cuda-nvrtc-cu11==11.8.89\" \"nvidia-cuda-runtime-cu11==11.8.89\" \"nvidia-cufft-cu11==10.9.0.58\" \"nvidia-curand-cu11==10.3.0.86\"" >&2
  fi
  echo "[AUTOSTART] Or pass --python <interpreter_with_gz+onnxruntime+mavsdk>." >&2
  exit 2
fi

if [[ "$SKIP_PX4" -eq 0 ]]; then
  if ! docker --context default image inspect "$PX4_IMAGE" >/dev/null 2>&1; then
    echo "[AUTOSTART] Docker image not found: $PX4_IMAGE" >&2
    exit 2
  fi

  xhost +local:docker >/dev/null
  XHOST_OPENED=1

  echo "[AUTOSTART] starting PX4 SITL container..."
  DOCKER_ENV_ARGS=(
    -e DISPLAY="$DISPLAY"
    -e QT_QPA_PLATFORM=xcb
    -e QT_X11_NO_MITSHM=1
    -e GZ_PARTITION="$GZ_PARTITION"
  )
  if [[ "$PX4_HEADLESS" -eq 1 ]]; then
    DOCKER_ENV_ARGS+=(-e HEADLESS=1)
  fi
  start_px4_container() {
    local gpu_mode="$1"
    local -a DOCKER_GPU_ARGS=()
    if [[ "$gpu_mode" == "on" || "$gpu_mode" == "auto" ]]; then
      DOCKER_GPU_ARGS=(
        --gpus all
        -e NVIDIA_VISIBLE_DEVICES=all
        -e NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute
      )
      echo "[AUTOSTART] starting PX4 with Docker GPU passthrough..."
    else
      echo "[AUTOSTART] starting PX4 without Docker GPU passthrough..."
    fi
    docker --context default run --rm --name "$PX4_CONTAINER_NAME" --network host \
      --user "$(id -u):$(id -g)" \
      "${DOCKER_GPU_ARGS[@]}" \
      "${DOCKER_ENV_ARGS[@]}" \
      -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
      -v "$PX4_REPO:$PX4_REPO" \
      -w "$PX4_REPO" \
      "$PX4_IMAGE" \
      bash -lc "make px4_sitl gz_x500" >"$PX4_LOG" 2>&1 &
    PX4_DOCKER_PID="$!"
  }
  start_px4_container "$PX4_GPU_MODE"

  READY=0
  GPU_FALLBACK_DONE=0
  for _ in $(seq 1 180); do
    if [[ -n "$PX4_DOCKER_PID" ]] && ! kill -0 "$PX4_DOCKER_PID" 2>/dev/null; then
      if [[ "$PX4_GPU_MODE" == "auto" && "$GPU_FALLBACK_DONE" -eq 0 ]]; then
        if grep -Eqi "could not select device driver|unknown flag: --gpus|nvidia-container-cli|could not load nvml|no nvidia driver" "$PX4_LOG" 2>/dev/null; then
          echo "[AUTOSTART] WARN: Docker GPU passthrough unavailable; falling back to CPU mode."
          echo "[AUTOSTART] HINT: install/configure nvidia-container-toolkit for Docker, then rerun with --px4-gpu auto|on."
          GPU_FALLBACK_DONE=1
          docker --context default rm -f "$PX4_CONTAINER_NAME" >/dev/null 2>&1 || true
          start_px4_container "off"
          sleep 1
          continue
        fi
      fi
      echo "[AUTOSTART] PX4 container exited unexpectedly. Last log lines:" >&2
      tail -n 80 "$PX4_LOG" >&2 || true
      exit 2
    fi
    if grep -Eq "Segmentation fault|gz_bridge failed to start|Timed out waiting for Gazebo world|Error creating symlink|Startup script returned with return value" "$PX4_LOG" 2>/dev/null; then
      echo "[AUTOSTART] PX4 startup fatal detected. Last log lines:" >&2
      tail -n 120 "$PX4_LOG" >&2 || true
      exit 2
    fi
    if grep -q "Ready for takeoff!" "$PX4_LOG" 2>/dev/null; then
      READY=1
      break
    fi
    sleep 1
  done
  if [[ "$READY" -ne 1 ]]; then
    echo "[AUTOSTART] timeout waiting PX4 readiness. Last log lines:" >&2
    tail -n 80 "$PX4_LOG" >&2 || true
    exit 2
  fi
  echo "[AUTOSTART] PX4 SITL is ready"
else
  echo "[AUTOSTART] skip-px4=true (assuming PX4+Gazebo already running)"
fi

echo "[AUTOSTART] camera smoke check..."
set +e
run_host_py scripts/px4_camera_smoke.py \
  --config "$CONFIG_PATH" \
  --platform-type px4 \
  --duration 6 \
  --startup-timeout 12 \
  --out-json "$CAM_SMOKE_JSON"
CAM_SMOKE_EC=$?
set -e
if [[ "$CAM_SMOKE_EC" -ne 0 ]]; then
  CAMERA_SMOKE_PASSED=0
  if [[ -f "$CAM_SMOKE_JSON" ]]; then
    CAMERA_SMOKE_PASSED="$(python3 - "$CAM_SMOKE_JSON" <<'PY'
import json, sys
path = sys.argv[1]
try:
    data = json.load(open(path, "r", encoding="utf-8"))
except Exception:
    print("0")
    raise SystemExit(0)
print("1" if bool(data.get("passed", False)) else "0")
PY
)"
  fi
  if [[ "$CAMERA_SMOKE_PASSED" == "1" ]]; then
    echo "[AUTOSTART] WARN: camera smoke exited with code $CAM_SMOKE_EC after PASS report; continuing." >&2
  else
    echo "[AUTOSTART] ERROR: camera smoke failed (exit=$CAM_SMOKE_EC)." >&2
    if [[ -f "$CAM_SMOKE_JSON" ]]; then
      echo "[AUTOSTART] camera smoke report: $CAM_SMOKE_JSON" >&2
      sed -n '1,120p' "$CAM_SMOKE_JSON" >&2 || true
    fi
    exit 2
  fi
fi

echo "[AUTOSTART] spawn/reposition actor..."
ACTOR_OK=0
for attempt in $(seq 1 "$ACTOR_SETUP_RETRIES"); do
  echo "[AUTOSTART] actor setup attempt $attempt/$ACTOR_SETUP_RETRIES"
  set +e
  run_host_py scripts/px4_actor_setup.py \
    --config "$CONFIG_PATH" \
    --out-json "$ACTOR_SETUP_JSON"
  ec=$?
  set -e
  if [[ "$ec" -eq 0 ]]; then
    ACTOR_OK=1
    break
  fi
  echo "[AUTOSTART] actor setup attempt failed (exit=$ec), retrying..."
  sleep 1
done

if [[ "$ACTOR_OK" -ne 1 ]]; then
  if [[ "$STRICT_ACTOR_SETUP" -eq 1 ]]; then
    echo "[AUTOSTART] ERROR: actor setup failed after retries" >&2
    exit 2
  fi
  echo "[AUTOSTART] WARN: actor setup failed after retries; continuing." >&2
  echo "[AUTOSTART] WARN: runtime motion thread will still try spawn_if_missing." >&2
fi

if [[ "$ENABLE_CAMERA_VIEWER" -eq 1 ]]; then
  echo "[AUTOSTART] starting camera viewer..."
  run_host_py scripts/camera_viewer.py \
    --config "$CONFIG_PATH" \
    --platform-type px4 >"$RUNS_ROOT/camera_viewer.log" 2>&1 &
  CAM_VIEWER_PID="$!"
fi

SCEN_RUN_ROOT="$RUNS_ROOT/scenarios"
mkdir -p "$SCEN_RUN_ROOT"

CMD=(
  scripts/run_scenarios.py
  --config "$CONFIG_PATH"
  --platform-type px4
  --target-mode actor
  --detector-type yolo_onnx
  --selector-backend external
  --scenarios "$SCENARIO_NAME"
  --duration "$DURATION_S"
  --runs-root "$SCEN_RUN_ROOT"
  --follow-mode "$FOLLOW_MODE"
  --follow-xy-strategy "$FOLLOW_XY_STRATEGY"
  --no-px4-cv-only
  --px4-auto-arm
  --px4-auto-takeoff
  --px4-takeoff-altitude "$TAKEOFF_ALT_M"
)

if [[ -n "$FOLLOW_PROFILE" ]]; then
  CMD+=(--follow-profile "$FOLLOW_PROFILE")
fi

if [[ -n "$PX4_TAKEOFF_CONFIRM_ALT" ]]; then
  CMD+=(--px4-takeoff-confirm-alt "$PX4_TAKEOFF_CONFIRM_ALT")
fi
if [[ -n "$PX4_OFFBOARD_MIN_ALT" ]]; then
  CMD+=(--px4-offboard-min-alt "$PX4_OFFBOARD_MIN_ALT")
fi
if [[ -n "$PX4_OFFBOARD_DELAY_AFTER_LIFTOFF" ]]; then
  CMD+=(--px4-offboard-delay-after-liftoff "$PX4_OFFBOARD_DELAY_AFTER_LIFTOFF")
fi
if [[ "$PX4_AUTO_ARM_REQUIRE_ARMABLE" == "true" ]]; then
  CMD+=(--px4-auto-arm-require-armable)
elif [[ "$PX4_AUTO_ARM_REQUIRE_ARMABLE" == "false" ]]; then
  CMD+=(--no-px4-auto-arm-require-armable)
fi
if [[ "$PX4_AUTO_ARM_REQUIRE_LOCAL_POSITION" == "true" ]]; then
  CMD+=(--px4-auto-arm-require-local-position)
elif [[ "$PX4_AUTO_ARM_REQUIRE_LOCAL_POSITION" == "false" ]]; then
  CMD+=(--no-px4-auto-arm-require-local-position)
fi
if [[ -n "$STATE_LOST_FRAME_THRESHOLD" ]]; then
  CMD+=(--state-lost-frame-threshold "$STATE_LOST_FRAME_THRESHOLD")
fi
if [[ -n "$STATE_REACQUIRE_THRESHOLD" ]]; then
  CMD+=(--state-reacquire-threshold "$STATE_REACQUIRE_THRESHOLD")
fi

if [[ "$FOLLOW_ENABLED" -eq 1 ]]; then
  CMD+=(--follow-enabled)
else
  CMD+=(--no-follow-enabled)
fi

if [[ "$ENABLE_VIZ" -eq 1 ]]; then
  CMD+=(--viz)
fi

if [[ "$ENABLE_RECORD" -eq 1 ]]; then
  CMD+=(--record)
fi

echo "[AUTOSTART] running runtime/scenario..."
echo "[AUTOSTART] command: $PYTHON_BIN ${CMD[*]}"
run_host_py "${CMD[@]}"

LATEST_RUN_META="$(ls -1t "$SCEN_RUN_ROOT"/*/run_meta.json 2>/dev/null | head -n1 || true)"
if [[ -n "$LATEST_RUN_META" && -f "$LATEST_RUN_META" ]]; then
  echo "[AUTOSTART] validating PX4 bridge status from run_meta..."
  run_host_py - "$LATEST_RUN_META" <<'PY'
import json
import sys

path = sys.argv[1]
data = json.load(open(path, "r", encoding="utf-8"))
pm = (data.get("platform_meta") or {})
connected = bool(pm.get("connected", False))
armed = bool(pm.get("armed", False))
in_air = bool(pm.get("in_air", False))
flight_mode = str(pm.get("flight_mode") or "")
relative_alt = float(pm.get("relative_altitude_m") or 0.0)
last_error = str(pm.get("last_error") or "")
print(f"[AUTOSTART] run_meta={path}")
print(
    f"[AUTOSTART] px4 connected={connected} armed={armed} in_air={in_air} "
    f"flight_mode={flight_mode} rel_alt={relative_alt:.2f}m last_error={last_error!r}"
)
if not connected:
    print("[AUTOSTART] ERROR: PX4 bridge was not connected during runtime.", file=sys.stderr)
    sys.exit(3)
if in_air and relative_alt < 0.30:
    print(
        "[AUTOSTART] WARN: PX4 reported in_air=True but relative altitude stayed low. "
        "Real liftoff likely failed; inspect px4_sitl.log and bridge takeoff settings.",
        file=sys.stderr,
    )
if last_error:
    print("[AUTOSTART] WARN: PX4 bridge reported last_error.", file=sys.stderr)
PY
fi

echo "[AUTOSTART] done"
echo "[AUTOSTART] artifacts: $RUNS_ROOT"
