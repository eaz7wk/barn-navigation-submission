#!/usr/bin/env bash
set -u

START=200
END=252
TIMEOUT=180

RESULTS_FILE="results_${START}_${END}.txt"
LOG_FILE="run_${START}_${END}.log"

export GAZEBO_MODEL_DATABASE_URI=""

if [ -f "${RESULTS_FILE}" ]; then
  mv "${RESULTS_FILE}" "${RESULTS_FILE%.txt}_old_$(date +%H%M%S).txt"
fi

echo "==== Batch run ${START}-${END} ====" | tee -a "${LOG_FILE}"
echo "Results -> ${RESULTS_FILE}" | tee -a "${LOG_FILE}"
echo "Log     -> ${LOG_FILE}" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

wait_for_result_line () {
  local world="$1"
  local file="$2"
  local timeout_s="$3"

  local start_ts
  start_ts=$(date +%s)

  while true; do
    if [ -f "$file" ] && grep -qE "^${world}[[:space:]]" "$file"; then
      return 0
    fi

    local now_ts
    now_ts=$(date +%s)
    if [ $((now_ts - start_ts)) -ge "$timeout_s" ]; then
      return 1
    fi

    sleep 0.2
  done
}

for i in $(seq ${START} ${END}); do
  echo "===============================" | tee -a "${LOG_FILE}"
  echo "[WORLD ${i}] Start run.py" | tee -a "${LOG_FILE}"

  python3 run.py --world_idx "${i}" --out "${RESULTS_FILE}" >> "${LOG_FILE}" 2>&1 &
  RUN_PID=$!

  echo "[WORLD ${i}] run.py PID=${RUN_PID}" | tee -a "${LOG_FILE}"

  if wait_for_result_line "${i}" "${RESULTS_FILE}" "${TIMEOUT}"; then
    echo "[WORLD ${i}] Result line detected." | tee -a "${LOG_FILE}"
  else
    echo "[WORLD ${i}] WARNING: No result line after timeout ${TIMEOUT}s." | tee -a "${LOG_FILE}"
  fi

  kill -TERM "${RUN_PID}" 2>/dev/null || true
  sleep 2
  kill -KILL "${RUN_PID}" 2>/dev/null || true

  pkill -f fixed_granular.py 2>/dev/null || true
  pkill -f gzserver 2>/dev/null || true
  pkill -f gzclient 2>/dev/null || true
  pkill -f roslaunch 2>/dev/null || true

  echo "[WORLD ${i}] cleanup #1" | tee -a "${LOG_FILE}"
  ./cleanup_sim.sh >> "${LOG_FILE}" 2>&1 || true

  sleep 5

  echo "[WORLD ${i}] cleanup #2" | tee -a "${LOG_FILE}"
  ./cleanup_sim.sh >> "${LOG_FILE}" 2>&1 || true

  sleep 5
done

echo "" | tee -a "${LOG_FILE}"
echo "==== Done. Results saved to ${RESULTS_FILE} ====" | tee -a "${LOG_FILE}"
