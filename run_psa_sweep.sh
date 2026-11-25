#!/bin/bash
# Runs the lane keeping experiment many times: a baseline (no attack) and several camera yaw offsets (PSA),
# each one REPEATS times, always with the same car, map and spawn point. No receiver, no video.
#
#   ./run_psa_sweep.sh                       # baseline + yaw 5 7 10 15 20, 2 repeats each
#   ./run_psa_sweep.sh --yaws "5 10" --repeats 3 --run-seconds 15
#   ./run_psa_sweep.sh --yaws "7" --camera-x 0.6 --defense compensate   # with the hood defence
#
# Output (default psa_sweep_<date>/): one CSV and one log per run, then a table and a plot.
# Takes about 2.5 minutes per run (CARLA is restarted every time).
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"

YAWS="5 7 10 15 20"; REPEATS=2; RUN_SECONDS=15; SPAWN=124; MAP=Town04; BASELINE=1; CAMERA_X=""; DEFENSE="off"
OUT="psa_sweep_$(date +%Y%m%d_%H%M)"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --yaws) YAWS="$2"; shift 2 ;;
        --repeats) REPEATS="$2"; shift 2 ;;
        --run-seconds) RUN_SECONDS="$2"; shift 2 ;;
        --spawn-index) SPAWN="$2"; shift 2 ;;
        --map) MAP="$2"; shift 2 ;;
        --out) OUT="$2"; shift 2 ;;
        --no-baseline) BASELINE=0; shift ;;
        --camera-x) CAMERA_X="$2"; shift 2 ;;     # 0.6 = camera inside the cabin (the hood is visible)
        --defense) DEFENSE="$2"; shift 2 ;;       # off | detect | stop | compensate (needs --camera-x 0.6)
        -h|--help) sed -n 2,10p "$0"; exit 0 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

PYTHON_EXEC="${PYTHON_EXEC:-python}"
mkdir -p "${OUT}"

# One sudo password at the start; keep the credentials alive during the whole sweep.
sudo -v || exit 1
( while true; do sudo -n true; sleep 50; done ) 2>/dev/null &
KEEPALIVE=$!
cleanup() { kill "${KEEPALIVE}" 2>/dev/null; ./2_down_environment.sh >/dev/null 2>&1; sudo ip link delete veth-s 2>/dev/null; }
trap cleanup EXIT
trap 'echo; echo "Interrupted."; exit 130' INT TERM

run_one() {   # run_one <yaw> <repeat>
    local yaw="$1" rep="$2"
    local name; name=$(printf 'yaw%02d_r%d' "${yaw}" "${rep}")
    local csv="${OUT}/${name}.csv" log="${OUT}/${name}.log"
    rm -f "${csv}"
    ./2_down_environment.sh >/dev/null 2>&1
    sudo ip link delete veth-s 2>/dev/null
    sleep 2
    echo -n "[$(date +%H:%M:%S)] ${name} (yaw ${yaw}) ... "
    NO_PROMPT=1 ./1_up_environment.sh --map "${MAP}" --lane-keeping drive --spawn-index "${SPAWN}" \
        --run-seconds "${RUN_SECONDS}" --psa-yaw "${yaw}" --log-csv "${csv}" --defense "${DEFENSE}" \
        ${CAMERA_X:+--camera-x "${CAMERA_X}"} > "${log}" 2>&1
    local waited=0 limit=$((RUN_SECONDS * 25 + 120))
    until grep -q "\[LK\] summary" "${log}" 2>/dev/null; do
        sleep 2; waited=$((waited + 2))
        if [[ ${waited} -ge ${limit} ]] || ! pgrep -f CARLA_client_module.py >/dev/null; then
            echo "FAILED (no summary)"; return 1
        fi
    done
    sleep 2
    grep -a "\[LK\] \(vehicle\|summary\|defence\)" "${log}" | tail -3 | tr '\n' ' '; echo
    ./2_down_environment.sh >/dev/null 2>&1
    return 0
}

CONDITIONS=""
[[ ${BASELINE} -eq 1 ]] && CONDITIONS="0"
CONDITIONS="${CONDITIONS} ${YAWS}"

echo "Sweep: yaws [${CONDITIONS}] x ${REPEATS} repeats, ${RUN_SECONDS} s each, camera x ${CAMERA_X:-front}, defence ${DEFENSE} -> ${OUT}/"
for rep in $(seq 1 "${REPEATS}"); do          # repeats are interleaved so a drift of the machine does not bias one yaw
    for yaw in ${CONDITIONS}; do
        run_one "${yaw}" "${rep}" || run_one "${yaw}" "${rep}" || echo "  giving up on yaw ${yaw}, repeat ${rep}"
    done
done

sudo chown -R "$(id -u):$(id -g)" "${OUT}" 2>/dev/null
echo
"${PYTHON_EXEC}" analyze_psa_sweep.py "${OUT}"
