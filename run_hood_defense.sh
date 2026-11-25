#!/bin/bash
# Hood defence experiment, all in one:
#   1. PROBE: looks for a camera position where the hood shows up in the image (tries --candidates in order) and measures
#      where the hood sits when nothing is wrong (the reference).
#   2. RUNS: for each camera yaw, with the defence off and with the defence on ("compensate"), same car/map/spawn.
#   3. Table and plot (analyze_hood_defense.py).
#
#   ./run_hood_defense.sh                         # probe + yaws -20 -10 0 10 20, defence off and on
#   (a negative yaw turns the camera to the left: the hood moves to the right, the direction the paper's rule flags)
#   ./run_hood_defense.sh --probe-only            # only step 1 (to look at the images in <out>/probe_*/)
#   ./run_hood_defense.sh --camera-x 0.6 --yaws "7"   # skip the probe search and use this camera position
#
# About 2.5 minutes per run (CARLA restarts every time): the default is 8 runs + the probe.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"

YAWS="-20 -10 0 10 20"; DEFENSES="off compensate"; RUN_SECONDS=15; SPAWN=124; MAP=Town04
CANDIDATES="0.6 0.9 0.3 1.2"; CAMERA_X=""; PSA_MODE="shift"; PROBE_ONLY=0; REPEATS=1
OUT="hood_defense_$(date +%Y%m%d_%H%M)"
MIN_HOOD_PX=800

while [[ $# -gt 0 ]]; do
    case "$1" in
        --yaws) YAWS="$2"; shift 2 ;;
        --defenses) DEFENSES="$2"; shift 2 ;;
        --run-seconds) RUN_SECONDS="$2"; shift 2 ;;
        --spawn-index) SPAWN="$2"; shift 2 ;;
        --map) MAP="$2"; shift 2 ;;
        --candidates) CANDIDATES="$2"; shift 2 ;;
        --camera-x) CAMERA_X="$2"; shift 2 ;;
        --repeats) REPEATS="$2"; shift 2 ;;
        --psa-mode) PSA_MODE="$2"; shift 2 ;;     # shift (default, hood moves with the scene) | rotate
        --out) OUT="$2"; shift 2 ;;
        --probe-only) PROBE_ONLY=1; shift ;;
        -h|--help) sed -n 2,14p "$0"; exit 0 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

PYTHON_EXEC="${PYTHON_EXEC:-python}"
mkdir -p "${OUT}"
sudo -v || exit 1
( while true; do sudo -n true; sleep 50; done ) 2>/dev/null &
KEEPALIVE=$!
cleanup() { kill "${KEEPALIVE}" 2>/dev/null; ./2_down_environment.sh >/dev/null 2>&1; sudo ip link delete veth-s 2>/dev/null
            sudo chown -R "$(id -u):$(id -g)" "${OUT}" 2>/dev/null; }
trap cleanup EXIT
trap 'echo; echo "Interrupted."; exit 130' INT TERM

# run_one <dir> <name> <yaw> <defense> <camera_x> <hood_ref> <hood_b_ref> <run_seconds> [dump_dir]
run_one() {
    local dir="$1" name="$2" yaw="$3" defense="$4" camx="$5" ref="$6" bref="$7" secs="$8" dump="${9:-}"
    mkdir -p "${dir}"
    local csv="${dir}/${name}.csv" log="${dir}/${name}.log"
    rm -f "${csv}"
    ./2_down_environment.sh >/dev/null 2>&1
    sudo ip link delete veth-s 2>/dev/null
    sleep 2
    echo -n "[$(date +%H:%M:%S)] ${dir}/${name} (yaw ${yaw}, defence ${defense}, camera x ${camx}) ... "
    LK_DUMP="${dump}" NO_PROMPT=1 ./1_up_environment.sh --map "${MAP}" --lane-keeping drive --spawn-index "${SPAWN}" \
        --run-seconds "${secs}" --psa-yaw "${yaw}" --log-csv "${csv}" --camera-x "${camx}" --defense "${defense}" \
        --hood-ref "${ref}" --hood-b-ref "${bref}" --psa-mode "${PSA_MODE}" > "${log}" 2>&1
    local waited=0 limit=$((secs * 25 + 120))
    until grep -q "\[LK\] summary" "${log}" 2>/dev/null; do
        sleep 2; waited=$((waited + 2))
        if [[ ${waited} -ge ${limit} ]] || ! pgrep -f CARLA_client_module.py >/dev/null; then
            echo "FAILED (no summary)"; return 1
        fi
    done
    sleep 2
    grep -a "\[LK\] \(summary\|defence\)" "${log}" | tr '\n' ' '; echo
    ./2_down_environment.sh >/dev/null 2>&1
    return 0
}

# b of the paper's hood fit measured in a probe log (the paper's own benign value if the hood edge was not found)
edge_b() {
    local l; l=$(grep -a "hood edge:" "$1" | head -1)
    if [[ "${l}" =~ b=([-+0-9.]+) ]]; then echo "${BASH_REMATCH[1]}"; else echo "-0.0154"; fi
}

# ---------------------------------------------------------------- 1. probe
HOOD_REF=0; HOOD_B=-0.0154
if [[ -z "${CAMERA_X}" ]]; then
    echo "== Step 1: looking for a camera position where the hood is visible"
    for x in ${CANDIDATES}; do
        run_one "${OUT}" "probe_x${x}" 0 detect "${x}" 0 -0.0154 2 "${OUT}/probe_x${x}" || continue
        line=$(grep -a "hood visible" "${OUT}/probe_x${x}.log" | head -1)
        if [[ "${line}" =~ hood\ visible\ \(([0-9]+)\ px,\ ([-+0-9.]+)\ px ]]; then
            npx="${BASH_REMATCH[1]}"; ref="${BASH_REMATCH[2]}"
            echo "   camera x ${x}: hood visible, ${npx} px, ${ref} px from the centre"
            if [[ ${npx} -ge ${MIN_HOOD_PX} ]]; then CAMERA_X="${x}"; HOOD_REF="${ref}"; HOOD_B=$(edge_b "${OUT}/probe_x${x}.log"); break; fi
        else
            echo "   camera x ${x}: $(grep -a 'bottom of the image' "${OUT}/probe_x${x}.log" | head -1)"
        fi
    done
    if [[ -z "${CAMERA_X}" ]]; then
        echo "The hood did not show up in any candidate (${CANDIDATES}). Look at the images in ${OUT}/probe_x*/ and the lines"
        echo "'[LK] bottom of the image' in the .log files, then try --camera-x with another value."
        exit 2
    fi
else
    # camera position given: still measure the reference on a run without attack
    run_one "${OUT}" "probe_x${CAMERA_X}" 0 detect "${CAMERA_X}" 0 -0.0154 2 "${OUT}/probe_x${CAMERA_X}" || exit 2
    line=$(grep -a "hood visible" "${OUT}/probe_x${CAMERA_X}.log" | head -1)
    if [[ "${line}" =~ hood\ visible\ \(([0-9]+)\ px,\ ([-+0-9.]+)\ px ]]; then HOOD_REF="${BASH_REMATCH[2]}"; HOOD_B=$(edge_b "${OUT}/probe_x${CAMERA_X}.log")
    else echo "The hood is not visible with camera x ${CAMERA_X}."; exit 2; fi
fi
echo "== Camera x = ${CAMERA_X}, hood reference = ${HOOD_REF} px, paper coefficient b (benign) = ${HOOD_B}"
echo "${CAMERA_X} ${HOOD_REF} ${HOOD_B}" > "${OUT}/camera_and_reference.txt"
[[ ${PROBE_ONLY} -eq 1 ]] && { echo "Probe only: done. Images in ${OUT}/probe_x${CAMERA_X}/"; exit 0; }

# ---------------------------------------------------------------- 2. runs
echo "== PSA mode: ${PSA_MODE}"
echo "== Step 2: yaws [${YAWS}] x defences [${DEFENSES}] x ${REPEATS} repeat(s)"
for rep in $(seq 1 "${REPEATS}"); do
    for yaw in ${YAWS}; do
        for defense in ${DEFENSES}; do
            ayaw=${yaw#-}; sign=""; [[ "${yaw}" == -* ]] && sign="m"       # file names: yawm10_r1 = -10 degrees
            name=$(printf 'yaw%s%02d_r%d' "${sign}" "${ayaw}" "${rep}")
            run_one "${OUT}/${defense}" "${name}" "${yaw}" "${defense}" "${CAMERA_X}" "${HOOD_REF}" "${HOOD_B}" "${RUN_SECONDS}" \
                || run_one "${OUT}/${defense}" "${name}" "${yaw}" "${defense}" "${CAMERA_X}" "${HOOD_REF}" "${HOOD_B}" "${RUN_SECONDS}" \
                || echo "  giving up on ${defense} yaw ${yaw}"
        done
    done
done

sudo chown -R "$(id -u):$(id -g)" "${OUT}" 2>/dev/null
echo
"${PYTHON_EXEC}" analyze_hood_defense.py "${OUT}"
