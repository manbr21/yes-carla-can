#!/bin/bash

CARLA_FOLDER_NAME="${CARLA_FOLDER_NAME:-carla-0-9-15}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-n4s_env}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DBC_PATH="${DBC_PATH:-data/carla.dbc}"
VCAN_INTERFACE="${VCAN_INTERFACE:-vcan0}"
MAP="${MAP:-}"
PSA_YAW="${PSA_YAW:-0}"
LANE_KEEPING="${LANE_KEEPING:-off}"
SPAWN_INDEX="${SPAWN_INDEX:-}"
CAMERA_X="${CAMERA_X:-}"     # lane keeping camera x (m from the car centre); 0.6 shows the hood
HOOD_B_REF="${HOOD_B_REF:--0.0154}"   # benign value of the paper's linear coefficient on this camera
HOOD_REF="${HOOD_REF:-0}"    # hood position measured without attack (px)
DEFENSE="${DEFENSE:-off}"    # hood defence: off | detect | stop | compensate
NO_PROMPT="${NO_PROMPT:-}"   # set (e.g. NO_PROMPT=1) to skip the "start the receiver" prompt (automated runs, no video)
VEHICLE="${VEHICLE:-}"       # blueprint id, e.g. vehicle.tesla.model3 (default: random; model3 when lane keeping is on)
RUN_SECONDS="${RUN_SECONDS:-0}"
LOG_CSV="${LOG_CSV:-}"
LK_BIAS="${LK_BIAS:-}"     # e.g. LK_BIAS=0.12 (error measured on the benign run)
LK_DUMP="${LK_DUMP:-}"     # e.g. LK_DUMP=seg_benign (saves segmentation images for debugging)
LK_GAINS="${LK_GAINS:-}"   # e.g. LK_GAINS="0.4 0.05" (lane keeping KP KD)
AVTP_DIR="${SCRIPT_DIR}/avtp_network"

usage() {
    cat <<EOF
Usage: $0 [-h|--help] [--dbc <path>] [--vcan <name>] [--map <name>]
       [--lane-keeping off|log|drive] [--psa-yaw <deg>] [--spawn-index <n>] [--vehicle <blueprint>] [--run-seconds <s>] [--log-csv <file>]

Start the "Yes, CARLA CAN" simulation environment.

What this script does:
  1. Launches the CARLA simulator in headless, low-quality mode
  2. Creates the virtual CAN bus (${VCAN_INTERFACE:-vcan0}) using the Linux kernel vcan module
  3. Creates a second virtual CAN bus (vcan1) for the attacker and bridges them via can-gw
     so that candump labels attacker frames as 'R' and normal frames as 'T'
  4. Waits 10 seconds for CARLA to initialise
  5. Starts the CARLA client module (spawns the vehicle and sensors)
  6. Starts the vehicle controls module (translates vehicle state into CAN frames)

Options:
  -h, --help      Show this help message and exit
  --dbc <path>    Path to the DBC file defining the virtual CAN network schema
                  (default: data/carla.dbc)
  --vcan <name>   Name of the virtual CAN interface to create (default: vcan0)
  --map <name>    CARLA map to load in the client (e.g. Town01, Town10HD).
                  If omitted, the client uses the map currently loaded by CARLA.
  --lane-keeping <mode>
                  Camera-based lane keeping node: off (default), log (measure only)
                  or drive (sends STEER/THROTTLE over CAN; the keyboard sender is not started).
  --psa-yaw <deg> Perspective-Shift Attack: yaw offset of the front cameras (default: 0 = no attack).
  --spawn-index <n>  Use spawn point n of the map (repeatable runs). --run-seconds <s> stops the run after s seconds.
  --log-csv <file>   Write the lane keeping measurements to a CSV file.

Environment variables:
  CARLA_FOLDER_NAME     Directory where CARLA is installed (default: carla-0-9-15)
  CONDA_ENV_NAME        Conda environment to use (default: n4s_env)
  DBC_PATH              DBC file path, overridden by --dbc if provided
  VCAN_INTERFACE        Virtual CAN interface name, overridden by --vcan if provided (default: vcan0)
  MAP                   CARLA map name, overridden by --map if provided
  VK_ICD_FILENAMES      Force a specific Vulkan ICD file (skips auto-detection)
EOF
}

# Allow overriding the DBC file via --dbc argument
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --dbc) DBC_PATH="$2"; shift 2 ;;
        --vcan) VCAN_INTERFACE="$2"; shift 2 ;;
        --map) MAP="$2"; shift 2 ;;
        --lane-keeping) LANE_KEEPING="$2"; shift 2 ;;
        --psa-yaw) PSA_YAW="$2"; shift 2 ;;
        --spawn-index) SPAWN_INDEX="$2"; shift 2 ;;
        --vehicle) VEHICLE="$2"; shift 2 ;;
        --camera-x) CAMERA_X="$2"; shift 2 ;;
        --defense) DEFENSE="$2"; shift 2 ;;
        --hood-ref) HOOD_REF="$2"; shift 2 ;;
        --hood-b-ref) HOOD_B_REF="$2"; shift 2 ;;
        --run-seconds) RUN_SECONDS="$2"; shift 2 ;;
        --log-csv) LOG_CSV="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; usage; exit 1 ;;
    esac
done

# Exporta o diretório do projeto para o PYTHONPATH (preservado pelo sudo -E)---- para evitar problemas com importaco de modulos
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

# Resolve Conda Python binary dynamically (no hardcoded user paths)
if [[ -n "${CONDA_PREFIX}" && "${CONDA_DEFAULT_ENV}" == "${CONDA_ENV_NAME}" ]]; then
    PYTHON_EXEC="${CONDA_PREFIX}/bin/python"
else
    # Fallback search via conda environment list
    PYTHON_EXEC=$(conda env list 2>/dev/null | grep -E "^${CONDA_ENV_NAME}[[:space:]]" | awk '{print $NF"/bin/python"}')
fi

# Fallback to system python if Conda is not active or environment not found
if [[ -z "${PYTHON_EXEC}" || ! -f "${PYTHON_EXEC}" ]]; then
    PYTHON_EXEC=$(which python)
fi

echo "Using Python binary: ${PYTHON_EXEC}"



# On hybrid Intel/NVIDIA systems, Vulkan may default to the Intel GPU and cause
# crashes. Force the NVIDIA ICD if available; otherwise fall back to the default.
# Skip auto-detection if the user already set VK_ICD_FILENAMES explicitly.

# Start CARLA simulator in the background
echo "Starting CARLA simulator..."
./${CARLA_FOLDER_NAME}/CarlaUE4.sh -quality-level=Low -nosound -windowed -ResX=800 -ResY=600 &

echo "Setting up AVTP virtual network..."
# veth-s is moved to the host namespace below, so deleting the "sender" namespace does not remove it.
# A leftover from a previous run makes setup.sh fail with "RTNETLINK answers: File exists".
sudo ip link delete veth-s 2>/dev/null || true
sudo bash "${AVTP_DIR}/setup.sh" --capture

# 3. Mover a ponta veth-s para o Host (permite ao CARLA rodar no Host com acesso ao AVTP L2)
echo "Exposing AVTP sender interface to Host..."
sudo ip netns exec sender ip link set veth-s netns 1
sleep 1
sudo ip link set dev veth-s up 

# Set up virtual CAN bus
echo "Setting up virtual CAN bus..."
sudo modprobe vcan
sudo modprobe can-gw
sudo ip link add dev "${VCAN_INTERFACE}" type vcan 2>/dev/null || true
sudo ip link set up "${VCAN_INTERFACE}" 2>/dev/null || true

# Set up attacker CAN bus and bridge it to the main bus via can-gw.
# Frames sent on vcan1 are forwarded to vcan0 and marked 'R' (received) by candump,
# while frames sent directly on vcan0 are marked 'T' (transmitted).
echo "Setting up attacker CAN bus and can-gw bridge..."
sudo ip link add dev vcan1 type vcan
sudo ip link set up vcan1
sudo cangw -A -s vcan1 -d "${VCAN_INTERFACE}" -e 2>/dev/null || true
sudo cangw -A -s "${VCAN_INTERFACE}" -d vcan1 -e 2>/dev/null || true



# Give CARLA a moment to initialise before connecting clients
echo "Waiting for CARLA to start..."
sleep 10

# Executa os módulos Python no Host com privilégios para o Scapy
echo "Starting CARLA client module..."
CLIENT_ARGS=(--vcan "${VCAN_INTERFACE}")
if [[ -n "${MAP}" ]]; then
    echo "Requesting CARLA map: ${MAP}"
    CLIENT_ARGS+=(--map "${MAP}")
fi
CLIENT_ARGS+=(--lane-keeping "${LANE_KEEPING}" --psa-yaw "${PSA_YAW}" --run-seconds "${RUN_SECONDS}")
[[ -n "${SPAWN_INDEX}" ]] && CLIENT_ARGS+=(--spawn-index "${SPAWN_INDEX}")
# The camera height (and so the lane width in the image) and the acceleration depend on the car model:
# lane keeping runs must always use the same one to be comparable.
if [[ -z "${VEHICLE}" && "${LANE_KEEPING}" != "off" ]]; then VEHICLE="vehicle.tesla.model3"; fi
[[ -n "${VEHICLE}" ]] && CLIENT_ARGS+=(--filter "${VEHICLE}")
[[ -n "${CAMERA_X}" ]] && CLIENT_ARGS+=(--camera-x "${CAMERA_X}")
CLIENT_ARGS+=(--defense "${DEFENSE}" --hood-ref "${HOOD_REF}" --hood-b-ref "${HOOD_B_REF}")
[[ -n "${LOG_CSV}" ]] && CLIENT_ARGS+=(--log-csv "${LOG_CSV}")
[[ -n "${LK_DUMP}" ]] && CLIENT_ARGS+=(--lk-dump "${LK_DUMP}")
[[ -n "${LK_BIAS}" ]] && CLIENT_ARGS+=(--lk-bias "${LK_BIAS}")
[[ -n "${LK_GAINS}" ]] && CLIENT_ARGS+=(--lk-gains ${LK_GAINS})

# Executar os módulos Python no Host com sudo -E (Acesso a vcan0 + veth-s Socket RAW + localhost)
echo ""
echo "======================================================================"
echo " INFRAESTRUTURA DE REDE E CARLA PRONTOS!"
echo " Agora, abra OUTRO terminal e inicie o RECEIVER:"
echo ""
echo "   xhost +local:sudo "
echo ""
echo ""
echo "   sudo -E DISPLAY=\$DISPLAY PYTHONPATH=. ip netns exec receiver \"\$CONDA_PREFIX/bin/python\" avtp_network/receiver.py -i veth-r -o ."
echo ""
echo "======================================================================"
if [[ -z "${NO_PROMPT}" ]]; then
    read -p "Pressione [ENTER] AQUI após iniciar o receiver para subir os clientes do CARLA..."
fi

# Executar os módulos Python no Host
echo "Starting CARLA client module..."
sudo -E "${PYTHON_EXEC}" "${SCRIPT_DIR}/CARLA_client_module.py" "${CLIENT_ARGS[@]}" &

if [[ "${LANE_KEEPING}" == "drive" ]]; then
    echo "Lane keeping drives the car over CAN: vehicle controls module not started."
else
    echo "Starting vehicle controls module..."
    sudo "${PYTHON_EXEC}" "${SCRIPT_DIR}/vehicle_controls_module.py" --dbc "${DBC_PATH}" --vcan "${VCAN_INTERFACE}" &
fi

echo "Environment is completely up!"