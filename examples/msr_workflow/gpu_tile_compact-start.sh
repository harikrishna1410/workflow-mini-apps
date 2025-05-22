#!/usr/bin/env bash

display_help() {
  echo " Will map gpu tile to rank in compact and then round-robin fashion"
  echo " Usage:"
  echo "   mpiexec -np N gpu_tile_compact.sh [--oneapi] ./a.out"
  echo
  echo " Example 3 GPU of 2 Tiles with 7 Ranks:"
  echo "   0 Rank 0.0"
  echo "   1 Rank 0.1"
  echo "   2 Rank 1.0"
  echo "   3 Rank 1.1"
  echo "   4 Rank 2.0"
  echo "   5 Rank 2.1"
  echo "   6 Rank 0.0"
  echo
  echo " By default, the script will use \`ZE_AFFINITY_MASK\` to set the visiblity,"
  echo "  \`--oneapi\` can be used so \`ONEAPI_DEVICE_SELECTOR\` is used instead"
  echo " Hacked together by apl@anl.gov, please contact if bug found"
  exit 1
}

# This give the exact GPU count i915 knows about and I use udev to only enumerate the devices with physical presence.
num_gpu=$(/usr/bin/udevadm info /sys/module/i915/drivers/pci:i915/* |& grep -v Unknown | grep -c "P: /devices")
num_tile=2

if [ "$#" -eq 0 ] || [ "$1" == "--help" ] || [ "$1" == "-h" ] || [ "$num_gpu" = 0 ] ; then
  display_help
fi

# Get the RankID from different launcher
if [[ -v MPI_LOCALRANKID ]]; then
  _MPI_RANKID=$MPI_LOCALRANKID
elif [[ -v PALS_LOCAL_RANKID ]]; then
  _MPI_RANKID=$PALS_LOCAL_RANKID
else
  display_help
fi

use_zam=true
if [[ "$1" == "--oneapi" ]]; then
  shift;
  use_zam=false
fi

export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
ulimit -c 0 # Until Aurora filesystem problems are fixed

# https://oneapi-src.github.io/level-zero-spec/level-zero/latest/core/PROG.html#environment-variables
if [[ $ZE_FLAT_DEVICE_HIERARCHY == COMPOSITE ]] ; then
  if [ -z "$START_GPU" ]; then
    gpu_id=$(((_MPI_RANKID / num_tile) % num_gpu))
  else
    gpu_id=$((${START_GPU} + ((_MPI_RANKID / num_tile) % num_gpu)))
  fi
  tile_id=$((_MPI_RANKID % num_tile))
  mask=$gpu_id.$tile_id
else
  gpu_id=$((_MPI_RANKID % ( num_gpu * num_tile )))
  mask=$gpu_id
fi

if $use_zam; then
  export ZE_AFFINITY_MASK=$mask
else
  # https://intel.github.io/llvm/EnvironmentVariables.html#oneapi-device-selector
  export ONEAPI_DEVICE_SELECTOR=level_zero:$mask
fi

# https://stackoverflow.com/a/28099707/7674852
exec "$@"
