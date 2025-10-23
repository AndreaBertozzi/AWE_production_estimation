#!/usr/bin/env bash
# -------------------------------------------------------
# Sets up the environment for AWE Production Estimation
# -------------------------------------------------------
# Export PYTHONPATH so Python finds the awe_pe package
export PYTHONPATH=$PWD

# Print confirmation
echo "Environment ready."
echo "PYTHONPATH set to: $PYTHONPATH"
echo "You can now run Python scripts, e.g.:"
echo "  python examples/examples_qsm.py"
