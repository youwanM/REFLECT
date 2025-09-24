#!/bin/bash

# Get the hostname
hostname=$(hostname)

# Check if "abacus" is in the hostname
if [[ "$hostname" != *abacus* ]]; then
  echo "Hostname does not contain 'abacus'. Exiting script."
  exit 1
fi

echo "Hostname contains 'abacus', continuing script"

# Activate virtual environment and run Python script
source venv3.10/bin/activate
torchrun train_REFLECT-2.py
