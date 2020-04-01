#!/bin/bash
for script in config/*; do sbatch ./run_rgi_generic_singularity_all.slurm "$script"; done
