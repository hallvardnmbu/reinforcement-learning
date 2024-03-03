Orion High-Performance Computing
--------------------------------

This directory contains the code for the Orion HPC cluster. The code is similar to the value-vision
code and example, but with included logging and model checkpoints.

Rearrange
---------

Rearrange the file structure in Orion to the following layout:

```
    ./
    ├── singularity/
    │   ├── singularity.def
    │   ├── singularity.sh
    │   │
    │   └── singularity.sif
    │
    ├── output/
    │   ├── print.out
    │   ├── debug.txt
    │   │
    │   ├── value-vision-tetris.gif
    │   ├── value-vision-tetris.png
    │   │
    │   ├── weights-{game-number}.pth
    │   │   ...
    │   └── weights-{final-game-number}.pth
    │
    ├── agent.py
    ├── train.py
    └── train.sh
```

Note that `./singularity/singularity.sif` must be created as mentioned below. Also note that
the files within the `output/` directory are created by the training script.

Execution
---------

Modify the `SIFFILE` path in `./train.sh` to point to the correct singularity file.

If you do not have a singularity file, you can create one by running:

```bash
cd singularity
sbatch singularity.sh singularity.def
```
    
This will create a `singularity.sif`-file in the `./singularity/` directory. You can then
reference this file in `./train.sh` by setting `SIFFILE` to `./singularity/singularity.sif`.

The job defined in `./train.sh` and is submitted by running:
    
```bash
sbatch train.sh
```

Notes
-----

The checkpointed weights are saved to `./output/weights-{game-number}.pth`.

The log messages (i.e., info and error messages) are written to `./output/debug.txt`. In order 
to view more detailed messages, set the log-level to debug in `./train.py`. 

Printouts are saved to `./output/print.out`, but the most informative output is the `.
/output/debug.txt`.
