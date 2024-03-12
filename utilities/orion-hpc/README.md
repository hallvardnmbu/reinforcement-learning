Orion High-Performance Computing
--------------------------------

Usage
-----

In order to train your own agent using Orion, upload the directory `singularity/` to the 
platform, and create the `singularity.sif` file as mentioned below.

In addition, upload the nessecary `agent.py` and `train.py` files. For instance, by uploading 
`breakout/DQN.py` (renaming this to `agent.py`) and `breakout/train.py` to Orion. 

Execution
---------

In `train.sh`, reference the `singularity.sif`-file (creation guide below) by setting `SIFFILE` 
to `singularity/singularity.sif` by modifying `SIFFILE` in `train.sh`. (Make sure to also modify 
`--job-name` to correspond to the task at hand.)

The job is (defined in `train.sh` and is) then submitted by running:

```bash
sbatch train.sh
```

Singularity
-----------

If you do not have a singularity file, you can create one by following the steps:

1. Modify `AUTHOR_NAME` and `AUTHOR_EMAIL` in `singularity.def`

2. ```bash
   cd singularity
   sbatch singularity.sh singularity.def
   ```

This will create a `singularity.sif`-file in the `singularity` directory. 

Notes
-----

The checkpointed weights are saved to `./output/weights-{game-number}.pth`.

The log messages (i.e., info and error messages) are written to `./output/info.txt`.

Printouts are saved to `./output/print.out`, but the most informative output is the displayed in 
`./output/info.txt`.

Metrics throughout training is saved to `./output/metrics.csv`.
