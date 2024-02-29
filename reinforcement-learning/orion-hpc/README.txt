Orion High-Performance Computing
--------------------------------

This directory contains the code for the Orion HPC cluster. The code is similar to the value-vision
code and example, but with included logging and model checkpoints.

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

The checkpointed weights are saved to `./output/weights-{game-number}.pth`. And the final weights
are saved to `./output/weights-final.pth`.

The log messages (i.e., debug and info) are written to `./output/debug.txt`.

Printouts are saved to `./output/log.txt` and `./output/print_{%s}.txt`.
