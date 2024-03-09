Orion High-Performance Computing
--------------------------------

The subdirectory `upload` contain the code for the Orion HPC cluster. The code in this module is 
similar to the `../DQN.py` code and example, but with included logging and model checkpoints.

Example
-------

The weights of the pre-trained agent can be loaded as seen in the notebook `EXAMPLE.ipynb`.

Usage
-----

In order to train your own agent using Orion, upload the directory (excepting `README.md` and 
`EXAMPLE.ipynb`) to the platform, and create the `upload/singularity/singularity.sif` file as 
mentioned below.

Execution
---------

Modify the `SIFFILE` path in `train.sh` to point to the correct singularity file.

If you do not have a singularity file, you can create one by running:

```bash
cd singularity
sbatch singularity.sh singularity.def
```
    
This will create a `singularity.sif`-file in the `singularity` directory. You can then
reference this file in `train.sh` by setting `SIFFILE` to `singularity/singularity.sif`.

The job defined in `train.sh` and is submitted by running:
    
```bash
sbatch train.sh
```

Notes
-----

The checkpointed weights are saved to `./output/weights-{game-number}.pth`.

The log messages (i.e., info and error messages) are written to `./output/info.txt`.

Printouts are saved to `./output/print.out`, but the most informative output is the displayed in 
`./output/info.txt`.

Metrics throughout training is saved to `./output/metrics.csv`.
