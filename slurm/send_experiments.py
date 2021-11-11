import os
import subprocess
import hydra
from fairseq.dataclass.configs import FairseqConfig
from omegaconf import OmegaConf


def create_slurm_script(slurm_name, run_name, working_dir, time, n_gpus, script_name):
    with open(slurm_name, "w") as f:
        f.write(f"""#!/bin/bash -x
#SBATCH --job-name={run_name}
#SBATCH --output={os.path.join(working_dir, "slurm.out")}
#SBATCH --error={os.path.join(working_dir, "slurm.err")}
#SBATCH --time={time}
#SBATCH --signal=USR1@120
#SBATCH --partition="killable"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=50000 
#SBATCH --cpus-per-task=4
#SBATCH --constraint="geforce_rtx_3090"
#SBATCH --gpus={n_gpus}

srun sh {script_name}""")


def create_run_script_and_config(run_working_dir, config, run_name):
    config_name = "run_config"
    cfg_file_name = os.path.join(run_working_dir, f"{config_name}.yaml")
    config = config[:config.find("slurm")]
    with open(cfg_file_name, "w") as f:
        f.write(config)

    with open(run_name, "w") as f:
        f.write(f"fairseq-hydra-train  \
 --config-dir {run_working_dir} \
 --config-name {config_name} \
 hydra.run.dir={run_working_dir} \
 checkpoint.save_dir={os.path.join(run_working_dir, 'checkpoints')}")


def send_job_and_report(slurm_name):
    print(f"sending {slurm_name}")
    process = subprocess.Popen(["sbatch", slurm_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    print("output:")
    print(stdout.decode("utf-8"))
    print("err:")
    print(stderr.decode("utf-8"))


@hydra.main(config_path=os.path.join("..", "fairseq", "config"), config_name="config")
def send(cfg: FairseqConfig) -> None:
    assert "slurm" in cfg, "add slurm.run_name, slurm.n_gpus, and slurm.time"

    working_dir = os.getcwd()
    slurm_name = os.path.join(working_dir, "slurm.sh")
    run_name = os.path.join(working_dir, "run.sh")
    create_slurm_script(slurm_name, cfg.slurm.run_name, working_dir,
                        cfg.slurm.time, cfg.slurm.n_gpus, run_name)

    yaml = OmegaConf.to_yaml(cfg)

    run_working_dir = os.path.join(working_dir, "run")
    os.mkdir(run_working_dir)
    create_run_script_and_config(run_working_dir, yaml, run_name)

    subprocess.Popen(["chmod", "ug+rx", run_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    send_job_and_report(slurm_name)


if __name__ == '__main__':
    send()
