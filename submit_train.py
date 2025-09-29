# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# A script to run multinode training with submitit.
# --------------------------------------------------------

import argparse
import logging
import os
import sys
import uuid
from pathlib import Path

import submitit

import train

logger = logging.getLogger(__name__)


def parse_args():
    trainer_parser = train.get_args_parser()
    parser = argparse.ArgumentParser(
        "Submitit for flow_matching training", parents=[trainer_parser]
    )
    parser.add_argument(
        "--ngpus", default=1, type=int, help="Number of gpus to request on each node"
    )
    parser.add_argument(
        "--nodes", default=1, type=int, help="Number of nodes to request"
    )
    parser.add_argument(
        "--timeout",
        default=1440,  # 24 hours * 60 minutes
        type=int,
        help="Duration of the job (in minutes)",
    )
    parser.add_argument(
        "--job_dir", default="", type=str, help="Job dir. Leave empty for automatic."
    )

    parser.add_argument(
        "--partition",
        default="2080-galvani",
        type=str,
        help="Partition where to submit",
    )
    parser.add_argument(
        "--constraint",
        default="",
        type=str,
        help="Slurm constraint eg.: ampere80gb For using A100s or volta32gb for using V100s.",
    )
    parser.add_argument(
        "--comment", default="", type=str, help="Comment to pass to scheduler"
    )
    parser.add_argument("--qos", default="", type=str, help="Slurm QOS")
    parser.add_argument("--account", default="",
                        type=str, help="Slurm account")
    parser.add_argument(
        "--exclude",
        default="",
        type=str,
        help="Exclude certain nodes from the slurm job.",
    )
    return parser.parse_args()


def get_shared_folder() -> Path:
    shared_dir = os.getenv("ATMOS_SLURM_DIR", None)
    if shared_dir is None:
        raise RuntimeError(
            "No shared folder available. Make sure ATMOS_SLURM_DIR is set."
        )
    if Path(shared_dir).is_dir():
        p = Path(shared_dir) / "experiments"
        p.mkdir(exist_ok=True)
        return p
    raise RuntimeError("No shared folder available")


def get_init_file():
    shared_dir = os.getenv("ATMOS_SLURM_DIR", None)
    if shared_dir is None:
        raise RuntimeError(
            "No shared folder available. Make sure ATMOS_SLURM_DIR is set."
        )
    # Init file must not exist, but it's parent dir must exist.
    os.makedirs(str(get_shared_folder()), exist_ok=True)
    init_file = get_shared_folder() / f"{uuid.uuid4().hex}_init"
    if init_file.exists():
        os.remove(str(init_file))
    return init_file


class Trainer(object):
    def __init__(self, args):
        self.args = args

    def __call__(self):
        import train

        self._setup_gpu_args()
        train.main(self.args)

    def checkpoint(self):
        import os

        import submitit

        self.args.dist_url = get_init_file().as_uri()
        checkpoint_file = os.path.join(self.args.output_dir, "checkpoint.pth")
        if os.path.exists(checkpoint_file) and not self.args.eval_only:
            self.args.resume = checkpoint_file
        logger.info("Requeuing ", self.args)
        empty_trainer = type(self)(self.args)
        return submitit.helpers.DelayedSubmission(empty_trainer)

    def _setup_gpu_args(self):
        import submitit

        job_env = submitit.JobEnvironment()
        self.args.output_dir = str(self.args.output_dir).replace(
            "%j", str(job_env.job_id)
        )
        self.args.log_dir = self.args.output_dir
        self.args.job_id = str(job_env.job_id)
        self.args.gpu = job_env.local_rank
        self.args.rank = job_env.global_rank
        self.args.world_size = job_env.num_tasks
        logger.info(
            f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}"
        )


def main():
    args = parse_args()
    if args.job_dir == "":
        args.job_dir = str(get_shared_folder() / "%j")

    # Note that the folder will depend on the job_id, to easily track experiments
    executor = submitit.AutoExecutor(folder=args.job_dir,
                                     slurm_max_num_timeout=30)

    num_gpus_per_node = args.ngpus
    nodes = args.nodes
    timeout_min = args.timeout

    partition = args.partition
    exclude = args.exclude
    kwargs = {}
    if len(args.constraint):
        kwargs["slurm_constraint"] = args.constraint
    if args.comment:
        kwargs["slurm_comment"] = args.comment
    if args.qos:
        kwargs["slurm_qos"] = args.qos
    if args.account:
        kwargs["slurm_account"] = args.account

    executor.update_parameters(
        mem_gb=32 * num_gpus_per_node,
        gpus_per_node=num_gpus_per_node,
        tasks_per_node=num_gpus_per_node,  # one task per GPU
        cpus_per_task=10,
        nodes=nodes,
        timeout_min=timeout_min,  # max is 60 * 72
        # Below are cluster dependent parameters
        slurm_partition=partition,
        slurm_signal_delay_s=120,
        slurm_exclude=exclude,
        **kwargs,
    )

    executor.update_parameters(name="atmos_training")

    args.dist_url = get_init_file().as_uri()
    args.output_dir = args.job_dir

    trainer = Trainer(args)
    job = executor.submit(trainer)

    # print("Submitted job_id:", job.job_id)
    logger.info(f"Submitted job {job.job_id}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
