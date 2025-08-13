# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

from mmdet.utils import setup_cache_size_limit_of_dynamo


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='enable automatically scaling LR.')
    parser.add_argument(
        '--resume',
        nargs='?',
        type=str,
        const='auto',
        help='If specify checkpoint path, resume from it, while if not '
        'specify, try to auto resume from the latest checkpoint '
        'in the work directory.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    # ClearML optional logging
    parser.add_argument(
        '--clearml',
        action='store_true',
        help='Enable ClearML logging (Task.init) if clearml is installed')
    parser.add_argument(
        '--clearml-project',
        type=str,
        default='mmdetection',
        help='ClearML project name when --clearml is enabled')
    parser.add_argument(
        '--clearml-task',
        type=str,
        default=None,
        help='ClearML task name when --clearml is enabled (default: config basename + "-train")')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    # Allowlist classes for PyTorch 2.6+ safe loading (weights_only=True)
    # to avoid UnpicklingError when checkpoints include mmengine objects.
    try:
        import torch
        ts = torch.serialization  # type: ignore[attr-defined]
        allow = []
        # mmengine objects possibly stored in checkpoints
        try:
            from mmengine.logging.history_buffer import HistoryBuffer  # type: ignore
            allow.append(HistoryBuffer)
        except Exception:
            pass
        try:
            # MessageHub module path differs between versions
            from mmengine.logging.message_hub import MessageHub  # type: ignore
            allow.append(MessageHub)
        except Exception:
            try:
                from mmengine.logging.messagehub import MessageHub  # type: ignore
                allow.append(MessageHub)
            except Exception:
                pass
        # NumPy reconstruct helpers sometimes appear in pickled state
        try:
            import numpy as np  # noqa: F401
            from numpy.core.multiarray import _reconstruct as np_reconstruct  # type: ignore
            from numpy import ndarray as np_ndarray  # type: ignore
            from numpy import dtype as np_dtype  # type: ignore
            allow.extend([np_reconstruct, np_ndarray, np_dtype])
        except Exception:
            pass
        if hasattr(ts, 'add_safe_globals') and allow:
            ts.add_safe_globals(allow)  # type: ignore[attr-defined]
    except Exception as e:
        print(f'[TorchSafeGlobals] Skipped registering safe globals: {e}')
    args = parse_args()

    # Reduce the number of repeated compilations and improve
    # training speed.
    setup_cache_size_limit_of_dynamo()

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # Initialize ClearML task if requested
    if getattr(args, 'clearml', False):
        try:
            from clearml import Task  # type: ignore
            cfg_base = osp.splitext(osp.basename(args.config))[0]
            # Compose task name with dataset/model/tag if available
            _dataset = os.environ.get('DATASET')
            _model = os.environ.get('MODEL')
            _tag = os.environ.get('TAG')
            default_name = f"train:{cfg_base}"
            parts = []
            if _dataset: parts.append(_dataset)
            if _model: parts.append(_model)
            if _tag: parts.append(_tag)
            if parts:
                default_name = default_name + " [" + "/".join(parts) + "]"
            task_name = args.clearml_task or default_name

            task = Task.init(project_name=args.clearml_project, task_name=task_name, auto_connect_frameworks=True)
            # Optionally connect the config for reproducibility
            try:
                task.connect(cfg.to_dict() if hasattr(cfg, 'to_dict') else dict(cfg))
            except Exception:
                pass
            # Add helpful tags for filtering in ClearML
            try:
                tags = []
                for key in ('STAGE', 'DATASET', 'MODEL', 'TAG', 'CONFIG_BASE', 'DEVICE'):
                    v = os.environ.get(key)
                    if v:
                        tags.append(f"{key.lower()}:{v}")
                if tags:
                    task.add_tags(tags)
            except Exception:
                pass
        except Exception as e:
            print(f'[ClearML] Skipped initializing ClearML Task: {e}')

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # enable automatic-mixed-precision training
    if args.amp is True:
        cfg.optim_wrapper.type = 'AmpOptimWrapper'
        cfg.optim_wrapper.loss_scale = 'dynamic'

    # enable automatically scaling LR
    if args.auto_scale_lr:
        if 'auto_scale_lr' in cfg and \
                'enable' in cfg.auto_scale_lr and \
                'base_batch_size' in cfg.auto_scale_lr:
            cfg.auto_scale_lr.enable = True
        else:
            raise RuntimeError('Can not find "auto_scale_lr" or '
                               '"auto_scale_lr.enable" or '
                               '"auto_scale_lr.base_batch_size" in your'
                               ' configuration file.')

    # resume is determined in this priority: resume from > auto_resume
    if args.resume == 'auto':
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # start training
    runner.train()


if __name__ == '__main__':
    main()
