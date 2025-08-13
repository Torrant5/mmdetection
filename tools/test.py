# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import warnings
from copy import deepcopy

from mmengine import ConfigDict
from mmengine.config import Config, DictAction
from mmengine.runner import Runner

from mmdet.engine.hooks.utils import trigger_visualization_hook
from mmdet.evaluation import DumpDetResults
from mmdet.registry import RUNNERS
from mmdet.utils import setup_cache_size_limit_of_dynamo


# TODO: support fuse_conv_bn and format_only
def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument(
        '--out',
        type=str,
        help='dump predictions to a pickle file for offline evaluation')
    parser.add_argument(
        '--show', action='store_true', help='show prediction results')
    parser.add_argument(
        '--show-dir',
        help='directory where painted images will be saved. '
        'If specified, it will be automatically saved '
        'to the work_dir/timestamp/show_dir')
    parser.add_argument(
        '--wait-time', type=float, default=2, help='the interval of show (s)')
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
    parser.add_argument('--tta', action='store_true')
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
        help='ClearML task name when --clearml is enabled (default: config basename + "-test")')
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
            # types can also be allowlisted
            from numpy import ndarray as np_ndarray  # type: ignore
            from numpy import dtype as np_dtype  # type: ignore
            allow.extend([np_reconstruct, np_ndarray, np_dtype])
            try:
                from numpy.dtypes import Float64DType as np_Float64DType  # type: ignore
                allow.append(np_Float64DType)
            except Exception:
                pass
        except Exception:
            pass
        if hasattr(ts, 'add_safe_globals') and allow:
            ts.add_safe_globals(allow)  # type: ignore[attr-defined]
    except Exception as e:
        print(f'[TorchSafeGlobals] Skipped registering safe globals: {e}')
    args = parse_args()

    # Reduce the number of repeated compilations and improve
    # testing speed.
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
            default_name = f"test:{cfg_base}"
            parts = []
            if _dataset: parts.append(_dataset)
            if _model: parts.append(_model)
            if _tag: parts.append(_tag)
            if parts:
                default_name = default_name + " [" + "/".join(parts) + "]"
            task_name = args.clearml_task or default_name

            task = Task.init(project_name=args.clearml_project, task_name=task_name, auto_connect_frameworks=True)
            try:
                from mmengine import Config as _Cfg
                cfg_dict = cfg.to_dict() if isinstance(cfg, _Cfg) or hasattr(cfg, 'to_dict') else dict(cfg)
            except Exception:
                cfg_dict = dict()
            try:
                task.connect(cfg_dict)
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

    cfg.load_from = args.checkpoint

    if args.show or args.show_dir:
        cfg = trigger_visualization_hook(cfg, args)

    if args.tta:

        if 'tta_model' not in cfg:
            warnings.warn('Cannot find ``tta_model`` in config, '
                          'we will set it as default.')
            cfg.tta_model = dict(
                type='DetTTAModel',
                tta_cfg=dict(
                    nms=dict(type='nms', iou_threshold=0.5), max_per_img=100))
        if 'tta_pipeline' not in cfg:
            warnings.warn('Cannot find ``tta_pipeline`` in config, '
                          'we will set it as default.')
            test_data_cfg = cfg.test_dataloader.dataset
            while 'dataset' in test_data_cfg:
                test_data_cfg = test_data_cfg['dataset']
            cfg.tta_pipeline = deepcopy(test_data_cfg.pipeline)
            flip_tta = dict(
                type='TestTimeAug',
                transforms=[
                    [
                        dict(type='RandomFlip', prob=1.),
                        dict(type='RandomFlip', prob=0.)
                    ],
                    [
                        dict(
                            type='PackDetInputs',
                            meta_keys=('img_id', 'img_path', 'ori_shape',
                                       'img_shape', 'scale_factor', 'flip',
                                       'flip_direction'))
                    ],
                ])
            cfg.tta_pipeline[-1] = flip_tta
        cfg.model = ConfigDict(**cfg.tta_model, module=cfg.model)
        cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # add `DumpResults` dummy metric
    if args.out is not None:
        assert args.out.endswith(('.pkl', '.pickle')), \
            'The dump file must be a pkl file.'
        runner.test_evaluator.metrics.append(
            DumpDetResults(out_file_path=args.out))

    # start testing
    runner.test()


if __name__ == '__main__':
    main()
