_base_ = './yolox_tiny_8xb8-300e_coco.py'

# ClearML + TensorBoard 可視化を拾いやすくするための設定
# TensorBoard のイベントファイルを出力し、ClearML(Task.init(auto_connect_frameworks=True))
# により自動で取り込ませます。

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
]
visualizer = dict(type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# ログの出力間隔を軽く短縮（必要に応じて調整）
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=10),
)

