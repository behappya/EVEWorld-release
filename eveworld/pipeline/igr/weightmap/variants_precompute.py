#!/usr/bin/env python3
"""T4G-WEIGHTMAP-VARIANTS-PRECOMPUTE: 六变体权重图缓存预计算 (纯查表, 无 GPU/无检测)。

写档位原值到 <root>/weightmap_cache_<variant>/<vid>.npy; 与 eveworld/pipeline/igr/weightmap/precompute.py 互不影响。
"""
from __future__ import annotations

import argparse
import os
import sys

try:                                     # 包内导入
    from eveworld.pipeline.igr.weightmap import variants as V
except ImportError:                      # 直接以脚本运行
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import variants as V


def main(argv=None):
    ap = argparse.ArgumentParser(description='六变体 IGR 空间权重图缓存预计算')
    ap.add_argument('--variant', default='all',
                    help=f'变体名或 all (默认 all); 可选: {", ".join(V.VARIANT_NAMES)}')
    ap.add_argument('--root', help=f'缓存根目录, 默认 ${V.ROOT_ENV} 或 {V.PROBE}')
    ap.add_argument('--out-dir', help='单个变体的精确输出目录 (只对单个 --variant 有效)')
    ap.add_argument('--vids', help='逗号分隔的 vid 子集, 默认 anno 目录全部')
    ap.add_argument('--normalize', action='store_true',
                    help='写归一化到均值 1 的图 (仅离线分析; 不满足训练侧 wmap_values 契约)')
    ap.add_argument('--anno-dir', help='覆盖 anno 目录')
    ap.add_argument('--assets-dir', help='覆盖 aug_assets 目录')
    ap.add_argument('--gripper-dir', help='覆盖 gripper_anno 目录')
    args = ap.parse_args(argv)

    V._check_registry()
    names = V.VARIANT_NAMES if args.variant == 'all' else (args.variant,)
    for name in names:
        V.variant_spec(name)                                   # 未知名 -> 明确报错
    if args.out_dir and len(names) != 1:
        ap.error('--out-dir 只能配合单个 --variant 使用')
    vids = [v for v in args.vids.split(',') if v] if args.vids else None
    if args.normalize:
        print('警告: --normalize 写出的图不满足训练侧 wmap_values / region 账本契约, '
              '仅用于离线分析。')

    for name in names:
        V.precompute_variant(name, root=args.root, out_dir=args.out_dir, vids=vids,
                             normalize=args.normalize, anno_dir=args.anno_dir,
                             assets_dir=args.assets_dir, gripper_dir=args.gripper_dir)
    return 0


if __name__ == '__main__':
    sys.exit(main())
