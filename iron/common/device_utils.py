# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import aie.utils as aie_utils
from aie.iron.device import NPU2


def get_kernel_dir(dev=None) -> str:
    """Returns 'aie2p' for NPU2 (Strix, Krackan), 'aie2' for NPU1 (Phoenix).

    mlir-aie ships the same mapping as
    ``aie.utils.compile.utils.resolve_target_arch``, which reads the device's
    AIE arch rather than isinstance-testing NPU2 (so it also rejects an
    unsupported arch instead of quietly answering 'aie2').  Once
    requirements.txt pins a wheel carrying it, this becomes
    ``resolve_target_arch(dev or aie_utils.get_current_device())`` -- keeping
    the "default to the current device" behaviour, which upstream does not have.
    """
    if dev is None:
        dev = aie_utils.get_current_device()
    return "aie2p" if isinstance(dev, NPU2) else "aie2"
