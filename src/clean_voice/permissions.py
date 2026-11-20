"""Whether macOS lets Clean Voice use the microphone."""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
import sys

log = logging.getLogger(__name__)


def _authorization_status() -> int:
    """``[AVCaptureDevice authorizationStatusForMediaType:AVMediaTypeAudio]``: 0 not decided, 1 restricted, 2 denied,
    3 allowed. A plain objc_msgSend call through ctypes, so no extra dependency (macOS only)."""
    ctypes.cdll.LoadLibrary("/System/Library/Frameworks/AVFoundation.framework/AVFoundation")
    objc = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc") or "libobjc.dylib")
    objc.objc_getClass.restype = ctypes.c_void_p
    objc.objc_getClass.argtypes = [ctypes.c_char_p]
    objc.sel_registerName.restype = ctypes.c_void_p
    objc.sel_registerName.argtypes = [ctypes.c_char_p]
    send = objc.objc_msgSend
    # AVMediaTypeAudio is the string "soun".
    send.restype = ctypes.c_void_p
    send.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p]
    media = send(objc.objc_getClass(b"NSString"), objc.sel_registerName(b"stringWithUTF8String:"), b"soun")
    send.restype = ctypes.c_long
    send.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    selector = objc.sel_registerName(b"authorizationStatusForMediaType:")
    return int(send(objc.objc_getClass(b"AVCaptureDevice"), selector, media))


def mic_permission() -> str | None:
    """ "denied" (restricted or denied), "ok", or None (not macOS, or the check failed: unknown never blocks). Checked,
    not prompted, as OBS's CheckPermission (frontend/utility/platform-osx.mm:208-231)."""
    if sys.platform != "darwin":
        return None
    try:
        return "denied" if _authorization_status() in (1, 2) else "ok"
    except Exception:
        log.debug("mic permission check failed", exc_info=True)
        return None
