import os
import re
import platform
import ctypes

class CameraDeviceManager:
    SERIAL_PATTERN = re.compile(r'[LMS]\d{10}')
    TARGET_TOKEN = "N160MU2"

    # ---- V4L2 常量/结构（Linux）----
    _VIDIOC_QUERYCAP = 0x80685600  # from _IOR('V', 0, struct v4l2_capability)
    _V4L2_CAP_VIDEO_CAPTURE         = 0x00000001
    _V4L2_CAP_VIDEO_CAPTURE_MPLANE  = 0x00001000
    _V4L2_CAP_DEVICE_CAPS           = 0x80000000
    _V4L2_CAP_STREAMING             = 0x04000000
    _V4L2_CAP_META_CAPTURE          = 0x00800000  # 仅元数据

    class _v4l2_capability(ctypes.Structure):
        _fields_ = [
            ("driver",     ctypes.c_uint8 * 16),
            ("card",       ctypes.c_uint8 * 32),
            ("bus_info",   ctypes.c_uint8 * 32),
            ("version",    ctypes.c_uint32),
            ("capabilities", ctypes.c_uint32),
            ("device_caps",  ctypes.c_uint32),
            ("reserved",   ctypes.c_uint32 * 3),
        ]

    def __init__(self):
        self.system = platform.system()
        self.index_serial_map = {}

    def _build_index_serial_map(self):
        if self.system == "Linux":
            self.index_serial_map = self._linux_index_serials_filtered()
        elif self.system == "Windows":
            self.index_serial_map = self._windows_index_serials_filtered()
        else:
            self.index_serial_map = {}

    # ========= Linux：只保留真正的 video-capture 节点 =========
    def _linux_index_serials_filtered(self):
        try:
            import pyudev
        except ImportError:
            raise RuntimeError("Linux 下需要: pip install pyudev")

        ctx = pyudev.Context()
        m = {}
        for dev in ctx.list_devices(subsystem="video4linux"):
            node = dev.device_node  # /dev/videoX
            if not node or not node.startswith("/dev/video"):
                continue
            try:
                idx = int(node[10:])  # len('/dev/video')=10
            except ValueError:
                continue

            # 先用 V4L2 能力硬校验：必须是可采集视频的节点
            if not self._v4l2_is_capture_node(node):
                continue

            # 名称校验（Generic_N160MU2 或包含 N160MU2）
            name = (dev.get("ID_V4L_PRODUCT") or dev.get("ID_MODEL") or "").upper()
            if not (("GENERIC_N160MU2" in name) or (self.TARGET_TOKEN in name)):
                continue

            # 序列号提取
            raw = dev.get("ID_SERIAL_SHORT") or dev.get("ID_SERIAL") or dev.get("ID_MODEL_ID") or ""
            serial = self._extract_compliant_serial(raw) or (raw if raw else None)
            m[idx] = serial
        return m

    def _v4l2_is_capture_node(self, devnode: str) -> bool:
        """用 VIDIOC_QUERYCAP 判断是否为可采集视频的节点（排除 meta 节点等）。"""
        try:
            fd = os.open(devnode, os.O_RDONLY | os.O_NONBLOCK)
        except OSError:
            return False
        try:
            import fcntl
        except ImportError:
            raise RuntimeError("Linux need: pip install fcntl")
        try:
            cap = self._v4l2_capability()
            buf = ctypes.create_string_buffer(ctypes.sizeof(cap))
            # ioctl 需要可写缓冲；用 memoryview 指向
            fcntl.ioctl(fd, self._VIDIOC_QUERYCAP, buf, True)
            ctypes.memmove(ctypes.addressof(cap), buf, ctypes.sizeof(cap))

            caps = cap.device_caps if (cap.capabilities & self._V4L2_CAP_DEVICE_CAPS) else cap.capabilities

            # 必须具备 capture 能力；排除仅 meta 节点；最好具备 streaming
            is_capture = bool(caps & (self._V4L2_CAP_VIDEO_CAPTURE | self._V4L2_CAP_VIDEO_CAPTURE_MPLANE))
            is_meta_only = bool(caps & self._V4L2_CAP_META_CAPTURE) and not is_capture
            has_stream = bool(caps & self._V4L2_CAP_STREAMING)
            return is_capture and not is_meta_only and has_stream
        except Exception:
            return False
        finally:
            os.close(fd)


    def _windows_index_serials_filtered(self):
        import ctypes, ctypes.wintypes as wtypes, uuid, re, os, sys

        # ---- ctypes 基础类型/常量 ----
        GUID = ctypes.c_byte * 16
        HDEVINFO = wtypes.HANDLE
        DEVINST = wtypes.DWORD
        DEVPROPTYPE = wtypes.ULONG
        ULONG = wtypes.ULONG
        ULONG_PTR = getattr(wtypes, "ULONG_PTR", wtypes.WPARAM)

        SPDRP_DEVICEDESC   = 0x00000000
        SPDRP_FRIENDLYNAME = 0x0000000C
        SPDRP_SERVICE      = 0x00000004
        DIGCF_PRESENT      = 0x00000002
        DIGCF_ALLCLASSES   = 0x00000004
        CR_SUCCESS         = 0x00000000

        class DEVPROPKEY(ctypes.Structure):
            _fields_ = [("fmtid", GUID), ("pid", wtypes.DWORD)]
        class SP_DEVINFO_DATA(ctypes.Structure):
            _fields_ = [("cbSize", wtypes.DWORD), ("ClassGuid", GUID),
                        ("DevInst", DEVINST), ("Reserved", ULONG_PTR)]

        def _guid(s: str) -> GUID:
            return GUID.from_buffer_copy(uuid.UUID(s.strip("{}")).bytes_le)

        DEVPKEY_Device_FriendlyName = DEVPROPKEY(_guid("A45C254E-DF1C-4EFD-8020-67D146A850E0"), 14)

        setupapi = ctypes.windll.setupapi
        cfgmgr32 = ctypes.windll.CfgMgr32

        SetupDiGetClassDevsW              = setupapi.SetupDiGetClassDevsW
        SetupDiEnumDeviceInfo             = setupapi.SetupDiEnumDeviceInfo
        SetupDiDestroyDeviceInfoList      = setupapi.SetupDiDestroyDeviceInfoList
        SetupDiGetDeviceRegistryPropertyW = setupapi.SetupDiGetDeviceRegistryPropertyW
        SetupDiGetDevicePropertyW         = setupapi.SetupDiGetDevicePropertyW
        CM_Get_Parent                     = cfgmgr32.CM_Get_Parent
        CM_Get_Device_IDW                 = cfgmgr32.CM_Get_Device_IDW

        # restype/argtypes
        SetupDiGetClassDevsW.restype              = HDEVINFO
        SetupDiEnumDeviceInfo.restype             = wtypes.BOOL
        SetupDiDestroyDeviceInfoList.restype      = wtypes.BOOL
        SetupDiGetDeviceRegistryPropertyW.restype = wtypes.BOOL
        SetupDiGetDevicePropertyW.restype         = wtypes.BOOL
        CM_Get_Parent.restype                     = wtypes.DWORD
        CM_Get_Device_IDW.restype                 = wtypes.DWORD

        SetupDiGetClassDevsW.argtypes = [ctypes.POINTER(GUID), wtypes.LPCWSTR, wtypes.HWND, wtypes.DWORD]
        SetupDiEnumDeviceInfo.argtypes = [HDEVINFO, wtypes.DWORD, ctypes.POINTER(SP_DEVINFO_DATA)]
        SetupDiDestroyDeviceInfoList.argtypes = [HDEVINFO]
        SetupDiGetDeviceRegistryPropertyW.argtypes = [
            HDEVINFO, ctypes.POINTER(SP_DEVINFO_DATA), wtypes.DWORD,
            ctypes.POINTER(wtypes.DWORD), ctypes.POINTER(wtypes.BYTE), wtypes.DWORD, ctypes.POINTER(wtypes.DWORD)]
        SetupDiGetDevicePropertyW.argtypes = [
            HDEVINFO, ctypes.POINTER(SP_DEVINFO_DATA),
            ctypes.POINTER(DEVPROPKEY), ctypes.POINTER(DEVPROPTYPE),
            ctypes.POINTER(wtypes.BYTE), wtypes.DWORD, ctypes.POINTER(wtypes.DWORD), wtypes.DWORD]
        CM_Get_Parent.argtypes = [ctypes.POINTER(DEVINST), DEVINST, ULONG]
        CM_Get_Device_IDW.argtypes = [DEVINST, wtypes.LPWSTR, ULONG, ULONG]

        # ---------- 读取属性/父链/实例ID ----------
        def _get_reg_property_str(hdev, info, prop):
            data_type = wtypes.DWORD(); req = wtypes.DWORD(0)
            SetupDiGetDeviceRegistryPropertyW(hdev, ctypes.byref(info), prop, ctypes.byref(data_type), None, 0, ctypes.byref(req))
            if req.value == 0: return None
            buf = (wtypes.BYTE * req.value)()
            if not SetupDiGetDeviceRegistryPropertyW(hdev, ctypes.byref(info), prop, ctypes.byref(data_type), buf, req, ctypes.byref(req)):
                return None
            return ctypes.wstring_at(ctypes.cast(buf, ctypes.c_wchar_p))

        def _get_devprop_str(hdev, info, key):
            prop_type = DEVPROPTYPE(); req = wtypes.DWORD(0)
            SetupDiGetDevicePropertyW(hdev, ctypes.byref(info), ctypes.byref(key), ctypes.byref(prop_type), None, 0, ctypes.byref(req), 0)
            if req.value == 0: return None
            buf = (wtypes.BYTE * req.value)()
            if not SetupDiGetDevicePropertyW(hdev, ctypes.byref(info), ctypes.byref(key), ctypes.byref(prop_type), buf, req, ctypes.byref(req), 0):
                return None
            return ctypes.wstring_at(ctypes.cast(buf, ctypes.c_wchar_p))

        def _cm_get_device_id(devinst):
            buf = ctypes.create_unicode_buffer(1024)
            return buf.value if CM_Get_Device_IDW(devinst, buf, 1024, 0) == CR_SUCCESS else None

        def _cm_get_parent(devinst):
            parent = DEVINST()
            return parent if CM_Get_Parent(ctypes.byref(parent), devinst, 0) == CR_SUCCESS else None

        # ---------- DLL：加载 & 查询 index（只用 child 尾段 token） ----------
        def _load_cvindex_dll():
            import os
            import ctypes
            import sys
            from pathlib import Path

            # 根据当前进程位数选择 DLL 名
            is64 = (ctypes.sizeof(ctypes.c_void_p) == 8)
            name = "CvCameraIndex_x64.dll" if is64 else "CvCameraIndex_x86.dll"

            # 1) 优先：本文件所在目录（而非当前工作目录）
            base_dir = Path(__file__).resolve().parent
            # 2) PyInstaller 打包后可执行的临时解包目录
            frozen_dir = Path(getattr(sys, "_MEIPASS", base_dir))

            # 搜索候选路径（由近及远）
            candidates = [
                base_dir / "lib" / name,     # 同目录/lib/xxx.dll
                base_dir / name,             # 同目录/xxx.dll
                frozen_dir / "lib" / name,   # 打包场景
                frozen_dir / name,
                Path(os.getcwd()) / name,    # 兼容你之前的做法
            ]

            last_err = None
            for p in candidates:
                try:
                    if p.exists():
                        # Py3.8+：把目录加入 DLL 搜索路径，便于解析其依赖
                        if hasattr(os, "add_dll_directory"):
                            os.add_dll_directory(str(p.parent))
                        dll = ctypes.CDLL(str(p))  # C++ 默认 cdecl；若是 stdcall 可改用 ctypes.WinDLL
                        dll.getCameraIndex.argtypes = [ctypes.c_char_p]
                        dll.getCameraIndex.restype  = ctypes.c_int
                        return dll
                except OSError as e:
                    last_err = e
                    continue
            return None

        def _dll_index_by_child_tail(dll, child_inst: str) -> int:
            if not dll or not child_inst:
                return -1
            # 取 child 实例的最后一段作为 token（例如 6&2A49E962&0&0000）
            if "\\" in child_inst:
                token = child_inst.split("\\")[-1]
            else:
                token = child_inst
            try:
                return int(dll.getCameraIndex(token.encode("utf-8")))
            except Exception:
                return -1

        # ---------- 序列号：只用 parent 尾段（如 M2505150237），找不到就返回 None ----------
        SERIAL_PAT = re.compile(r'[LMS]\d{10}', re.I)
        def _serial_from_parents(devinst):
            # 从 child 开始沿父链找，取第一个符合 L/M/S+10 或者 USB\VID_xxxx&PID_xxxx\<TAIL> 的 TAIL
            cur = devinst
            for _ in range(8):
                inst = _cm_get_device_id(cur)
                if inst:
                    tail = inst.split("\\")[-1]
                    m = SERIAL_PAT.search(tail)
                    if m:
                        return m.group(0).upper()
                    # 若是 USB\VID_...&PID_...\SERIAL 这种，也可接受
                    if "\\" in inst and "VID_" in inst.upper() and "&PID_" in inst.upper():
                        if tail and not tail.upper().startswith(("MI_", "&MI_", "6&")):
                            return tail
                p = _cm_get_parent(cur)
                if not p or p == cur:
                    break
                cur = p
            return None  # 找不到就 None

        # ---------- 主流程：枚举 usbvideo -> 过滤型号 -> DLL 取 index ----------
        mapping = {}
        dll = _load_cvindex_dll()
        if not dll:
            # 没有 DLL 无法可靠得到 OpenCV index
            return mapping

        h = SetupDiGetClassDevsW(None, None, None, DIGCF_PRESENT | DIGCF_ALLCLASSES)
        if h == wtypes.HANDLE(-1).value:
            return mapping

        try:
            i = 0
            info = SP_DEVINFO_DATA(); info.cbSize = ctypes.sizeof(SP_DEVINFO_DATA)
            while SetupDiEnumDeviceInfo(h, i, ctypes.byref(info)):
                i += 1
                service = _get_reg_property_str(h, info, SPDRP_SERVICE)
                if (service or "").lower() != "usbvideo":
                    continue

                # 型号过滤（例如 N160MU2），不需要可去掉此判断
                friendly = (_get_devprop_str(h, info, DEVPKEY_Device_FriendlyName)
                            or _get_reg_property_str(h, info, SPDRP_FRIENDLYNAME)
                            or _get_reg_property_str(h, info, SPDRP_DEVICEDESC)
                            or "")
                if self.TARGET_TOKEN not in (friendly or "").upper():
                    continue

                # 1) child 实例路径（用于 DLL 查 index）
                child_inst = _cm_get_device_id(info.DevInst) or ""
                idx_found = _dll_index_by_child_tail(dll, child_inst)

                if idx_found < 0:
                    continue  # 没法确定 OpenCV index，跳过

                # 2) parent 链上取序列号（用于 serial 映射）
                serial = _serial_from_parents(info.DevInst)
                serial = self._extract_compliant_serial(serial) or serial  # 最终规整一下

                mapping[idx_found] = serial
        finally:
            SetupDiDestroyDeviceInfoList(h)

        return mapping

    def find_devices(self, target_index=None, target_serial_suffix=None,
                     require_compliant=True, verify=False, max_probe=20,
                     prefer_dshow=True):
        self._build_index_serial_map()

        def ok_serial(s):
            if s is None:
                return not require_compliant
            if require_compliant and not self.SERIAL_PATTERN.fullmatch(s):
                return False
            if target_serial_suffix and not str(s).endswith(target_serial_suffix):
                return False
            return True

        if target_index is not None:
            s = self.index_serial_map.get(target_index)
            if s is None and require_compliant:
                return []
            if not ok_serial(s):
                return []
            results = [{'device_index': target_index, 'serial': s}]
        else:
            results = [{'device_index': i, 'serial': s}
                       for i, s in sorted(self.index_serial_map.items())
                       if ok_serial(s)]
        return results
    

    # ========= 帮助 =========
    def _extract_compliant_serial(self, raw):
        if not raw:
            return None
        m = self.SERIAL_PATTERN.search(str(raw).upper())
        return m.group(0) if m else None


