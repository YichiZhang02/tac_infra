import h5py
import pkg_resources
from . import dmSDK
from .CameraDeviceManager import CameraDeviceManager
import numpy as np
import cv2
__version__ = pkg_resources.get_distribution("dmrobotics").version

import logging
import pickle
import logging
from pathlib import Path

class Sensor:
    def __init__(self, dev_id, roi = np.array([[133, 112], [493, 110], [497, 370], [136, 388]],dtype="float32"), width=640,height=480 , KEEP_FPS_Print = False, base_frame = None) -> None:
        self.hardware = dmSDK.DMV1(dev_id,roi = roi, width=width,height=height,KEEP_FPS_Print = KEEP_FPS_Print)
        status = self.getStatus()
        logging.info(f"Sensor Status is: {status}")
        if base_frame:
            img = load_object(base_frame)
            self.hardware.setBaseFrame(img)

    def reset(self):
        self.hardware.reset()

    def getBaseFrame(self, save_path: str | Path | None = None, overwrite=False):
        frame = self.hardware.getBaseFrame()
        if frame is None:
            raise RuntimeError("hardware.getFrame() returned None")

        if save_path is not None:
            self.save_object(frame,out_path=save_path, overwrite=overwrite)
        return frame

    def getRawImage(self, save_path: str | Path | None = None, overwrite=False):
        """
        获取一帧；若提供 save_path，则把该帧保存到该文件。
        仅负责保存文件，不创建目录。
        """
        frame = self.hardware.getFrame()
        if frame is None:
            raise RuntimeError("hardware.getFrame() returned None")

        if save_path is not None:
            self.save_object(frame,out_path=save_path, overwrite=overwrite)
        return frame
    
    def save_object(self, obj, out_path: str | Path, *, overwrite: bool = False) -> Path:
        """
        整体保存对象为一个二进制文件（pickle）。不创建目录；默认不覆盖。
        """
        p = Path(out_path)
        parent = p.parent if str(p.parent) != "" else Path(".")
        if not (parent.exists() and parent.is_dir()):
            raise FileNotFoundError(f"Parent directory does not exist: {parent}")
        if p.exists() and not overwrite:
            raise FileExistsError(f"File already exists: {p}")

        with open(p, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        return p


    def setH5(self, path,
                rdcc_nbytes: int = 64 * 1024**2,
                rdcc_nslots: int = 1_000_003,
                rdcc_w0: float = 0.75):
        
        return self.hardware.setH5(path,
                rdcc_nbytes,
                rdcc_nslots,
                rdcc_w0)

    def getFeatH5(self, indx, getdepth: bool = False, getshear: bool = False):
        return self.hardware.getFeatH5(indx, getdepth, getshear)
    
    def getFeat(self,img,getdepth=False,getshear=False):
        return self.hardware.getFeat(img,getdepth,getshear)
    
    def getDeformation2D(self):
        return self.hardware.getDeformation()

    def disconnect(self):
        self.hardware.release()

    def getStatus(self):
        return self.hardware.get_sensor_state(print_info=False)
    
    def setBaseFrame(self, img):
        self.hardware.setBaseFrame(img)
        
    def getFeatImage(self):
        return self.hardware.getFeatFrame()

def load_object(path: str | Path):
    """
    读取整个对象；为稳妥起见，读出后再次把 ndarray 设为只读。
    """
    p = Path(path)
    with open(p, "rb") as f:
        obj = pickle.load(f)
    # 保险：确保 img 仍是只读（某些场景下 flags 可能被重置）
    return obj
    
def put_arrows_on_image(image, arrows, scale = 1.0):
    image = image.copy()

    scaled_flow = arrows * scale  # scale factor

    # Get start and end coordinates
    flow_start = np.stack(
        np.meshgrid(range(0, scaled_flow.shape[1], scaled_flow.shape[1]//15),
                    range(0, scaled_flow.shape[0], scaled_flow.shape[0]//15)), 2)
    
    flow_end = (scaled_flow[flow_start[:, :, 1], flow_start[:, :, 0], :] +
            flow_start).astype(np.int32)

    norm = np.linalg.norm(scaled_flow[flow_start[:, :, 1], flow_start[:, :,
                                                                    0], :],
                        axis=2)
    # print(norm.max(), norm.min())
    nz = np.nonzero(norm)

    norm = np.asarray(norm / (scaled_flow.shape[0]/30) * 255.0, dtype='uint8')
    for i in range(len(nz[0])):
        y, x = nz[0][i], nz[1][i]
        cv2.arrowedLine(image,
                        pt1=tuple(flow_start[y, x]),
                        pt2=tuple(flow_end[y, x]),
                        color=(0,255,0),
                        thickness=1,
                        tipLength=.3)
    return image


# --- 初始化：创建文件，写入第一张图像并记录 serial ---
def init_h5(path: str, first_img, roi, 
            compression: str = "gzip", gzip_level: int = 4,
            use_shuffle: bool = True, use_fletcher32: bool = True) -> None:
    img = first_img.img
    H, W= img.shape

    with h5py.File(path, "w") as f:
        # 记录元信息与唯一 serial
        f.attrs["class"] = "DMTacImageSet"
        f.attrs["version"] = 1
        f.attrs["roi"] = roi
        ds_kwargs = dict(chunks=(1, H, W))
        if use_shuffle:
            ds_kwargs["shuffle"] = True
        if use_fletcher32:
            ds_kwargs["fletcher32"] = True
        if compression == "gzip":
            ds_kwargs["compression"] = "gzip"
            ds_kwargs["compression_opts"] = int(gzip_level)
        elif compression == "lzf":
            ds_kwargs["compression"] = "lzf"
        elif compression is None:
            pass
        else:
            raise ValueError("compression 仅支持 'gzip'/'lzf'/None")

        # 可扩展数据集：首维为样本数
        dset = f.create_dataset(
            "images",
            shape=(1, H, W),
            maxshape=(None, H, W),
            dtype=np.uint8,
            **ds_kwargs
        )
        dset[0] = img  # 写入首张

# --- 追加：校验 serial 相同 & 形状相同，合规才写 ---
def append_h5(path: str, dmimg) -> int:
    """
    追加一张图像；返回写入后的样本总数。
    若 serial 与文件记录不同，抛 ValueError。
    若形状或 dtype 不匹配，抛 ValueError/TypeError。
    """
    img = dmimg.img

    with h5py.File(path, "a") as f:

        if "images" not in f:
            raise RuntimeError("文件缺少数据集 'images'，不是正确的初始化文件。")
        dset = f["images"]

        # 形状检查（除样本维以外必须完全一致）
        if img.shape != dset.shape[1:]:
            raise ValueError(f"图像形状不匹配：传入 {img.shape} ≠ 文件 {dset.shape[1:]}")

        # 追加写
        n = dset.shape[0]
        dset.resize((n + 1, ) + dset.shape[1:])
        dset[n] = img
        return n + 1


def listConnectedDevIDs():
    ConnectedDevIDlist = CameraDeviceManager()
    DevIDlist = ConnectedDevIDlist.find_devices()
    print(DevIDlist)
    return DevIDlist