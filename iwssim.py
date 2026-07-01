#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "netpbmfile>=2026.1.29",
#     "numpy>=2.5.0",
#     "pillow>=12.2.0",
#     "pyrtools>=1.0.10",
#     "torch>=2.12.1",
# ]
# ///

from torch._tensor import Tensor
from numpy import ndarray
from numpy import generic
from numpy import dtype
import argparse

import numpy as np
import pyrtools as pt
import torch
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError


class IW_SSIM:
    def __init__(
        self,
        iw_flag: bool = True,
        Nsc: int = 5,
        blSzX: int = 3,
        blSzY: int = 3,
        parent: bool = True,
        sigma_nsq: float = 0.4,
        use_cuda: bool = False,
        use_double: bool = False,
    ) -> None:
        # MS-SSIM parameters
        self.K = [0.01, 0.03]
        self.L = 255
        self.weight = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
        self.winsize = 11
        self.sigma = 1.5

        # IW-SSIM parameters
        self.iw_flag = iw_flag
        self.Nsc = Nsc  # scales
        self.blSzX = blSzX  # Neighbor size
        self.blSzY = blSzY
        self.parent = parent
        self.sigma_nsq = sigma_nsq

        self.bound = np.ceil((self.winsize - 1) / 2)
        self.bound1 = self.bound - np.floor((self.blSzX - 1) / 2)
        self.use_cuda = use_cuda
        self.use_double = use_double

        self.samplet = torch.tensor([1.0])
        if self.use_cuda:
            self.samplet = self.samplet.cuda()
        if self.use_double:
            self.samplet = self.samplet.double()
        self.samplen = np.array([1.0])
        if not self.use_double:
            self.samplen = self.samplen.astype("float32")

    def fspecial(self, fltr: str, ws, **kwargs):
        if fltr == "uniform":
            return np.ones((ws, ws)) / ws**2

        elif fltr == "gaussian":
            x, y = np.mgrid[-ws // 2 + 1 : ws // 2 + 1, -ws // 2 + 1 : ws // 2 + 1]
            g = np.exp(-((x**2 + y**2) / (2.0 * kwargs["sigma"] ** 2)))
            g[g < np.finfo(g.dtype).eps * g.max()] = 0
            assert g.shape == (ws, ws)
            den = g.sum()
            if den != 0:
                g /= den
            return g

        return None

    def get_pyrd(self, imgo, imgd) -> tuple[dict[int, Tensor], dict[int, Tensor]]:
        imgopr = {}
        imgdpr = {}
        lpo = pt.pyramids.LaplacianPyramid(imgo, height=5)
        lpd = pt.pyramids.LaplacianPyramid(imgd, height=5)
        for scale in range(1, self.Nsc + 1):
            imgopr[scale] = (
                torch.from_numpy(lpo.pyr_coeffs[(scale - 1, 0)])
                .unsqueeze(0)
                .unsqueeze(0)
                .type(self.samplet.type())
            )
            imgdpr[scale] = (
                torch.from_numpy(lpd.pyr_coeffs[(scale - 1, 0)])
                .unsqueeze(0)
                .unsqueeze(0)
                .type(self.samplet.type())
            )

        return imgopr, imgdpr

    def scale_qualty_maps(
        self, imgopr: dict[int, Tensor], imgdpr: dict[int, Tensor]
    ) -> tuple[Tensor, dict[int, Tensor]]:

        ms_win = self.fspecial("gaussian", ws=self.winsize, sigma=self.sigma)
        ms_win = (
            torch.from_numpy(ms_win).unsqueeze(0).unsqueeze(0).type(self.samplet.type())
        )
        C1 = (self.K[0] * self.L) ** 2
        C2 = (self.K[1] * self.L) ** 2
        cs_map = {}
        for i in range(1, self.Nsc + 1):
            imgo = imgopr[i]
            imgd = imgdpr[i]
            mu1 = F.conv2d(imgo, ms_win)
            mu2 = F.conv2d(imgd, ms_win)
            sigma12 = F.conv2d(imgo * imgd, ms_win) - mu1 * mu2
            sigma1_sq = F.conv2d(imgo**2, ms_win) - mu1 * mu1
            sigma2_sq = F.conv2d(imgd**2, ms_win) - mu2 * mu2
            sigma1_sq = torch.max(
                torch.zeros(sigma1_sq.shape).type(self.samplet.type()), sigma1_sq
            )
            sigma2_sq = torch.max(
                torch.zeros(sigma2_sq.shape).type(self.samplet.type()), sigma2_sq
            )
            cs_map[i] = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
            if i == self.Nsc:
                l_map = (2 * mu1 * mu2 + C1) / (mu1**2 + mu2**2 + C1)

        return l_map, cs_map

    def roll(self, x: Tensor, shift: int, dim: int) -> Tensor:
        if dim == 0:
            return torch.cat((x[-shift:, :], x[:-shift, :]), dim)
        else:
            return torch.cat((x[:, -shift:], x[:, :-shift]), dim)

    def imenlarge2(self, im) -> Tensor:
        _, _, M, N = im.shape
        t1 = F.interpolate(
            im,
            size=(int(4 * M - 3), int(4 * N - 3)),
            mode="bilinear",
            align_corners=False,
        )
        t2 = torch.zeros([1, 1, 4 * M - 1, 4 * N - 1]).type(self.samplet.type())
        t2[:, :, 1:-1, 1:-1] = t1
        t2[:, :, 0, :] = 2 * t2[:, :, 1, :] - t2[:, :, 2, :]
        t2[:, :, -1, :] = 2 * t2[:, :, -2, :] - t2[:, :, -3, :]
        t2[:, :, :, 0] = 2 * t2[:, :, :, 1] - t2[:, :, :, 2]
        t2[:, :, :, -1] = 2 * t2[:, :, :, -2] - t2[:, :, :, -3]
        imu = t2[:, :, ::2, ::2]

        return imu

    def info_content_weight_map(
        self, imgopr: dict[int, Tensor], imgdpr: dict[int, Tensor]
    ) -> dict[int, Tensor]:

        tol = 1e-15
        iw_map = {}
        for scale in range(1, self.Nsc):
            imgo = imgopr[scale]
            imgd = imgdpr[scale]
            win = np.ones([self.blSzX, self.blSzY])
            win = win / np.sum(win)
            win = (
                torch.from_numpy(win)
                .unsqueeze(0)
                .unsqueeze(0)
                .type(self.samplet.type())
            )
            padding = int((self.blSzX - 1) / 2)

            # Prepare for estimating IW-SSIM parameters
            mean_x = F.conv2d(imgo, win, padding=padding)
            mean_y = F.conv2d(imgd, win, padding=padding)
            cov_xy = F.conv2d(imgo * imgd, win, padding=padding) - mean_x * mean_y
            ss_x = F.conv2d(imgo**2, win, padding=padding) - mean_x**2
            ss_y = F.conv2d(imgd**2, win, padding=padding) - mean_y**2

            ss_x[ss_x < 0] = 0
            ss_y[ss_y < 0] = 0

            # Estimate gain factor and error
            g = cov_xy / (ss_x + tol)
            vv = ss_y - g * cov_xy
            g[ss_x < tol] = 0
            vv[ss_x < tol] = ss_y[ss_x < tol]
            ss_x[ss_x < tol] = 0
            g[ss_y < tol] = 0
            vv[ss_y < tol] = 0

            # Prepare parent band
            aux = imgo
            _, _, Nsy, Nsx = aux.shape
            prnt = self.parent and scale < self.Nsc - 1
            BL = torch.zeros([1, 1, aux.shape[2], aux.shape[3], 1 + prnt])
            if self.use_cuda:
                BL = BL.cuda()
            if self.use_double:
                BL = BL.double()

            BL[:, :, :, :, 0] = aux
            if prnt:
                auxp = imgopr[scale + 1]
                auxp = self.imenlarge2(auxp)
                BL[:, :, :, :, 1] = auxp[:, :, 0:Nsy, 0:Nsx]
            imgo = BL
            _, _, nv, nh, nb = imgo.shape

            block = torch.tensor([win.shape[2], win.shape[3]])
            if self.use_cuda:
                block = block.cuda()

            # Group neighboring pixels
            nblv = nv - block[0] + 1
            nblh = nh - block[1] + 1
            nexp = nblv * nblh
            N = torch.prod(block) + prnt
            Ly = int((block[0] - 1) / 2)
            Lx = int((block[1] - 1) / 2)
            Y = torch.zeros([nexp, N]).type(self.samplet.type())

            n = -1
            for ny in range(-Ly, Ly + 1):
                for nx in range(-Lx, Lx + 1):
                    n = n + 1
                    temp = imgo[0, 0, :, :, 0]
                    foo1 = self.roll(temp, ny, 0)
                    foo = self.roll(foo1, nx, 1)
                    foo = foo[Ly : Ly + nblv, Lx : Lx + nblh]
                    Y[:, n] = foo.flatten()
            if prnt:
                n = n + 1
                temp = imgo[0, 0, :, :, 1]
                foo = temp
                foo = foo[Ly : Ly + nblv, Lx : Lx + nblh]
                Y[:, n] = foo.flatten()

            C_u = torch.mm(torch.transpose(Y, 0, 1), Y) / nexp.type(self.samplet.type())
            eig_values, H = torch.linalg.eigh(C_u)
            eig_values = eig_values.type(self.samplet.type())
            H = H.type(self.samplet.type())
            positive = eig_values > 0
            positive_sum = torch.sum(eig_values * positive.type(self.samplet.type()))
            eig_scale = torch.sum(eig_values) / (positive_sum + (positive_sum == 0))
            if self.use_double:
                L = torch.diag(eig_values * positive.double()) * eig_scale
            else:
                L = torch.diag(eig_values * positive.float()) * eig_scale
            C_u = torch.mm(torch.mm(H, L), torch.transpose(H, 0, 1))
            C_u_inv = torch.inverse(C_u)
            ss = (torch.mm(Y, C_u_inv)) * Y / N.type(self.samplet.type())
            ss = torch.sum(ss, 1)
            ss = ss.view(nblv, nblh)
            ss = ss.unsqueeze(0).unsqueeze(0)
            g = g[:, :, Ly : Ly + nblv, Lx : Lx + nblh]
            vv = vv[:, :, Ly : Ly + nblv, Lx : Lx + nblh]

            # Calculate mutual information
            infow = torch.zeros(g.shape).type(self.samplet.type())
            for j in range(len(eig_values)):
                infow = infow + torch.log2(
                    1
                    + (
                        (vv + (1 + g * g) * self.sigma_nsq) * ss * eig_values[j]
                        + self.sigma_nsq * vv
                    )
                    / (self.sigma_nsq * self.sigma_nsq)
                )
            infow[infow < tol] = 0
            iw_map[scale] = infow

        return iw_map

    def test(self, imgo: ndarray, imgd: ndarray) -> Tensor:

        imgo = imgo.astype(self.samplen.dtype)
        imgd = imgd.astype(self.samplen.dtype)
        imgopr, imgdpr = self.get_pyrd(imgo, imgd)
        l_map, cs_map = self.scale_qualty_maps(imgopr, imgdpr)
        if self.iw_flag:
            iw_map = self.info_content_weight_map(imgopr, imgdpr)

        wmcs = []
        for s in range(1, self.Nsc + 1):
            cs = cs_map[s]
            if s == self.Nsc:
                cs = cs_map[s] * l_map

            if self.iw_flag:
                if s < self.Nsc:
                    iw = iw_map[s]
                    if self.bound1 != 0:
                        iw = iw[
                            :,
                            :,
                            int(self.bound1) : -int(self.bound1),
                            int(self.bound1) : -int(self.bound1),
                        ]
                    else:
                        iw = iw[:, :, int(self.bound1) :, int(self.bound1) :]
                else:
                    iw = torch.ones(cs.shape).type(self.samplet.type())
                wmcs.append(torch.sum(cs * iw) / torch.sum(iw))
            else:
                wmcs.append(torch.mean(cs))

        wmcs = torch.tensor(wmcs).type(self.samplet.type())
        self.weight = torch.tensor(self.weight).type(self.samplet.type())
        score = torch.prod((torch.abs(wmcs)) ** (self.weight))

        return score


def _to_luma_array(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)

    if arr.dtype == np.bool_:
        return np.where(arr, 0.0, 255.0).astype(np.float32, copy=False)

    if arr.ndim == 2:
        return arr

    if arr.ndim == 3:
        if arr.shape[2] == 1:
            return arr[:, :, 0]
        if arr.shape[2] >= 3:
            rgb = arr[:, :, :3].astype(np.float32, copy=False)
            if arr.dtype.kind in "ui":
                rgb = rgb.astype(np.float32)
            return 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]

    raise ValueError(f"unsupported image shape {arr.shape!r}")


def _load_netpbm_image(path: str) -> np.ndarray:
    try:
        import netpbmfile
    except ImportError as exc:
        raise ValueError(
            f"{path}: NetPBM support requires the netpbmfile package"
        ) from exc

    if hasattr(netpbmfile, "NetpbmFile"):
        with netpbmfile.NetpbmFile(path) as netpbm:
            arr = np.asarray(netpbm.asarray())
            maxval = getattr(netpbm, "maxval", None)
    elif hasattr(netpbmfile, "imread"):
        arr = np.asarray(netpbmfile.imread(path))
        maxval = None
    else:
        raise ValueError(
            f"{path}: installed netpbmfile package does not expose a reader"
        )

    if maxval is not None and maxval != 255 and np.issubdtype(arr.dtype, np.integer):
        if maxval > 0:
            arr = arr.astype(np.float32, copy=False) * (255.0 / float(maxval))
    return np.asarray(arr, dtype=np.float32)


def load_image(path: str) -> np.ndarray:
    suffix = path.lower().rsplit(".", 1)[-1] if "." in path else ""

    if suffix in {"npy", "npz"}:
        data = np.load(path)
        if isinstance(data, np.lib.npyio.NpzFile):
            if len(data.files) != 1:
                raise ValueError(f"{path}: expected one array in .npz file")
            arr = data[data.files[0]]
        else:
            arr = data
        return np.asarray(_to_luma_array(arr), dtype=np.float32)

    try:
        with Image.open(path) as img:
            arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    except UnidentifiedImageError, OSError:
        arr = _load_netpbm_image(path)
    return np.asarray(_to_luma_array(arr), dtype=np.float32)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare two images using IW-SSIM")
    parser.add_argument("reference", help="reference image path")
    parser.add_argument("distorted", help="distorted image path")
    parser.add_argument("--cuda", action="store_true", help="use CUDA if available")
    parser.add_argument(
        "--double", action="store_true", help="use float64 for torch tensors"
    )
    parser.add_argument(
        "--no-iw", action="store_true", help="disable information weighting"
    )
    args = parser.parse_args(argv)

    if args.cuda and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")

    ref = load_image(args.reference)
    dis = load_image(args.distorted)
    if ref.shape != dis.shape:
        raise SystemExit(
            f"image dimensions do not match: reference={ref.shape}, distorted={dis.shape}"
        )

    metric = IW_SSIM(iw_flag=not args.no_iw, use_cuda=args.cuda, use_double=args.double)
    score = metric.test(ref, dis)
    print(f"IW-SSIM: {float(score):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
