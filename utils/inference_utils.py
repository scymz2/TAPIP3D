
from typing import Tuple
import torch
from pathlib import Path
from third_party.cotracker.model_utils import get_points_on_a_grid
import models
import av
import cv2
import numpy as np
from einops import repeat, rearrange

def get_grid_queries(grid_size: int, depths: torch.Tensor, intrinsics: torch.Tensor, extrinsics: torch.Tensor):
    if len (depths.shape) == 3:
        return get_grid_queries(
            grid_size=grid_size,
            depths=depths.unsqueeze(0),
            intrinsics=intrinsics.unsqueeze(0),
            extrinsics=extrinsics.unsqueeze(0)
        ).squeeze(0)

    image_size = depths.shape[-2:]
    xy = get_points_on_a_grid(grid_size, image_size).to(intrinsics.device) # type: ignore
    ji = torch.round(xy).to(torch.int32)
    d = depths[:, 0][torch.arange(depths.shape[0])[:, None], ji[..., 1], ji[..., 0]]

    assert d.shape[0] == 1, "batch size must be 1"
    mask = d[0] > 0
    d = d[:, mask]
    xy = xy[:, mask]
    ji = ji[:, mask]

    inv_intrinsics0 = torch.linalg.inv(intrinsics[0, 0])
    inv_extrinsics0 = torch.linalg.inv(extrinsics[0, 0])

    xy_homo = torch.cat([xy, torch.ones_like(xy[..., :1])], dim=-1)
    xy_homo = torch.einsum('ij,bnj->bni', inv_intrinsics0, xy_homo)
    local_coords = xy_homo * d[..., None]
    local_coords_homo = torch.cat([local_coords, torch.ones_like(local_coords[..., :1])], dim=-1)
    world_coords = torch.einsum('ij,bnj->bni', inv_extrinsics0, local_coords_homo)
    world_coords = world_coords[..., :3]

    queries = torch.cat([torch.zeros_like(xy[:, :, :1]), world_coords], dim=-1).to(depths.device)  # type: ignore
    return queries

@torch.inference_mode()
def _inference_with_grid(
    *,
    model: torch.nn.Module,
    video: torch.Tensor,
    depths: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    query_point: torch.Tensor,
    num_iters: int = 6,
    grid_size: int = 8,
    **kwargs,
):
    if grid_size != 0:
        additional_queries = get_grid_queries(grid_size, depths=depths, intrinsics=intrinsics, extrinsics=extrinsics)
        query_point = torch.cat([query_point, additional_queries], dim=1)
        N_supports = additional_queries.shape[1]
    else:
        N_supports = 0

    preds, train_data_list = model(
        rgb_obs=video,
        depth_obs=depths,
        num_iters=num_iters,
        query_point=query_point,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        mode="inference",
        **kwargs
    )
    N_total = preds.coords.shape[2]
    preds = preds.query_slice(slice(0, N_total - N_supports))
    return preds, train_data_list

def load_model(checkpoint_path: str):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    model, cfg = models.from_pretrained(checkpoint_path)
    if hasattr(model, "eval_mode"):
        model.set_eval_mode("raw")
    model.eval()

    return model

def read_video(video_path: str) -> np.ndarray:
    container = av.open(video_path)
    frames = []
    for frame in container.decode(video=0):
        frames.append(frame.to_ndarray(format="rgb24"))
    container.close()
    return np.stack(frames)

def resize_depth_bilinear(depth: np.ndarray, new_shape: Tuple[int, int]) -> np.ndarray:
    is_valid = (depth > 0).astype(np.float32)
    depth_resized = cv2.resize(depth, new_shape, interpolation=cv2.INTER_LINEAR)
    is_valid_resized = cv2.resize(is_valid, new_shape, interpolation=cv2.INTER_LINEAR)
    depth_resized = depth_resized / (is_valid_resized + 1e-6)
    depth_resized[is_valid_resized <= 1e-6] = 0.0
    return depth_resized

@torch.no_grad()
def inference(
    *,
    model: torch.nn.Module,
    video: torch.Tensor,
    depths: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    query_point: torch.Tensor,
    num_iters: int = 6,
    grid_size: int = 8,
    bidrectional: bool = True, # 双向追踪（正序+反序）
    vis_threshold = 0.9,       # 可见性阈值
) -> Tuple[torch.Tensor, torch.Tensor]:
    _depths = depths.clone() # clone会保留计算图，也就是于原张量的梯度关联，但是内存独立
    _depths = _depths[_depths > 0].reshape(-1) # 过滤无效值并展平
    q25 = torch.kthvalue(_depths, int(0.25 * len(_depths))).values # 找到第25%的值， 四分位数
    q75 = torch.kthvalue(_depths, int(0.75 * len(_depths))).values # 找到第75%的值， 四分位数
    iqr = q75 - q25 # 计算四分位间距（IQR）
    _depth_roi = torch.tensor( # 定义深度值的合理范围
        [1e-7, (q75 + 1.5 * iqr).item()], 
        dtype=torch.float32, 
        device=video.device
    )

    T, C, H, W = video.shape
    assert depths.shape == (T, H, W)
    N = query_point.shape[0]

    model.set_image_size((H, W))

    preds, _ = _inference_with_grid(
        model=model,
        video=video[None], # None的作用时在最前面增加一个新的维度，也就是等价于unsqueeze(0),通常用于增加batch维度
        depths=depths[None],
        intrinsics=intrinsics[None],
        extrinsics=extrinsics[None],
        query_point=query_point[None],
        num_iters=num_iters,
        depth_roi=_depth_roi,
        grid_size=grid_size
    )

    # 反向推理
    # 此外，当query_point[..., 0] 不存在>0的情况时，表示所有点都是从第一帧开始的（都是默认生成的辅助网格点），正向推理就足够了
    if bidrectional and not model.bidirectional and (query_point[..., 0] > 0).any():
        preds_backward, _ = _inference_with_grid(
            model=model,
            video=video[None].flip(dims=(1,)), # 表示在时间维度（帧序列）上翻转
            depths=depths[None].flip(dims=(1,)),
            intrinsics=intrinsics[None].flip(dims=(1,)),
            extrinsics=extrinsics[None].flip(dims=(1,)),
            query_point=torch.cat([T - 1 - query_point[..., :1], query_point[..., 1:]], dim=-1)[None],
            num_iters=num_iters,
            depth_roi=_depth_roi,
            grid_size=grid_size,
        )
        # query_point的shape是[b, N, 4], query_point[..., 0]是时间索引，query_point[..., 1:]是空间坐标
        # 有些点不是从第一帧就出现的，而是中途才出现，对于这些点，在它出现之前的帧，正向推理结果可能不准确，反向推理能更好地补全
        # 这段代码的作用是在正向和反向推理结果之间，按时间条件选择最终的点坐标，实现前半段用反向结果，后半段用正向结果
        # 第一个repeat将时间序列扩展到[b, T, N, 3], 第二个repeat将查询点的时间索引扩展到[b, T, N, 3], 然后进行比较时间
        # where的条件是， 如果当前时间小于查询点的时间索引，就用反向推理的结果，否则用正向推理的结果
        # 第一个repeat的结果按时间增长， 第二个repeat的结果保持查询点的时间索引不变，然后当前者小于后者时，结果为true, 反之为false
        # true时用反向推理的结果， false时用正向推理的结果
        preds.coords = torch.where(
            repeat(torch.arange(T, device=video.device), 't -> b t n 3', b=1, n=N) < repeat(query_point[..., 0][None], 'b n -> b t n 3', t=T, n=N),
            preds_backward.coords.flip(dims=(1,)),
            preds.coords
        )
        preds.visibs = torch.where(
            repeat(torch.arange(T, device=video.device), 't -> b t n', b=1, n=N) < repeat(query_point[..., 0][None], 'b n -> b t n', t=T, n=N),
            preds_backward.visibs.flip(dims=(1,)),
            preds.visibs
        )

    # 可见性阈值化与输出
    coords, visib_logits = preds.coords, preds.visibs
    visibs = torch.sigmoid(visib_logits) >= vis_threshold
    return coords.squeeze(), visibs.squeeze()