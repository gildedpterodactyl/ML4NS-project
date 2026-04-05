import torch

def batched_gather(data, inds, dim=0, no_batch_dims=0):
    ranges = []
    for i, s in enumerate(data.shape[:no_batch_dims]):
        r = torch.arange(s)
        r = r.view(*(*((1,) * i), -1, *((1,) * (len(inds.shape) - i - 1))))
        ranges.append(r)

    remaining_dims = [slice(None) for _ in range(len(data.shape) - no_batch_dims)]
    remaining_dims[dim - no_batch_dims if dim >= 0 else dim] = inds
    ranges.extend(remaining_dims)
    return data[ranges]


def node_gather(s: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    return batched_gather(s.unsqueeze(-3), edges, -2, no_batch_dims=len(s.shape) - 1)


MAX_SUPPORTED_DISTANCE = 1e6

def knn_graph(
    coords: torch.Tensor,
    coord_mask: torch.Tensor,
    padding_mask: torch.Tensor,
    sequence_id: torch.Tensor,
    *,
    no_knn: int,
):
    L = coords.shape[-2]
    num_by_dist = min(no_knn, L) if no_knn is not None else L
    device = coords.device

    coords = coords.nan_to_num()
    coord_mask = ~(coord_mask[..., None, :] & coord_mask[..., :, None])
    padding_pairwise_mask = padding_mask[..., None, :] | padding_mask[..., :, None]
    if sequence_id is not None:
        padding_pairwise_mask |= torch.unsqueeze(sequence_id, 1) != torch.unsqueeze(
            sequence_id, 2
        )
    dists = (coords.unsqueeze(-2) - coords.unsqueeze(-3)).norm(dim=-1)
    arange = torch.arange(L, device=device)
    seq_dists = (arange.unsqueeze(-1) - arange.unsqueeze(-2)).abs()
    # We only support up to a certain distance, above that, we use sequence distance
    # instead. This is so that when a large portion of the structure is masked out,
    # the edges are built according to sequence distance.
    max_dist = MAX_SUPPORTED_DISTANCE
    torch._assert_async((dists[~coord_mask] < max_dist).all())
    struct_then_seq_dist = (
        seq_dists.to(dists.dtype)
        .mul(1e2)
        .add(max_dist)
        .where(coord_mask, dists)
        .masked_fill(padding_pairwise_mask, torch.inf)
    )
    dists, edges = struct_then_seq_dist.sort(dim=-1, descending=False)
    # This is a L x L tensor, where we index by rows first,
    # and columns are the edges we should pick.
    chosen_edges = edges[..., :num_by_dist]
    chosen_mask = dists[..., :num_by_dist].isfinite()
    return chosen_edges, chosen_mask


def find_knn_edges(
    coords,     # CA coords
    padding_mask,
    coord_mask,
    sequence_id: torch.Tensor | None = None,
    knn: int | None = None,
) -> tuple:
    # assert knn is not None, "Must specify a non-null knn to find_knn_edges"
    if sequence_id is None:
        sequence_id = torch.zeros(
            (coords.shape[0], coords.shape[1]), device=coords.device
        ).long()
    print(coords.shape, padding_mask.shape, coord_mask.shape, sequence_id.shape)

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):  # type: ignore
        edges, edge_mask = knn_graph(
            coords, coord_mask, padding_mask, sequence_id, no_knn=knn
        )

    return edges, edge_mask