import torch
from torch import Tensor
from torch_geometric.loader.neighbor_sampler import NeighborSampler, Adj, EdgeIndex
from torch_geometric.utils import to_undirected

class NeighborSamplerbyNFT(NeighborSampler):
    def __init__(self, edge_index, sizes, edge_attr=None, transform=None, prob_vector=None, **kwargs):
        super(NeighborSamplerbyNFT, self).__init__(edge_index, sizes=sizes, transform=transform, **kwargs)
        self.edge_attr = edge_attr.to('cpu') if edge_attr is not None else None
        self.first_layer_edge_attr = None
        self.prob_vector = prob_vector


    def sample(self, batch):
        if not isinstance(batch, Tensor):
            batch = torch.tensor(batch, dtype=torch.long)

        batch_size: int = len(batch)
        adjs = []
        n_id = batch
        if self.prob_vector is not None:
            n_id = torch.multinomial(self.prob_vector, num_samples=len(n_id), replacement=True)

        for i, size in enumerate(self.sizes):
            adj_t, n_id = self.adj_t.sample_adj(n_id, size, replace=False)
            e_id = adj_t.storage.value()
            size = adj_t.sparse_sizes()[::-1]

            if self.edge_attr is not None:
                edge_attr = self.edge_attr[e_id].to('cpu')
                if i == 0:
                    self.first_layer_edge_attr = edge_attr
                else:
                    mask = self.compute_mask(edge_attr, self.first_layer_edge_attr)
                    adj_t = adj_t.masked_select_nnz(mask, layout='coo')
                    e_id = adj_t.storage.value()
                    size = adj_t.sparse_sizes()[::-1]

            if self.is_sparse_tensor:
                adjs.append(Adj(adj_t, e_id, size))
            else:
                row, col, _ = adj_t.coo()
                edge_index = torch.stack([col, row], dim=0)
                adjs.append(EdgeIndex(edge_index, e_id, size))

        adjs = adjs[0] if len(adjs) == 1 else adjs[::-1]
        out = (batch_size, n_id, adjs)
        out = self.transform(*out) if self.transform is not None else out
        return out

    def compute_mask(self, edge_attr, first_layer_edge_attr):
        mask = (edge_attr[:, None] == first_layer_edge_attr).all(dim=-1).any(dim=1)
        return mask