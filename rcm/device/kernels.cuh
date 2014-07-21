#ifndef KERNELS_CUH
#define KERNELS_CUH

namespace rcm {

namespace device {

__global__ void generalToSymmetric(int          nnz,
								   const int *  row_indices,
								   const int *  column_indices,
								   int *        new_row_indices,
								   int *        new_column_indices)
{
	int tid = threadIdx.x + (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x;

	if (tid >= nnz) return;

	new_row_indices[tid << 1] = row_indices[tid];
	new_column_indices[tid << 1] = column_indices[tid];
	new_row_indices[(tid << 1) + 1] = column_indices[tid];
	new_column_indices[(tid << 1) + 1] = row_indices[tid];
}

__global__ void achieveLevels(int        N,
							  const int* row_offsets,
							  const int* column_indices,
							  bool*      frontier,
							  bool*      n_frontier,
							  int*       visited,
							  int*       updated_by,
							  int*       levels,
							  bool*      has_frontier)
{
	int bid = blockIdx.x + blockIdx.y * gridDim.x;

	if (bid >= N) return;

	if (!frontier[bid]) return;

	int start_idx = row_offsets[bid], end_idx = row_offsets[bid + 1];
	int cur_cost  = levels[bid];

	for (int tid = start_idx + threadIdx.x; tid < end_idx; tid += blockDim.x) {
		int column = column_indices[tid];
		if (!visited[column]) {
			visited[column] = true;
			n_frontier[column] = true;
			updated_by[column] = bid;
			levels[column]  = cur_cost + 1;
			if (!(*has_frontier))
				*has_frontier = true;
		}
	}
}

} // namespace device
} // namespace rcm

#endif
