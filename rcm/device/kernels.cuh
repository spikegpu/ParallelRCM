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

__global__ void alterAchieveLevels(int        cur_level,
								   const int* row_offsets,
								   const int* column_indices,
								   int*       reordering,
								   int        queue_begin,
								   int        queue_end,
								   int*       p_queue_end,
								   int*       visited,
								   int*       levels,
								   const int* degrees,
								   int*       n_degrees,
								   int*       updated_by)
{
	int bid = blockIdx.x + blockIdx.y * gridDim.x;

	if (bid + queue_begin >= queue_end) return;

	levels[bid + queue_begin] = cur_level;
	int row = reordering[bid + queue_begin];

	int start_idx = row_offsets[row], end_idx = row_offsets[row + 1];

	for (int tid = start_idx + threadIdx.x; tid < end_idx; tid += blockDim.x) {
		int column = column_indices[tid];

		int local_visited = atomicCAS(visited + column, 0, 1);

		if (!local_visited) {
			visited[column] = true;
			int old_queue_end = atomicAdd(p_queue_end, 1);
			reordering[old_queue_end] = column;
			updated_by[old_queue_end] = row;
			n_degrees[old_queue_end]  = degrees[column];
		}
	}
}

} // namespace device
} // namespace rcm

#endif
