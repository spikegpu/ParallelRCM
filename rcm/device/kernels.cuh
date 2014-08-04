#ifndef KERNELS_CUH
#define KERNELS_CUH

namespace rcm {

namespace device {

__global__ void achieveLevels(int        cur_level,
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
			int old_queue_end = atomicAdd(p_queue_end, 1);
			reordering[old_queue_end] = column;
			updated_by[old_queue_end] = row;
			n_degrees[old_queue_end]  = degrees[column];
		}
	}
}

__global__ void alterAchieveLevels(int        cur_iter,
		                           int        cur_level,
							       const int* row_offsets,
								   const int* column_indices,
								   int*       reordering,
								   int        queue_begin,
								   int        queue_end,
								   int*       p_queue_end,
								   int*       visited,
								   int*       levels)
{
	int bid = blockIdx.x + blockIdx.y * gridDim.x;

	if (bid + queue_begin >= queue_end) return;

	levels[bid + queue_begin] = cur_level;
	int row = reordering[bid + queue_begin];

	int start_idx = row_offsets[row], end_idx = row_offsets[row + 1];

	for (int tid = start_idx + threadIdx.x; tid < end_idx; tid += blockDim.x) {
		int column = column_indices[tid];

		int local_visited = atomicCAS(visited + column, cur_iter - 1, cur_iter);

		if (local_visited == cur_iter - 1) {
			int old_queue_end = atomicAdd(p_queue_end, 1);
			reordering[old_queue_end] = column;
		}
	}
}

} // namespace device
} // namespace rcm

#endif
