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
		//// int ori_visited = atomicCAS(visited + column, 0, 1);
		//// if (ori_visited) continue;
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

#if 0
__global__ void achieveLevelsChild(int        row,
		                           const int* row_offsets,
								   const int* column_indices,
								   bool *     n_frontier,
								   int  *     visited,
								   int*       updated_by,
								   int  *     levels,
								   bool *     has_frontier)
{
	int start_idx = row_offsets[row];
	int cur_cost  = levels[row];

	int column = column_indices[start_idx + blockIdx.x];
	if (!visited[column]) {
		visited[column] = true;
		n_frontier[column] = true;
		updated_by[column] = row;
		levels[column]  = cur_cost + 1;
		if (!(*has_frontier))
			*has_frontier = true;
	}
}

__global__ void alterAchieveLevels(int        N,
						 		   const int* row_offsets,
								   const int* column_indices,
								   bool*      frontier,
								   bool*      n_frontier,
								   int*       visited,
								   int*       updated_by,
								   int*       levels,
								   bool*      has_frontier)
{
	for (int l = 0; l < N; l++) {
		if (*has_frontier) {
			if (threadIdx.x == 0)
				*has_frontier = false;

		} else
			break;

		__syncthreads();

		for (int i = threadIdx.x; i < N; i += blockDim.x)
			if (frontier[i]) {
				int threadsNum = row_offsets[i+1] - row_offsets[i];
				if (threadsNum > 0)
					achieveLevelsChild<<<threadsNum, 1>>>(i, row_offsets, column_indices, n_frontier, visited, updated_by, levels, has_frontier);
			}

		__syncthreads();
		if (threadIdx.x == 0)
			cudaDeviceSynchronize();
		__syncthreads();

		for (int i = threadIdx.x; i < N; i += blockDim.x) {
			frontier[i] = n_frontier[i];
			n_frontier[i] = false;
		}

		__syncthreads();
	}
}
#endif


} // namespace device
} // namespace rcm

#endif
