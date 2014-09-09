#ifndef RCM_UM_H
#define RCM_UM_H

#include <rcm/common.h>
#include <rcm/timer.h>
#include <rcm/exception.h>
#include <rcm/rcm.h>
#include <rcm/device/kernels.cuh>

#include <thrust/scan.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/gather.h>
#include <thrust/binary_search.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/logical.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/adjacent_difference.h>
#include <thrust/inner_product.h>

#include <omp.h>

extern "C" {
#include "mm_io/mm_io.h"
}

namespace rcm {

class RCM_UM: public RCM_base
{
private:
  typedef ManagedVector<int>                      IntVector;
  typedef ManagedVector<double>                   DoubleVector;
  typedef ManagedVector<bool>                     BoolVector;

  typedef thrust::host_vector<int>                IntVectorH;
  typedef thrust::host_vector<bool>               BoolVectorH;

  IntVector      m_row_offsets;
  IntVector      m_column_indices;
  DoubleVector   m_values;

  IntVector      m_perm;

  void local_sort(int *key_begin, int *key_end, int minimum, int maximum, int *value_begin)
  {
	  IntVectorH bucket(maximum - minimum + 1, 0);
	  IntVectorH tmp_key(key_end - key_begin);
	  IntVectorH tmp_value(key_end - key_begin);

	  for (int * key_iterator = key_begin; key_iterator < key_end; key_iterator ++)
		  bucket[(*key_iterator) - minimum] ++;

	  thrust::inclusive_scan(bucket.begin(), bucket.end(), bucket.begin());

	  for (int * key_iterator = key_end - 1; key_iterator >= key_begin; key_iterator --) {
		  int tmp_idx = (--bucket[(*key_iterator) - minimum]);
		  tmp_key[tmp_idx]   = *key_iterator;
		  tmp_value[tmp_idx] = *(value_begin + (key_iterator - key_begin));
	  }

	  thrust::copy(tmp_key.begin(), tmp_key.end(), key_begin);
	  thrust::copy(tmp_value.begin(), tmp_value.end(), value_begin);
  }

public:
  RCM_UM(const IntVector&    row_offsets,
         const IntVector&    column_indices,
         const DoubleVector& values,
		 int                 iteration_count)
  : m_row_offsets(row_offsets),
    m_column_indices(column_indices),
    m_values(values)
  {
    size_t n = row_offsets.size() - 1;
    m_perm.resize(n);
    m_n               = n;
    m_nnz             = m_values.size();
	m_iteration_count = 0;
	m_max_iteration_count = iteration_count;
  }

  ~RCM_UM() {}

  void execute();
};

void
RCM_UM::execute()
{
	IntVector        visited(m_n);
	IntVectorH       post_visited(m_n);
	IntVectorH       special(m_n);
	BoolVector       tried(m_n, false);

	IntVector        levels(m_n);

	const int ITER_COUNT = m_max_iteration_count;

	IntVector        row_indices(m_nnz);

	IntVector        tmp_row_offsets(m_n + 1);
	IntVector        tmp_column_indices(m_nnz << 1);
	IntVector        ori_degrees(m_n);

	IntVector        ori_levels(m_n);

	CPUTimer         local_timer;

	local_timer.Start();

	offsets_to_indices(m_row_offsets, row_indices);

	{
		IntVector        extended_degrees(m_nnz << 1);
		IntVector        tmp_row_indices(m_nnz << 1);
		thrust::transform(m_row_offsets.begin() + 1, m_row_offsets.end(), m_row_offsets.begin(), ori_degrees.begin(), thrust::minus<int>());

		thrust::copy(thrust::make_zip_iterator(thrust::make_tuple(row_indices.begin(), m_column_indices.begin(), m_column_indices.begin(), row_indices.begin())), 
				     thrust::make_zip_iterator(thrust::make_tuple(row_indices.end(), m_column_indices.end(), m_column_indices.end(), row_indices.end())),
				     thrust::make_zip_iterator(thrust::make_tuple(tmp_row_indices.begin(), tmp_row_indices.begin() + m_nnz, tmp_column_indices.begin(), tmp_column_indices.begin() + m_nnz))
				);

		thrust::gather(tmp_column_indices.begin(), tmp_column_indices.end(), ori_degrees.begin(), extended_degrees.begin());

		thrust::sort(thrust::make_zip_iterator(thrust::make_tuple(tmp_row_indices.begin(), extended_degrees.begin(), tmp_column_indices.begin())),
                     thrust::make_zip_iterator(thrust::make_tuple(tmp_row_indices.end(), extended_degrees.end(), tmp_column_indices.end())), TripleCompare()
				);
		indices_to_offsets(tmp_row_indices, tmp_row_offsets);

		m_half_bandwidth = m_half_bandwidth_original = thrust::inner_product(row_indices.begin(), row_indices.end(), m_column_indices.begin(), 0, thrust::maximum<int>(), Difference());

		thrust::sequence(m_perm.begin(), m_perm.end());
		cudaDeviceSynchronize();
	}

	IntVector tmp_reordering(m_n);
	IntVector tmp_perm(m_n);

	IntVector level_offsets;

	IntVectorH h_row_offsets(tmp_row_offsets.begin(), tmp_row_offsets.end());
	IntVectorH h_column_indices(tmp_column_indices.begin(), tmp_column_indices.end());

	thrust::fill(visited.begin(),  visited.end(), -1);
	thrust::fill(post_visited.begin(),  post_visited.end(), -1);
	thrust::fill(special.begin(),  special.end(), -1);

	cudaDeviceSynchronize();

	int S = 0;
	int max_level = 0;
	int p_max_level;

	const double stop_ratio = 0.01;

	m_time_bfs = m_time_node_order = 0.0;

	IntVectorH h_tmp_reordering(m_n);

	if (m_nnz >= 5000000)
		omp_set_num_threads(20);
	else
		omp_set_num_threads(10);

	local_timer.Stop();
	m_time_pre = local_timer.getElapsed();

	for (int i = 0; i < ITER_COUNT; i++) {
		m_iteration_count = i + 1;
		local_timer.Start();
		if (i > 0) {
			int start_idx = level_offsets[max_level];
			int max_count = level_offsets[max_level + 1] - start_idx;

			if( max_count > 1 ) {
				IntVector max_level_valence(max_count);

				thrust::copy_if(ori_degrees.begin(),
						ori_degrees.end(),
						ori_levels.begin(),
						max_level_valence.begin(),
						EqualTo(max_level)); 

				int min_valence_pos = thrust::min_element(max_level_valence.begin(), max_level_valence.end()) - max_level_valence.begin();
				cudaDeviceSynchronize();

				S = tmp_reordering[start_idx + min_valence_pos];
			}
			else
				S = tmp_reordering[m_n - 1];

			while (tried[S]) S = (S + 1) % m_n;
		} else {
			S = thrust::min_element(ori_degrees.begin(), ori_degrees.end()) - ori_degrees.begin();
			cudaDeviceSynchronize();
		}

		tried[S] = true;

		visited[S]  = i;
		tmp_reordering[0] = S;

		int last = 0;
		const int NODE_COUNT_THRESHOLD = 10;

		const int *p_row_offsets    = thrust::raw_pointer_cast(&tmp_row_offsets[0]);
		const int *p_column_indices = thrust::raw_pointer_cast(&tmp_column_indices[0]);
		int  *     p_visited        = thrust::raw_pointer_cast(&visited[0]);
		int  *     p_levels         = thrust::raw_pointer_cast(&levels[0]);

		int queue_begin = 0;
		IntVector queue_end(1, 1);
		int *p_queue_end = thrust::raw_pointer_cast(&queue_end[0]);
		int *p_reordering= thrust::raw_pointer_cast(&tmp_reordering[0]);

		special[0] = i;

		for (int l = 0; l < m_n; l ++) {
			int local_queue_end = queue_end[0];
			if (local_queue_end - queue_begin >= NODE_COUNT_THRESHOLD)
			{
				int blockX = local_queue_end - queue_begin, blockY = 1;
				kernelConfigAdjust(blockX, blockY, MAX_GRID_DIMENSION);
				dim3 tmp_grids(blockX, blockY);
				device::alterAchieveLevels<<<tmp_grids, 64>>>(i, l, p_row_offsets, p_column_indices, p_reordering, queue_begin, local_queue_end, p_queue_end, p_visited, p_levels);
				cudaDeviceSynchronize();
				queue_begin = local_queue_end;
			} else {
				if (local_queue_end == queue_begin) {
					if (queue_begin >= m_n) break;
					special[l] = i;
					for (int j = last; j < m_n; j++)
						if (visited[j] != i) {
							visited[j] = i;
							tmp_reordering[local_queue_end] = j;
							last = j;
							tried[j] = true;
							queue_end[0]++;
							l --;
							break;
						}
				} else {
					for (int l2 = queue_begin; l2 < local_queue_end; l2++) {
						levels[l2] = l;
						int row = tmp_reordering[l2];
						int start_idx = tmp_row_offsets[row], end_idx = tmp_row_offsets[row + 1];
						for (int j = start_idx; j < end_idx; j++) {
							int column = tmp_column_indices[j];
							if (visited[column] != i) {
								visited[column] = i;
								tmp_reordering[queue_end[0]] = column;
								queue_end[0] ++;
							}
						}
					}
					queue_begin = local_queue_end;
				}
			}
		}
		local_timer.Stop();
		m_time_bfs += local_timer.getElapsed();

		local_timer.Start();
		thrust::scatter(levels.begin(), levels.end(), tmp_reordering.begin(), ori_levels.begin());
		cudaDeviceSynchronize();

		max_level = levels[levels.size() - 1];
		level_offsets.resize(max_level + 2);
		indices_to_offsets(levels, level_offsets);
		cudaDeviceSynchronize();

		IntVectorH h_ori_levels(ori_levels.begin(), ori_levels.end());
		IntVectorH h_level_offsets(level_offsets.begin(), level_offsets.end());


		cudaDeviceSynchronize();
		IntVectorH h_write_offsets = h_level_offsets;

		int threadId, numThreads;
		int * volatile p_write_offsets = thrust::raw_pointer_cast(&h_write_offsets[0]);
		for (int l = max_level; l >= 0; l--) {
			if (special[l] == i) {
				p_write_offsets[l]++;
				h_tmp_reordering[h_level_offsets[l]] = tmp_reordering[h_level_offsets[l]];
			}
		}

#pragma omp parallel if (m_nnz >= 500000) private (threadId) shared(p_write_offsets, numThreads, h_level_offsets, h_tmp_reordering, h_row_offsets, h_column_indices, post_visited)
		{

			threadId   = omp_get_thread_num();
			numThreads = omp_get_num_threads();

			for (int l = threadId; l < max_level; l += numThreads) { // Parallelizable
				int local_read_offset = h_level_offsets[l], local_level_offset = h_level_offsets[l+1];
				while (local_read_offset < local_level_offset) {
					while(local_read_offset == p_write_offsets[l]) {
					} // Spin

					int local_write_offset = p_write_offsets[l];

					for (; local_read_offset < local_write_offset; local_read_offset++) {
						int row = h_tmp_reordering[local_read_offset];

						int start_idx = h_row_offsets[row], end_idx = h_row_offsets[row+1];

						for (int l2 = start_idx; l2 < end_idx; l2 ++) {
							int column = h_column_indices[l2];
							if (h_ori_levels[column] == l + 1) {
								if (post_visited[column] != i) {
									post_visited[column] = i;
									int local_n_write_offset = p_write_offsets[l+1];
									h_tmp_reordering[local_n_write_offset] = column;
									p_write_offsets[l+1] = local_n_write_offset + 1;
								}
							}
						}
					}
				}
			}
		}

		int *p_tmp_perm = thrust::raw_pointer_cast(&tmp_perm[0]);

		thrust::copy(h_tmp_reordering.begin(), h_tmp_reordering.end(), tmp_reordering.begin());

		thrust::scatter(thrust::make_counting_iterator(0),
				        thrust::make_counting_iterator(int(m_n)),
						tmp_reordering.begin(), 
						tmp_perm.begin());


		int tmp_half_bandwidth = thrust::inner_product(row_indices.begin(), row_indices.end(), m_column_indices.begin(), 0, thrust::maximum<int>(), ExtendedDifference(p_tmp_perm));

		cudaDeviceSynchronize();

		if (tmp_half_bandwidth < m_half_bandwidth) {
			m_half_bandwidth = tmp_half_bandwidth;
			m_perm           = tmp_perm;
		}

		if (i > 0) {
			if (m_half_bandwidth <= tmp_half_bandwidth && p_max_level >= max_level) {
				local_timer.Stop();
				m_time_node_order += local_timer.getElapsed();
				break;
			}
			double hb_ratio = 1.0 * (m_half_bandwidth - tmp_half_bandwidth) / m_half_bandwidth;
			if (hb_ratio < stop_ratio) {
				double max_level_ratio = 1.0 * (max_level - p_max_level) / p_max_level;
				if (max_level_ratio < stop_ratio) {
					local_timer.Stop();
					m_time_node_order += local_timer.getElapsed();
					break;
				}
			}

			if (p_max_level < max_level)
				p_max_level = max_level;
		} else
			p_max_level = max_level;
		local_timer.Stop();
		m_time_node_order += local_timer.getElapsed();
	}

}

} // end namespace rcm

#endif
