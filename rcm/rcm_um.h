#ifndef RCM_UM_H
#define RCM_UM_H

#include <rcm/common.h>
#include <rcm/timer.h>
#include <rcm/exception.h>
#include <rcm/rcm.h>
#include <rcm/device/kernels.cuh>

#include <cusp/csr_matrix.h>
#include <cusp/coo_matrix.h>
#include <cusp/blas.h>
#include <cusp/print.h>
#include <cusp/graph/symmetric_rcm.h>

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

extern "C" {
#include "mm_io/mm_io.h"
}

namespace rcm {

class RCM_UM: public RCM_base
{
private:
  typedef ManagedVector<int>                      IntVector;
  typedef cusp::array1d<int, cusp::device_memory> CuspIntVectorD;
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
         const DoubleVector& values)
  : m_row_offsets(row_offsets),
    m_column_indices(column_indices),
    m_values(values)
  {
    size_t n = row_offsets.size() - 1;
    m_perm.resize(n);
    m_n   = n;
    m_nnz = m_values.size();
  }

  ~RCM_UM() {}

  void execute();
  void execute_cusp();
};

void
RCM_UM::execute()
{
	BoolVector       frontier(m_n);
	BoolVector       n_frontier(m_n);
	IntVector        visited(m_n);
	BoolVectorH      post_visited(m_n);
	BoolVector       tried(m_n, false);

	IntVector        updated_by(m_n);
	IntVector        levels(m_n);

	const int ITER_COUNT = 5;

	IntVector        row_indices(m_nnz);

	IntVector        tmp_row_offsets(m_n + 1);
	IntVector        tmp_row_indices(m_nnz << 1);
	IntVector        tmp_column_indices(m_nnz << 1);
	IntVector        degrees(m_n);
	IntVector        ori_degrees(m_n);
	IntVector        tmp_degrees(m_n);

	offsets_to_indices(m_row_offsets, row_indices);

	{
		int numThreads = m_nnz, numBlockX = 1, numBlockY = 1;

		kernelConfigAdjust(numThreads, numBlockX, numBlockY, BLOCK_SIZE, MAX_GRID_DIMENSION);
		dim3 grids(numBlockX, numBlockY);

		const int*       p_row_indices = thrust::raw_pointer_cast(&row_indices[0]);
		const int*       p_column_indices = thrust::raw_pointer_cast(&m_column_indices[0]);
		int *            p_new_row_indices = thrust::raw_pointer_cast(&tmp_row_indices[0]);
		int *            p_new_column_indices = thrust::raw_pointer_cast(&tmp_column_indices[0]);
		thrust::transform(m_row_offsets.begin() + 1, m_row_offsets.end(), m_row_offsets.begin(), ori_degrees.begin(), thrust::minus<int>());

		device::generalToSymmetric<<<grids, numThreads>>>(m_nnz, p_row_indices, p_column_indices, p_new_row_indices, p_new_column_indices);

		thrust::sort_by_key(tmp_row_indices.begin(), tmp_row_indices.end(), tmp_column_indices.begin());
		indices_to_offsets(tmp_row_indices, tmp_row_offsets);
		thrust::transform(tmp_row_offsets.begin() + 1, tmp_row_offsets.end(), tmp_row_offsets.begin(), degrees.begin(), thrust::minus<int>());

		m_half_bandwidth = m_half_bandwidth_original = thrust::inner_product(row_indices.begin(), row_indices.end(), m_column_indices.begin(), 0, thrust::maximum<int>(), Difference());

		thrust::sequence(m_perm.begin(), m_perm.end());
		cudaDeviceSynchronize();
	}

	IntVector tmp_reordering(m_n);
	IntVector tmp_perm(m_n);

	int numBlockX = m_n, numBlockY = 1;
	kernelConfigAdjust(numBlockX, numBlockY, MAX_GRID_DIMENSION);
	dim3 grids(numBlockX, numBlockY);

	int S = 0;
	int max_level = 0;

	for (int i = 0; i < ITER_COUNT; i++) {
		
		if (i > 0) {
			int max_count = thrust::count(levels.begin(), levels.end(), max_level);

			if( max_count > 1 ) {
				IntVector max_level_vertices(max_count);
				IntVector max_level_valence(max_count);

				thrust::copy_if(thrust::counting_iterator<int>(0),
						thrust::counting_iterator<int>(int(m_n)),
						levels.begin(),
						max_level_vertices.begin(),
						EqualTo(max_level));

				thrust::gather(thrust::counting_iterator<int>(0),
						thrust::counting_iterator<int>(max_count),
						ori_degrees.begin(),
						max_level_valence.begin());
				int min_valence_pos = thrust::min_element(max_level_valence.begin(), max_level_valence.end()) - max_level_valence.begin();
				cudaDeviceSynchronize();

				S = tmp_perm[max_level_vertices[min_valence_pos]];
			}
			else
				S = tmp_perm[m_n - 1];

			if (tried[S]) break;
		}

		tried[S] = true;


		thrust::fill(frontier.begin(), frontier.end(), false);
		thrust::fill(n_frontier.begin(), n_frontier.end(), false);
		thrust::fill(visited.begin(),  visited.end(),  0);
		thrust::fill(post_visited.begin(),  post_visited.end(),  false);
		thrust::sequence(tmp_reordering.begin(), tmp_reordering.end());

		thrust::fill(levels.begin(), levels.end(), 0);
		cudaDeviceSynchronize();

		frontier[S] = true;
		visited[S]  = 1;
		levels[S] = 0;
		updated_by[S] = -1;

		int last = 0;

		BoolVector has_frontier(1);
		bool *     p_has_frontier = thrust::raw_pointer_cast(&has_frontier[0]);
		const int *p_row_offsets    = thrust::raw_pointer_cast(&tmp_row_offsets[0]);
		const int *p_column_indices = thrust::raw_pointer_cast(&tmp_column_indices[0]);
		bool *     p_frontier       = thrust::raw_pointer_cast(&frontier[0]);
		bool *     p_n_frontier     = thrust::raw_pointer_cast(&n_frontier[0]);
		int  *     p_visited        = thrust::raw_pointer_cast(&visited[0]);
		int  *     p_updated_by     = thrust::raw_pointer_cast(&updated_by[0]);
		int  *     p_levels         = thrust::raw_pointer_cast(&levels[0]);

		for (int l = 0; l < m_n; l ++)
		{
			if (*p_has_frontier)
				*p_has_frontier = false;
			else {
				int sum = thrust::reduce(visited.begin(), visited.end());
				cudaDeviceSynchronize();

				if (sum >= m_n) break;

				for (int j = last; j < m_n; j++)
					if (!visited[j]) {
						visited[j] = 1;
						frontier[j] = true;
						levels[j] = l;
						updated_by[j] = -1;
						last = j;
						*p_has_frontier = true;
						tried[j] = true;
						break;
					}

				continue;
			}

			device::achieveLevels<<<grids, 64>>>(m_n, p_row_offsets, p_column_indices, p_frontier, p_n_frontier, p_visited, p_updated_by, p_levels, p_has_frontier);
			thrust::copy(n_frontier.begin(), n_frontier.end(), frontier.begin());
			thrust::fill(n_frontier.begin(), n_frontier.end(), false);
			cudaDeviceSynchronize();
		}

		thrust::copy(degrees.begin(), degrees.end(), tmp_degrees.begin());

		thrust::sort(thrust::make_zip_iterator(thrust::make_tuple(levels.begin(), updated_by.begin(), tmp_degrees.begin(), tmp_reordering.begin() )),
				     thrust::make_zip_iterator(thrust::make_tuple(levels.end(), updated_by.end(), tmp_degrees.end(), tmp_reordering.end())),
					 TupleCompare()
				);


		thrust::scatter(thrust::make_counting_iterator(0),
				       thrust::make_counting_iterator((int)(m_n)),
					   tmp_reordering.begin(),
					   tmp_perm.begin());

		cudaDeviceSynchronize();

		max_level = levels[levels.size() - 1];
		IntVector level_offsets(max_level + 2);
		indices_to_offsets(levels, level_offsets);
		cudaDeviceSynchronize();

		int *p_tmp_perm       = thrust::raw_pointer_cast(&tmp_perm[0]);

		int tmp_half_bandwidth = thrust::inner_product(row_indices.begin(), row_indices.end(), m_column_indices.begin(), 0, thrust::maximum<int>(), ExtendedDifference(p_tmp_perm));

		if (tmp_half_bandwidth < m_half_bandwidth) {
			m_half_bandwidth = tmp_half_bandwidth;
			m_perm           = tmp_perm;
		}

		IntVectorH h_level_offsets(level_offsets.begin(), level_offsets.end());
		IntVectorH h_updated_by(updated_by.begin(), updated_by.end());
		IntVectorH h_tmp_reordering(tmp_reordering.begin(), tmp_reordering.end());
		IntVectorH h_tmp_perm(tmp_perm.begin(), tmp_perm.end());
		IntVectorH h_row_offsets(tmp_row_offsets.begin(), tmp_row_offsets.end());
		IntVectorH h_column_indices(tmp_column_indices.begin(), tmp_column_indices.end());

		int *p_tmp_reordering = thrust::raw_pointer_cast(&h_tmp_reordering[0]);
		int *p_h_updated_by = thrust::raw_pointer_cast(&h_updated_by[0]);

		for (int l = 0; l < m_n; l++) {
			int start_idx = h_level_offsets[l], l2 = h_level_offsets[l + 1];

			if (start_idx >= m_n) break;
			if (l2 <= start_idx) continue;
			if (h_updated_by[start_idx] == -1) {
				post_visited[start_idx] = true;
				continue;
			}

			for (int l3 = start_idx; l3 < l2; l3 ++)
				h_updated_by[l3] = h_tmp_perm[h_updated_by[l3]];

			for (int l3 = start_idx; l3 < l2; l3 ++) {
				int node = h_tmp_reordering[l3];
				int cur_updated_by = h_updated_by[l3];

				int first = h_row_offsets[node], last = h_row_offsets[node + 1];

				for (int l4 = first; l4 < last; l4 ++) {
					int column = h_tmp_perm[h_column_indices[l4]];
					if (!post_visited[column]) continue;

					if (cur_updated_by > column)
						cur_updated_by = column;
				}

				h_updated_by[l3] = cur_updated_by;
			}

			int maximum = 0, minimum = (int)(m_n);

			for (int l3 = start_idx; l3 < l2; l3 ++) {
				int tmp_num = h_updated_by[l3];
				if (maximum < tmp_num)
					maximum = tmp_num;
				if (minimum > tmp_num)
					minimum = tmp_num;

				post_visited[l3] = true;
			}

			local_sort(p_h_updated_by + start_idx, p_h_updated_by + l2, minimum, maximum, p_tmp_reordering + start_idx);

			for (int l3 = start_idx; l3 < l2; l3++)
				h_tmp_perm[h_tmp_reordering[l3]] = l3;
		}

		thrust::copy(h_tmp_perm.begin(), h_tmp_perm.end(), tmp_perm.begin());

		tmp_half_bandwidth = thrust::inner_product(row_indices.begin(), row_indices.end(), m_column_indices.begin(), 0, thrust::maximum<int>(), ExtendedDifference(p_tmp_perm));

		cudaDeviceSynchronize();

		if (tmp_half_bandwidth < m_half_bandwidth) {
			m_half_bandwidth = tmp_half_bandwidth;
			m_perm           = tmp_perm;
		}
	}

}

void
RCM_UM::execute_cusp()
{
	IntVector        row_indices(m_nnz);

	IntVector        tmp_row_offsets(m_n + 1);
	IntVector        tmp_row_indices(m_nnz << 1);
	IntVector        tmp_column_indices(m_nnz << 1);

	offsets_to_indices(m_row_offsets, row_indices);

	{
		int numThreads = m_nnz, numBlockX = 1, numBlockY = 1;

		kernelConfigAdjust(numThreads, numBlockX, numBlockY, BLOCK_SIZE, MAX_GRID_DIMENSION);
		dim3 grids(numBlockX, numBlockY);

		const int*       p_row_indices = thrust::raw_pointer_cast(&row_indices[0]);
		const int*       p_column_indices = thrust::raw_pointer_cast(&m_column_indices[0]);
		int *            p_new_row_indices = thrust::raw_pointer_cast(&tmp_row_indices[0]);
		int *            p_new_column_indices = thrust::raw_pointer_cast(&tmp_column_indices[0]);
		device::generalToSymmetric<<<grids, numThreads>>>(m_nnz, p_row_indices, p_column_indices, p_new_row_indices, p_new_column_indices);

		thrust::sort_by_key(tmp_row_indices.begin(), tmp_row_indices.end(), tmp_column_indices.begin());
		indices_to_offsets(tmp_row_indices, tmp_row_offsets);

		m_half_bandwidth = m_half_bandwidth_original = thrust::inner_product(row_indices.begin(), row_indices.end(), m_column_indices.begin(), 0, thrust::maximum<int>(), Difference());

		thrust::sequence(m_perm.begin(), m_perm.end());
		cudaDeviceSynchronize();
	}

	CuspIntVectorD tmp_perm(m_n);
	CuspIntVectorD cusp_row_indices(m_nnz << 1);

	cusp::csr_matrix<int, double, cusp::device_memory> d_cuspA(m_n, m_n, m_nnz << 1);
	thrust::copy(tmp_row_offsets.begin(), tmp_row_offsets.end(), d_cuspA.row_offsets.begin());
	thrust::copy(tmp_column_indices.begin(), tmp_column_indices.end(), d_cuspA.column_indices.begin());
	thrust::fill(d_cuspA.values.begin(), d_cuspA.values.end(), 1.0);
	cusp::graph::symmetric_rcm(d_cuspA, tmp_perm);

	offsets_to_indices(d_cuspA.row_offsets, cusp_row_indices);
	int tmp_half_bandwidth = thrust::inner_product(cusp_row_indices.begin(), cusp_row_indices.end(), d_cuspA.column_indices.begin(), 0, thrust::maximum<int>(), Difference());

	if (m_half_bandwidth > tmp_half_bandwidth) {
		m_half_bandwidth = tmp_half_bandwidth;
		thrust::copy(tmp_perm.begin(), tmp_perm.end(), m_perm.begin());
	}
}

} // end namespace rcm

#endif
