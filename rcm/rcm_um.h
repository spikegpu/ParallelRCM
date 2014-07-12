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

	  for (int i = 1; i <= maximum - minimum; i++)
		  bucket[i] += bucket[i-1];

	  for (int * key_iterator = key_end - 1; key_iterator >= key_begin; key_iterator --) {
		  int tmp_idx = (--bucket[(*key_iterator) - minimum]);
		  tmp_key[tmp_idx]   = *key_iterator;
		  tmp_value[tmp_idx] = *(value_begin + (key_iterator - key_begin));
	  }

	  for (int * key_iterator = key_begin; key_iterator < key_end; key_iterator ++) {
		  *key_iterator = tmp_key[key_iterator - key_begin];
		  value_begin[key_iterator - key_begin] = tmp_value[key_iterator - key_begin];
	  }
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
	IntVector        visited(m_n);
	BoolVector       tried(m_n, false);

	IntVector        updated_by(m_n);
	IntVector        levels(m_n);

	const int ITER_COUNT = 5;

	IntVector        row_indices(m_nnz);

	IntVector        tmp_row_offsets(m_n + 1);
	IntVector        tmp_row_indices(m_nnz << 1);
	IntVector        tmp_column_indices(m_nnz << 1);
	IntVector        degrees(m_n + 1);
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
		device::generalToSymmetric<<<grids, numThreads>>>(m_nnz, p_row_indices, p_column_indices, p_new_row_indices, p_new_column_indices);

		thrust::sort_by_key(tmp_row_indices.begin(), tmp_row_indices.end(), tmp_column_indices.begin());
		indices_to_offsets(tmp_row_indices, tmp_row_offsets);
		thrust::adjacent_difference(tmp_row_offsets.begin(), tmp_row_offsets.end(), degrees.begin());

		m_half_bandwidth = m_half_bandwidth_original = thrust::inner_product(row_indices.begin(), row_indices.end(), m_column_indices.begin(), 0, thrust::maximum<int>(), Difference());

		thrust::sequence(m_perm.begin(), m_perm.end());
		cudaDeviceSynchronize();
	}

	IntVector tmp_reordering(m_n);
	IntVector tmp_perm(m_n);

	int numBlockX = m_n, numBlockY = 1;
	kernelConfigAdjust(numBlockX, numBlockY, MAX_GRID_DIMENSION);
	dim3 grids(numBlockX, numBlockY);

	for (int i = 0; i < ITER_COUNT; i++) {
		int S = 0;
		
		if (i > 0) {
			while(tried[S])
				S = rand() % m_n;
		}

		tried[S] = true;


		thrust::fill(frontier.begin(), frontier.end(), false);
		thrust::fill(visited.begin(),  visited.end(),  0);
		thrust::sequence(tmp_reordering.begin(), tmp_reordering.end());
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
						levels[j] = l + 1;
						updated_by[j] = -1;
						last = j;
						*p_has_frontier = true;
						tried[j] = true;
						break;
					}

				continue;
			}

			device::achieveLevels<<<grids, 64>>>(m_n, p_row_offsets, p_column_indices, p_frontier, p_visited, p_updated_by, p_levels, p_has_frontier);
			cudaDeviceSynchronize();
		}

		thrust::copy(degrees.begin() + 1, degrees.end(), tmp_degrees.begin());

#if 0
		thrust::sort(thrust::make_zip_iterator(thrust::make_tuple(levels.begin(), updated_by.begin(), tmp_degrees.begin(), tmp_reordering.begin() )),
				     thrust::make_zip_iterator(thrust::make_tuple(levels.end(), updated_by.end(), tmp_degrees.end(), tmp_reordering.end())),
					 TupleCompare()
				);
#endif
		thrust::sort_by_key(levels.begin(), levels.end(), tmp_reordering.begin());


		thrust::scatter(thrust::make_counting_iterator(0),
				       thrust::make_counting_iterator((int)(m_n)),
					   tmp_reordering.begin(),
					   tmp_perm.begin());

		cudaDeviceSynchronize();

		int *p_tmp_perm       = thrust::raw_pointer_cast(&tmp_perm[0]);

#if 0
		IntVectorH h_updated_by(p_updated_by, p_updated_by + m_n);
		IntVectorH h_levels(p_levels, p_levels + m_n);
		IntVectorH h_tmp_reordering(p_tmp_reordering, p_tmp_reordering + m_n);
		IntVectorH h_tmp_perm(p_tmp_perm, p_tmp_perm + m_n);

		int *p_h_updated_by = thrust::raw_pointer_cast(&h_updated_by[0]);
		int *p_h_tmp_reordering = thrust::raw_pointer_cast(&h_tmp_reordering[0]);

		int start_idx = 0;
		for (int l = 0; l < m_n; l++) {
			if (start_idx >= m_n) break;
			if (h_updated_by[start_idx] == -1) {
				start_idx ++;
				continue;
			}

			int l2;
			int tmp_level = h_levels[start_idx];
			for (l2 = start_idx + 1; l2 < m_n; l2++)
				if (h_levels[l2] != tmp_level)
					break;

			int maximum = 0, minimum = (int)(m_n);
			for (int l3 = start_idx; l3 < l2; l3 ++) {
				int tmp_num = (h_updated_by[l3] = h_tmp_perm[h_updated_by[l3]]);
				if (maximum < tmp_num)
					maximum = tmp_num;
				if (minimum > tmp_num)
					minimum = tmp_num;
			}

			local_sort(p_h_updated_by + start_idx, p_h_updated_by + l2, minimum, maximum, p_h_tmp_reordering + start_idx);

			for (int l3 = start_idx; l3 < l2; l3++)
				h_tmp_perm[h_tmp_reordering[l3]] = l3;

			start_idx = l2;
		}
		thrust::copy(h_tmp_perm.begin(), h_tmp_perm.end(), p_tmp_perm);

		int *p_perm = thrust::raw_pointer_cast(&tmp_perm[0]);
#endif
		int tmp_half_bandwidth = thrust::inner_product(row_indices.begin(), row_indices.end(), m_column_indices.begin(), 0, thrust::maximum<int>(), ExtendedDifference(p_tmp_perm));

		if (tmp_half_bandwidth < m_half_bandwidth) {
			m_half_bandwidth = tmp_half_bandwidth;
			m_perm           = tmp_perm;
		}

		cudaDeviceSynchronize();

	}

}

void
RCM_UM::execute_cusp()
{
	IntVector        row_indices(m_nnz);

	IntVector        tmp_row_offsets(m_n + 1);
	IntVector        tmp_row_indices(m_nnz << 1);
	IntVector        tmp_column_indices(m_nnz << 1);
	IntVector        degrees(m_n + 1);
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
		device::generalToSymmetric<<<grids, numThreads>>>(m_nnz, p_row_indices, p_column_indices, p_new_row_indices, p_new_column_indices);

		thrust::sort_by_key(tmp_row_indices.begin(), tmp_row_indices.end(), tmp_column_indices.begin());
		indices_to_offsets(tmp_row_indices, tmp_row_offsets);
		thrust::adjacent_difference(tmp_row_offsets.begin(), tmp_row_offsets.end(), degrees.begin());

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
