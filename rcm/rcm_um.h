#ifndef RCM_UM_H
#define RCM_UM_H

#include <rcm/common.h>
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

  IntVector      m_row_offsets;
  IntVector      m_column_indices;
  DoubleVector   m_values;

  IntVector      m_perm;

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


		thrust::sort(thrust::make_zip_iterator(thrust::make_tuple(levels.begin(), updated_by.begin(), tmp_degrees.begin(), tmp_reordering.begin() )),
				     thrust::make_zip_iterator(thrust::make_tuple(levels.end(), updated_by.end(), tmp_degrees.end(), tmp_reordering.end())),
					 TupleCompare()
				);


		thrust::scatter(thrust::make_counting_iterator(0),
				       thrust::make_counting_iterator((int)(m_n)),
					   tmp_reordering.begin(),
					   tmp_perm.begin());

		int *p_perm = thrust::raw_pointer_cast(&tmp_perm[0]);
		int tmp_half_bandwidth = thrust::inner_product(row_indices.begin(), row_indices.end(), m_column_indices.begin(), 0, thrust::maximum<int>(), ExtendedDifference(p_perm));

		if (tmp_half_bandwidth < m_half_bandwidth) {
			m_half_bandwidth = tmp_half_bandwidth;
			m_perm           = tmp_perm;
		}

		cudaDeviceSynchronize();

	}

}

} // end namespace rcm

#endif
