#ifndef RCM_H
#define RCM_H

#include <rcm/common.h>
#include <rcm/exception.h>
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

#include <queue>

extern "C" {
#include "mm_io/mm_io.h"
}

namespace rcm {

class RCM_base
{
protected:
	int            m_half_bandwidth;
	int            m_half_bandwidth_original;

	size_t         m_n;
	size_t         m_nnz;

	typedef typename thrust::tuple<int, int, int, int>       IntTuple;

	template <typename IVector>
	static void offsets_to_indices(const IVector& offsets, IVector& indices)
	{
		// convert compressed row offsets into uncompressed row indices
		thrust::fill(indices.begin(), indices.end(), 0);
		thrust::scatter( thrust::counting_iterator<int>(0),
				thrust::counting_iterator<int>(offsets.size()-1),
				offsets.begin(),
				indices.begin());
		thrust::inclusive_scan(indices.begin(), indices.end(), indices.begin(), thrust::maximum<int>());
	}

	template <typename IVector>
	static void indices_to_offsets(const IVector& indices, IVector& offsets)
	{
		// convert uncompressed row indices into compressed row offsets
		thrust::lower_bound(indices.begin(),
				indices.end(),
				thrust::counting_iterator<int>(0),
				thrust::counting_iterator<int>(offsets.size()),
				offsets.begin());
	}

public:

	virtual ~RCM_base() {}

	struct Difference: public thrust::binary_function<int, int, int>
	{
		inline
			__host__ __device__
			int operator() (const int &a, const int &b) const {
				return abs(a-b);
			}
	};

	struct ExtendedDifference: public thrust::binary_function<int, int, int>
	{
		int *m_perm;

		ExtendedDifference(int *perm): m_perm(perm) {}
		inline
			__host__ __device__
			int operator() (const int &a, const int &b) const {
				return abs(m_perm[a]-m_perm[b]);
			}
	};

	struct TupleCompare
	{
		inline
			__host__ __device__
			bool operator() (IntTuple a, IntTuple b) const
			{
				int a_level = thrust::get<0>(a), b_level = thrust::get<0>(b);
				if (a_level != b_level) return a_level < b_level;
				int a_updated_by = thrust::get<1>(a), b_updated_by = thrust::get<1>(b);
				if (a_updated_by != b_updated_by) return a_updated_by < b_updated_by;
				return thrust::get<2>(a) < thrust::get<2>(b);
			}
	};

	int getHalfBandwidth() const         {return m_half_bandwidth;}
	int getHalfBandwidthOriginal() const {return m_half_bandwidth_original;}

    virtual void execute() = 0;
};

class RCM: public RCM_base
{
private:
  typedef typename thrust::host_vector<int>       IntVectorH;
  typedef typename thrust::host_vector<double>    DoubleVectorH;
  typedef typename thrust::host_vector<bool>      BoolVectorH;

  typedef typename IntVectorH::iterator           IntIterator;
  typedef typename thrust::tuple<IntIterator, IntIterator>     IntIteratorTuple;
  typedef typename thrust::zip_iterator<IntIteratorTuple>      EdgeIterator;

  typedef typename thrust::tuple<int, int>        NodeType;

  IntVectorH     m_row_offsets;
  IntVectorH     m_column_indices;
  DoubleVectorH  m_values;

  IntVectorH     m_perm;

  void buildTopology(EdgeIterator&      begin,
                     EdgeIterator&      end,
				     int                node_begin,
				     int                node_end,
                     IntVectorH&        row_offsets,
                     IntVectorH&        column_indices);

  struct CompareValue
  {
	  inline
	  bool operator() (NodeType a, NodeType b) const {
		  return thrust::get<1>(a) > thrust::get<1>(b);
	  }
  };

public:
  RCM(const IntVectorH&    row_offsets,
      const IntVectorH&    column_indices,
      const DoubleVectorH& values)
  : m_row_offsets(row_offsets),
    m_column_indices(column_indices),
    m_values(values)
  {
    size_t n = row_offsets.size() - 1;
    m_perm.resize(n);
    m_n   = n;
    m_nnz = m_values.size();
  }

  ~RCM() {}

  void execute();
};

void
RCM::execute()
{
	IntVectorH tmp_reordering(m_n);
	IntVectorH tmp_perm(m_n);

	thrust::sequence(tmp_reordering.begin(), tmp_reordering.end());

	IntVectorH row_indices(m_nnz);
	IntVectorH tmp_row_indices(m_nnz << 1);
	IntVectorH tmp_column_indices(m_nnz << 1);
	IntVectorH tmp_row_offsets(m_n + 1);
	offsets_to_indices(m_row_offsets, row_indices);

	m_half_bandwidth_original = m_half_bandwidth = thrust::inner_product(row_indices.begin(), row_indices.end(), m_column_indices.begin(), 0, thrust::maximum<int>(), Difference());

	EdgeIterator begin = thrust::make_zip_iterator(thrust::make_tuple(row_indices.begin(), m_column_indices.begin()));
	EdgeIterator end   = thrust::make_zip_iterator(thrust::make_tuple(row_indices.end(),   m_column_indices.end()));
	buildTopology(begin, end, 0, m_n, tmp_row_offsets, tmp_column_indices);

	const int MAX_NUM_TRIAL = 5;

	BoolVectorH tried(m_n, false);
	tried[0] = true;

	int last_tried = 0;

	m_perm.resize(m_n);
	thrust::sequence(m_perm.begin(), m_perm.end());

	for (int trial_num = 0; trial_num < MAX_NUM_TRIAL ; trial_num++)
	{
		std::queue<int> q;
		std::priority_queue<NodeType, std::vector<NodeType>, CompareValue > pq;

		int tmp_node;
		BoolVectorH pushed(m_n, false);

		int left_cnt = m_n;
		int j = 0, last = 0;

		if (trial_num > 0) {

			if (trial_num < MAX_NUM_TRIAL) {
				tmp_node = rand() % m_n;

				while(tried[tmp_node])
					tmp_node = rand() % m_n;
			} else {
				if (last_tried >= m_n - 1) {
					fprintf(stderr, "All possible starting points have been tried in RCM\n");
					break;
				}
				for (tmp_node = last_tried+1; tmp_node < m_n; tmp_node++)
					if (!tried[tmp_node]) {
						last_tried = tmp_node;
						break;
					}
			}

			pushed[tmp_node] = true;
			tried[tmp_node] = true;
			q.push(tmp_node);
		}

		while(left_cnt--) {
			if(q.empty()) {
				left_cnt++;
				int i;

				for(i = last; i < m_n; i++) {
					if(!pushed[i]) {
						q.push(i);
						pushed[i] = true;
						last = i;
						break;
					}
				}
				if(i < m_n) continue;
				fprintf(stderr, "Can never get here!\n");
				return;
			}

			tmp_node = q.front();
			tmp_reordering[j] = tmp_node;
			j++;

			q.pop();

			int start_idx = tmp_row_offsets[tmp_node], end_idx = tmp_row_offsets[tmp_node + 1];

			for (int i = start_idx; i < end_idx; i++)  {
				int target_node = tmp_column_indices[i];
				if(!pushed[target_node]) {
					pushed[target_node] = true;
					pq.push(thrust::make_tuple(target_node, tmp_row_offsets[target_node + 1] - tmp_row_offsets[target_node]));
				}
			}

			while(!pq.empty()) {
				q.push(thrust::get<0>(pq.top()));
				pq.pop();
			}
		}

		thrust::scatter(thrust::make_counting_iterator(0), 
						thrust::make_counting_iterator((int)(m_n)),
						tmp_reordering.begin(),
						tmp_perm.begin());

		{
			int *perm_array = thrust::raw_pointer_cast(&tmp_perm[0]);
			int tmp_bdwidth = thrust::inner_product(row_indices.begin(), row_indices.end(), m_column_indices.begin(), 0, thrust::maximum<int>(), ExtendedDifference(perm_array));

			if (m_half_bandwidth > tmp_bdwidth) {
				m_half_bandwidth = tmp_bdwidth;
				m_perm           = tmp_perm;
			}
		}
	}
}

void
RCM::buildTopology(EdgeIterator&      begin,
                   EdgeIterator&      end,
				   int                node_begin,
				   int                node_end,
                   IntVectorH&        row_offsets,
                   IntVectorH&        column_indices)
{
	if (row_offsets.size() != m_n + 1)
		row_offsets.resize(m_n + 1, 0);
	else
		thrust::fill(row_offsets.begin(), row_offsets.end(), 0);

	IntVectorH row_indices((end - begin) << 1);
	column_indices.resize((end - begin) << 1);
	int actual_cnt = 0;

	for(EdgeIterator edgeIt = begin; edgeIt != end; edgeIt++) {
		int from = thrust::get<0>(*edgeIt), to = thrust::get<1>(*edgeIt);
		if (from != to) {
			row_indices[actual_cnt]        = from;
			column_indices[actual_cnt]     = to;
			row_indices[actual_cnt + 1]    = to;
			column_indices[actual_cnt + 1] = from;
			actual_cnt += 2;
		}
	}
	row_indices.resize(actual_cnt);
	column_indices.resize(actual_cnt);
	// thrust::sort_by_key(row_indices.begin(), row_indices.end(), column_indices.begin());
	{
		int&      nnz = actual_cnt;
		IntVectorH tmp_column_indices(nnz);
		for (int i = 0; i < nnz; i++)
			row_offsets[row_indices[i]] ++;

		thrust::inclusive_scan(row_offsets.begin() + node_begin, row_offsets.begin() + (node_end + 1), row_offsets.begin() + node_begin);

		for (int i = nnz - 1; i >= 0; i--) {
			int idx = (--row_offsets[row_indices[i]]);
			tmp_column_indices[idx] = column_indices[i];
		}
		column_indices = tmp_column_indices;
	}
}

} // end namespace rcm

#endif
