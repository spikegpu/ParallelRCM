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

	struct empty_row_functor
	{
		typedef bool result_type;
		typedef typename thrust::tuple<int, int>       IntTuple;
			__host__ __device__
			bool operator()(const IntTuple& t) const
			{
				const int a = thrust::get<0>(t);
				const int b = thrust::get<1>(t);

				return a != b;
			}
	};

class RCM_base
{
protected:
	int            m_half_bandwidth;
	int            m_half_bandwidth_original;
	int            m_max_iteration_count;
	int            m_iteration_count;
	double         m_time_pre;
	double         m_time_bfs;
	double         m_time_node_order;

	size_t         m_n;
	size_t         m_nnz;

	typedef typename thrust::tuple<int, int, int, int>       IntTuple;
	typedef typename thrust::tuple<int, int, int>            IntTriple;


	template <typename IVector>
	static void offsets_to_indices(const IVector& offsets, IVector& indices)
	{
		// convert compressed row offsets into uncompressed row indices
		thrust::fill(indices.begin(), indices.end(), 0);
		thrust::scatter_if( thrust::counting_iterator<int>(0),
				thrust::counting_iterator<int>(offsets.size()-1),
				offsets.begin(),
                    	thrust::make_transform_iterator(
                                thrust::make_zip_iterator( thrust::make_tuple( offsets.begin(), offsets.begin()+1 ) ),
                                empty_row_functor()),
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

	template <typename IVectorIterator>
	static void indices_to_offsets(IVectorIterator begin, IVectorIterator end, IVectorIterator output, int start_value, int end_value) 
	{
		thrust::lower_bound(begin,
				end,
				thrust::counting_iterator<int>(start_value),
				thrust::counting_iterator<int>(end_value + 1),
				output);
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

	struct TripleCompare
	{
		inline
			__host__ __device__
			bool operator() (IntTriple a, IntTriple b) const
			{ 
				int a0 = thrust::get<0>(a), b0 = thrust::get<0>(b);
				if (a0 != b0) return a0 < b0;
				return thrust::get<1>(a) < thrust::get<1>(b);
			}
	};


	struct EqualTo: public thrust::unary_function<int, int>
	{
		int m_max;
		EqualTo(int m): m_max(m) {}

		inline 
			__host__ __device__
			bool operator() (const int &a) {
				return a == m_max;
			}
	};

	int getHalfBandwidth() const         {return m_half_bandwidth;}
	int getHalfBandwidthOriginal() const {return m_half_bandwidth_original;}
	int getIterationCount() const        {return m_iteration_count;}
	double getTimePreprocessing() const  {return m_time_pre;}
	double getTimeBFS() const            {return m_time_bfs;}
	double getTimeNodeOrder() const      {return m_time_node_order;}

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
      const DoubleVectorH& values,
	  int                  iteration_count)
  : m_row_offsets(row_offsets),
    m_column_indices(column_indices),
    m_values(values)
  {
    size_t n = row_offsets.size() - 1;
    m_perm.resize(n);
    m_n               = n;
    m_nnz             = m_values.size();
	m_max_iteration_count = iteration_count;
	m_iteration_count = 0;
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

	const int MAX_NUM_TRIAL = m_max_iteration_count;

	BoolVectorH tried(m_n, false);
	tried[0] = true;

	m_perm.resize(m_n);
	thrust::sequence(m_perm.begin(), m_perm.end());

	IntVectorH levels(m_n);
	IntVectorH ori_degrees(m_n);

	thrust::transform(m_row_offsets.begin() + 1, m_row_offsets.end(), m_row_offsets.begin(), ori_degrees.begin(), thrust::minus<int>());

	int S;
	int p_max_level;

	IntVectorH pushed(m_n, -1);

	for (int trial_num = 0; trial_num < MAX_NUM_TRIAL ; trial_num++)
	{
		m_iteration_count = trial_num + 1;
		std::queue<int> q;
		std::priority_queue<NodeType, std::vector<NodeType>, CompareValue > pq;
		int max_level = 0;

		int tmp_node;

		int left_cnt = m_n;
		int j = 0, last = 0;

		if (trial_num > 0) {
			IntIterator max_level_iter = thrust::max_element(levels.begin(), levels.end());
			int max_count = thrust::count(levels.begin(), levels.end(), max_level);

			if (max_count > 1) {
				IntVectorH max_level_vertices(max_count);
				IntVectorH max_level_valence(max_count);

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
				tmp_node = max_level_vertices[min_valence_pos];
			} else
				tmp_node = max_level_iter - levels.begin();

			while(tried[tmp_node])
				tmp_node = (tmp_node + 1) % m_n;
		} else
			tmp_node = thrust::min_element(ori_degrees.begin(), ori_degrees.end()) - ori_degrees.begin();

		S = tmp_node;

		pushed[S] = trial_num;
		tried[S] = true;
		q.push(S);
		levels[S] = 0;

		while(left_cnt--) {
			if(q.empty()) {
				left_cnt++;
				int i;

				for(i = last; i < m_n; i++) {
					if(pushed[i] != trial_num) {
						q.push(i);
						pushed[i] = trial_num;
						last = i;
						levels[i] = (++max_level);
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
				if(pushed[target_node] != trial_num) {
					pushed[target_node] = trial_num;
					pq.push(thrust::make_tuple(target_node, tmp_row_offsets[target_node + 1] - tmp_row_offsets[target_node]));
					max_level = levels[target_node] = levels[tmp_node] + 1;
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

		if (trial_num > 0) {
			if (p_max_level >= max_level)
				break;

			const double stop_ratio = 0.01;
			double max_level_ratio = 1.0 * (max_level - p_max_level) / p_max_level;

			if (max_level_ratio < stop_ratio)
				break;

			if (p_max_level < max_level)
				p_max_level = max_level;
		} else
			p_max_level = max_level;
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
