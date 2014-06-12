#ifndef COMMON_H
#define COMMON_H



#define ALWAYS_ASSERT

#ifdef WIN32
typedef long long int64_t;
#endif



// ----------------------------------------------------------------------------
// If ALWAYS_ASSERT is defined, we make sure that  assertions are triggered 
// even if NDEBUG is defined.
// ----------------------------------------------------------------------------
#ifdef ALWAYS_ASSERT
// If NDEBUG is actually defined, remember this so
// we can restore it.
#  ifdef NDEBUG
#    define NDEBUG_ACTIVE
#    undef NDEBUG
#  endif
// Include the assert.h header file here so that it can
// do its stuff while NDEBUG is guaranteed to be disabled.
#  include <assert.h>
// Restore NDEBUG mode if it was active.
#  ifdef NDEBUG_ACTIVE
#    define NDEBUG
#    undef NDEBUG_ACTIVE
#  endif
#else
// Include the assert.h header file using whatever the
// current definition of NDEBUG is.
#  include <assert.h>
#endif

# include <memory.h>
# include <thrust/device_ptr.h>
# include <thrust/system/cuda/execution_policy.h>


// ----------------------------------------------------------------------------


namespace rcm {

const unsigned int BLOCK_SIZE = 512;

const unsigned int MAX_GRID_DIMENSION = 32768;

inline
void kernelConfigAdjust(int &numThreads, int &numBlockX, const int numThreadsMax) {
	if (numThreads > numThreadsMax) {
		numBlockX = (numThreads + numThreadsMax - 1) / numThreadsMax;
		numThreads = numThreadsMax;
	}
}

inline
void kernelConfigAdjust(int &numThreads, int &numBlockX, int &numBlockY, const int numThreadsMax, const int numBlockXMax) {
	kernelConfigAdjust(numThreads, numBlockX, numThreadsMax);
	if (numBlockX > numBlockXMax) {
		numBlockY = (numBlockX + numBlockXMax - 1) / numBlockXMax;
		numBlockX = numBlockXMax;
	}
}

template <typename T>
class ManagedVector {
	T*     m_p_array;
	size_t m_size;

public:
	typedef typename thrust::device_ptr<T> iterator;

	ManagedVector(): m_p_array(0), m_size(0) {}
	ManagedVector(size_t n): m_size(n) {
		cudaMallocManaged(&m_p_array, sizeof(T) * n);
	}
	ManagedVector(size_t n, T val): m_size(n) {
		cudaMallocManaged(&m_p_array, sizeof(T) * n);
		thrust::fill(thrust::cuda::par, m_p_array, m_p_array + n, val);
		cudaDeviceSynchronize();
	}
	ManagedVector(const ManagedVector &a): m_size(a.m_size) {
		cudaMallocManaged(&m_p_array, sizeof(T) * a.m_size);
		thrust::copy(thrust::cuda::par, a.m_p_array, a.m_p_array + a.m_size, m_p_array);
		cudaDeviceSynchronize();
	}
	~ManagedVector() {cudaFree(m_p_array);}

	ManagedVector& operator=(const ManagedVector &a) {
		if (m_size < a.m_size) {
			m_size = a.m_size;
			cudaFree(m_p_array);
			cudaMallocManaged(&m_p_array, sizeof(T) * a.m_size);
			thrust::copy(thrust::cuda::par, a.m_p_array, a.m_p_array + a.m_size, m_p_array);
			cudaDeviceSynchronize();
		} else {
			m_size = a.m_size;
			thrust::copy(thrust::cuda::par, a.m_p_array, a.m_p_array + a.m_size, m_p_array);
			cudaDeviceSynchronize();
		}

		return *this;
	}

	thrust::device_ptr<T> begin() const {return thrust::device_pointer_cast(&m_p_array[0]);}
	thrust::device_ptr<T> end()   const {return thrust::device_pointer_cast(&m_p_array[m_size]);}

	T& operator[](size_t n)    {return m_p_array[n];}
	const T& operator[](size_t n)  const  {return m_p_array[n];}

	size_t size() const {return m_size;}

	void resize(size_t n)  {
		if (m_size >= n) m_size = n;
		else {
			T *p_tmp;
			cudaMallocManaged(&p_tmp, sizeof(T) * n);

			if (m_size > 0) {
				thrust::copy(thrust::cuda::par, m_p_array, m_p_array + m_size, p_tmp);
				cudaDeviceSynchronize();
			}

			m_size = n;
			cudaFree(m_p_array);

			m_p_array = p_tmp;
		}
	}

};

// -----------------------------------------------------------------------------
// Convert matrix from COO to CSR format
// -----------------------------------------------------------------------------
template <typename Ivec, typename Rvec>
void coo2csr(const int    n_row,
             const int    n_col,
             const int    nnz,
             const int    Ai[],
             const int    Aj[],
             const double Ax[],
             Ivec&        Bp,
             Ivec&        Bj,
             Rvec&        Bx)
{
  //compute number of non-zero entries per row of A 
  for (int i = 0; i <= n_row; i++)
    Bp[i] = 0;

  for (int i = 0; i < nnz; i++)
    Bp[Ai[i]]++;

  //cumsum the nnz per row to get Bp[]
  for(int i = 0, cumsum = 0; i < n_row; i++) {
    int temp = Bp[i];
    Bp[i] = cumsum;
    cumsum += temp;
  }
  Bp[n_row] = nnz; 

  //write Aj,Ax into Bj,Bx
  for(int i = 0; i < nnz; i++) {
    int row  = Ai[i];
    int dest = Bp[row];

    Bj[dest] = Aj[i];
    Bx[dest] = Ax[i];

    Bp[row]++;
  }

  for(int i = 0, last = 0; i <= n_row; i++) {
    int temp = Bp[i];
    Bp[i]  = last;
    last   = temp;
  }
}


} // namespace rcm


#endif
