#include <algorithm>
#include <fstream>
#include <cmath>
#include <string>

extern "C" {
#include "mm_io/mm_io.h"
}

#include <rcm/rcm_um.h>
#include <rcm/common.h>

typedef typename rcm::ManagedVector<int>     IntVector;
typedef typename rcm::ManagedVector<double>  DoubleVector;

int main(int argc, char **argv)
{
  if (argc < 3) {
    std::cout << "Usage:\n  driver input_file output_file" << std::endl;
    return 1;
  }

  // Read matrix from file (COO format)
  MM_typecode matcode;
  int M, N, nnz;
  int* row_i;
  int* col_j;
  double* vals;

  std::cout << "Read MTX file... ";
  int err = mm_read_mtx_crd(argv[1], &M, &N, &nnz, &row_i, &col_j, &vals, &matcode);
  if (err != 0) {
    std::cout << "error: " << err << std::endl;
    return 1;
  }
  std::cout << "M = " << M << " N = " << N << " nnz = " << nnz << std::endl;

  // Switch to 0-based indices
  for (int i = 0; i < nnz; i++) {
    row_i[i]--;
    col_j[i]--;
  }
  // Convert to CSR format and load into thrust vectors.
  IntVector    row_offsets(N + 1);
  IntVector    column_indices(nnz);
  DoubleVector values(nnz);

  std::cout << "Convert to CSR" << std::endl;
  rcm::coo2csr(M, N, nnz, row_i, col_j, vals, row_offsets, column_indices, values);

  // Print thrust vectors
  /*
  std::cout << "Row offsets\n";
  thrust::copy(row_offsets.begin(), row_offsets.end(), std::ostream_iterator<int>(std::cout, "\n"));
  std::cout << "Column indices\n";
  thrust::copy(column_indices.begin(), column_indices.end(), std::ostream_iterator<int>(std::cout, "\n"));
  std::cout << "Values\n";
  thrust::copy(values.begin(), values.end(), std::ostream_iterator<double>(std::cout, "\n"));
  */

  // Run the RCM algorithm
  rcm::RCM_UM algo(row_offsets, column_indices, values, 5);

  std::cout << "Run RCM... ";
  try {
    algo.execute();
  } catch (const rcm::system_error& se) {
    std::cout << "error " << se.reason() << std::endl;
    return 1;
  }
  std::cout << "success" << std::endl;

  std::cout << "Write output file " << argv[2] << std::endl;
  std::ofstream fout;
  fout.open(argv[2]);
  //// algo.print(fout);
  fout.close();

  return 0;
}
