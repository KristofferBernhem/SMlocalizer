
// KernelDevelopment.driftCorr
extern "C" __global__  void runAdd(int n,  int* referenceDataSet, int referenceDataSetLen0,  int* targetDataSet, int targetDataSetLen0,  int* shift, int shiftLen0,  float* means, int meansLen0,  int* dimensions, int dimensionsLen0,  double* result, int resultLen0);

// KernelDevelopment.driftCorr
extern "C" __global__  void runAdd(int n,  int* referenceDataSet, int referenceDataSetLen0,  int* targetDataSet, int targetDataSetLen0,  int* shift, int shiftLen0,  float* means, int meansLen0,  int* dimensions, int dimensionsLen0,  double* result, int resultLen0)
{
	int num = blockIdx.x * blockDim.x + threadIdx.x;
	int num2 = blockDim.x * gridDim.x;
	for (int i = num; i < n; i += num2)
	{
		int num3 = i / (dimensions[(0)] * dimensions[(1)]);
		int num4 = i - num3 * dimensions[(0)] * dimensions[(1)];
		int num5 = num4 / dimensions[(0)];
		num4 -= num5 * dimensions[(0)];
		for (int j = 0; j < shiftLen0 / 3; j++)
		{
			if (num4 - shift[(j * 3)] >= 0 && num4 - shift[(j * 3)] < dimensions[(0)] && num5 - shift[(j * 3 + 1)] >= 0 && num5 - shift[(j * 3 + 1)] < dimensions[(1)] && num3 - shift[(j * 3 + 2)] >= 0 && num3 - shift[(j * 3 + 2)] < dimensions[(2)])
			{
				int num6 = num4 - shift[(j * 3)] + (num5 - shift[(j * 3 + 1)]) * dimensions[(0)] + (num3 - shift[(j * 3 + 2)]) * dimensions[(1)] * dimensions[(0)];
				result[(j)] += (double)(((float)referenceDataSet[(i)] - means[(0)]) * ((float)targetDataSet[(num6)] - means[(1)]));
			}
		}
	}
}
