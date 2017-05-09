
// KernelDevelopment.driftCorr
extern "C" __global__  void runAdd( int* referenceDataSet, int referenceDataSetLen0,  int* targetDataSet, int targetDataSetLen0,  int* shift, int shiftLen0,  float* means, int meansLen0,  int* dimensions, int dimensionsLen0,  double* result, int resultLen0);

// KernelDevelopment.driftCorr
extern "C" __global__  void runAdd( int* referenceDataSet, int referenceDataSetLen0,  int* targetDataSet, int targetDataSetLen0,  int* shift, int shiftLen0,  float* means, int meansLen0,  int* dimensions, int dimensionsLen0,  double* result, int resultLen0)
{
	int num = blockIdx.x + gridDim.x * blockIdx.y;
	if (num < dimensions[(0)] * dimensions[(1)] * dimensions[(2)])
	{
		int num2 = num / (dimensions[(0)] * dimensions[(1)]);
		int num3 = num - num2 * dimensions[(0)] * dimensions[(1)];
		int num4 = num3 / dimensions[(0)];
		num3 -= num4 * dimensions[(0)];
		for (int i = 0; i < shiftLen0 / 3; i++)
		{
			if (num3 - shift[(i * 3)] >= 0 && num3 - shift[(i * 3)] < dimensions[(0)] && num4 - shift[(i * 3 + 1)] >= 0 && num4 - shift[(i * 3 + 1)] < dimensions[(1)] && num2 - shift[(i * 3 + 2)] >= 0 && num2 - shift[(i * 3 + 2)] < dimensions[(2)])
			{
				int num5 = num3 - shift[(i * 3)] + (num4 - shift[(i * 3 + 1)]) * dimensions[(0)] + (num2 - shift[(i * 3 + 2)]) * dimensions[(1)] * dimensions[(0)];
				result[(i)] += (double)(((float)referenceDataSet[(num)] - means[(0)]) * ((float)targetDataSet[(num5)] - means[(1)]));
			}
		}
	}
}
