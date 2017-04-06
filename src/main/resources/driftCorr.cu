
// KernelDevelopment.driftCorr
extern "C" __global__  void runAdd( int* referenceDataSet, int referenceDataSetLen0,  int* targetDataSet, int targetDataSetLen0,  int* shift, int shiftLen0,  float* means, int meansLen0,  int* dimensions, int dimensionsLen0,  double* result, int resultLen0);

// KernelDevelopment.driftCorr
extern "C" __global__  void runAdd( int* referenceDataSet, int referenceDataSetLen0,  int* targetDataSet, int targetDataSetLen0,  int* shift, int shiftLen0,  float* means, int meansLen0,  int* dimensions, int dimensionsLen0,  double* result, int resultLen0)
{
	int num = blockIdx.x + gridDim.x * blockIdx.y;
	if (num < dimensions[(0)] * dimensions[(1)] * dimensions[(2)] && referenceDataSet[(num)] > 0)
	{
		short num2 = (short)(num / (dimensions[(0)] * dimensions[(1)]));
		short num3 = (short)(num - (int)num2 * dimensions[(0)] * dimensions[(1)]);
		short num4 = (short)((int)num3 / dimensions[(0)]);
		num3 -= (short)((int)num4 * dimensions[(0)]);
		for (int i = 0; i < shiftLen0 / 3; i++)
		{
			if ((int)num3 - shift[(i * 3)] >= 0 && (int)num3 - shift[(i * 3)] < dimensions[(0)] && (int)num4 - shift[(i * 3 + 1)] >= 0 && (int)num4 - shift[(i * 3 + 1)] < dimensions[(1)] && (int)num2 - shift[(i * 3 + 2)] >= 0 && (int)num2 - shift[(i * 3 + 2)] < dimensions[(2)])
			{
				int num5 = (int)num3 - shift[(i * 3)] + ((int)num4 - shift[(i * 3 + 1)]) * dimensions[(0)] + ((int)num2 - shift[(i * 3 + 2)]) * dimensions[(1)] * dimensions[(0)];
				result[(i)] += (double)(((float)referenceDataSet[(num)] - means[(0)]) * ((float)targetDataSet[(num5)] - means[(1)]));
			}
		}
	}
}
