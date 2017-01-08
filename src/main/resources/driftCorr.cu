
// KernelDevelopment.driftCorr
extern "C" __global__  void run( int* firstDataSet, int firstDataSetLen0,  int* secondDataSet, int secondDataSetLen0,  int* maxShift, int maxShiftLen0,  int* stepSize, int stepSizeLen0,  int* numSteps, int numStepsLen0,  int* result, int resultLen0);

// KernelDevelopment.driftCorr
extern "C" __global__  void run( int* firstDataSet, int firstDataSetLen0,  int* secondDataSet, int secondDataSetLen0,  int* maxShift, int maxShiftLen0,  int* stepSize, int stepSizeLen0,  int* numSteps, int numStepsLen0,  int* result, int resultLen0)
{
	int y = blockIdx.y;
	int x = blockIdx.x;
	int num = x + y * gridDim.x;
	if (num < numSteps[(0)] * numSteps[(0)] * numSteps[(1)])
	{
		result[(num)] = 0;
		if (numSteps[(1)] == 1)
		{
			float num2 = (float)(maxShift[(0)] - num / numSteps[(0)] * stepSize[(0)]);
			float num3 = (float)(maxShift[(0)] - num % numSteps[(0)] * stepSize[(0)]);
			for (int i = 0; i < firstDataSetLen0; i += 2)
			{
				for (int j = 0; j < secondDataSetLen0; j += 2)
				{
					if (abs((float)(firstDataSet[(i)] - secondDataSet[(j)]) - num2) < 5.0f && abs((float)(firstDataSet[(i + 1)] - secondDataSet[(j + 1)]) - num3) < 5.0f)
					{
						result[(num)]++;
					}
				}
			}
			return;
		}
		float num4 = (float)(maxShift[(0)] - num / (numSteps[(0)] * numSteps[(1)]) * stepSize[(0)]);
		float num5 = (float)(maxShift[(0)] - num / numSteps[(1)] * stepSize[(0)]);
		float num6 = (float)(maxShift[(1)] - num % numSteps[(1)] * stepSize[(1)]);
		for (int k = 0; k < firstDataSetLen0; k += 3)
		{
			for (int l = 0; l < secondDataSetLen0; l += 3)
			{
				if (abs((float)(firstDataSet[(k)] - secondDataSet[(l)]) - num4) < 5.0f && abs((float)(firstDataSet[(k + 1)] - secondDataSet[(l + 1)]) - num5) < 5.0f && abs((float)(firstDataSet[(k + 2)] - secondDataSet[(l + 2)]) - num6) < 5.0f)
				{
					result[(num)]++;
				}
			}
		}
	}
}
