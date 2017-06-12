
// KernelDevelopment.filterImage
extern "C" __global__  void filterKernel(int n,  int* data, int dataLen0, int frameWidth, int frameHeight,  double* kernel, int kernelLen0, int kernelSize,  int* output, int outputLen0);

// KernelDevelopment.filterImage
extern "C" __global__  void filterKernel(int n,  int* data, int dataLen0, int frameWidth, int frameHeight,  double* kernel, int kernelLen0, int kernelSize,  int* output, int outputLen0)
{
	int num = blockIdx.x * blockDim.x + threadIdx.x;
	int num2 = blockDim.x * gridDim.x;
	for (int i = num; i < n; i += num2)
	{
		int num3 = i * frameWidth * frameHeight;
		for (int j = 0; j < frameWidth; j++)
		{
			for (int k = 0; k < frameHeight; k++)
			{
				int num4 = j + k * frameHeight + num3;
				for (int l = -kernelSize / 2; l <= kernelSize / 2; l++)
				{
					if (j + l < frameWidth && j + l >= 0)
					{
						for (int m = -kernelSize / 2; m <= kernelSize / 2; m++)
						{
							if (k + m < frameHeight && k + m >= 0)
							{
								int num5 = j + l + (k + m) * frameHeight + num3;
								int num6 = l + kernelSize / 2 + (m + kernelSize / 2) * kernelSize;
								output[(num4)] += (int)((double)data[(num5)] * kernel[(num6)]);
							}
						}
					}
				}
			}
		}
		for (int num7 = i * frameHeight * frameWidth; num7 < (i + 1) * frameHeight * frameWidth; num7++)
		{
			if (output[(num7)] < 0)
			{
				output[(num7)] = 0;
			}
		}
	}
}
