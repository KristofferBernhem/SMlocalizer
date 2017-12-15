
// KernelDevelopment.filterImage
extern "C" __global__  void filterKernel(int n,  int* data, int dataLen0, int frameWidth, int frameHeight,  float* kernel, int kernelLen0, int kernelSize,  int* output, int outputLen0);

// KernelDevelopment.filterImage
extern "C" __global__  void filterKernel(int n,  int* data, int dataLen0, int frameWidth, int frameHeight,  float* kernel, int kernelLen0, int kernelSize,  int* output, int outputLen0)
{
	int num = blockIdx.x * blockDim.x + threadIdx.x;
	int num2 = blockDim.x * gridDim.x;
	for (int i = num; i < n; i += num2)
	{
		int num3 = (int)ceil((double)(kernelSize / 2));
		int num4 = i * frameWidth * frameHeight;
		for (int j = num4; j < (i + 1) * frameWidth * frameHeight; j++)
		{
			float num5 = 0.0f;
			int k = j - num3 * (frameWidth + 1);
			int num6 = 0;
			int num7 = 0;
			while (k <= j + num3 * frameWidth)
			{
				if (k < (i + 1) * frameWidth * frameHeight && k >= i * frameWidth * frameHeight)
				{
					num5 += (float)data[(k)] * kernel[(num7)];
				}
				k++;
				num6++;
				num7++;
				if (num6 == kernelSize)
				{
					k += frameWidth - kernelSize;
					num6 = 0;
				}
			}
			output[(j)] = (int)num5;
		}
	}
}
