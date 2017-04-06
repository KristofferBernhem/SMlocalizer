
// KernelDevelopment.filterImage
extern "C" __global__  void filterKernel( int* data, int dataLen0, int frameWidth, int frameHeight,  double* kernel, int kernelLen0, int kernelSize,  int* output, int outputLen0);

// KernelDevelopment.filterImage
extern "C" __global__  void filterKernel( int* data, int dataLen0, int frameWidth, int frameHeight,  double* kernel, int kernelLen0, int kernelSize,  int* output, int outputLen0)
{
	int num = blockIdx.x + gridDim.x * blockIdx.y;
	if (num < dataLen0 / (frameWidth * frameHeight))
	{
		int num2 = num * frameWidth * frameHeight;
		for (int i = 0; i < frameWidth; i++)
		{
			for (int j = 0; j < frameHeight; j++)
			{
				int num3 = i + j * frameHeight + num2;
				for (int k = -kernelSize / 2; k <= kernelSize / 2; k++)
				{
					if (i + k < frameWidth && i + k >= 0)
					{
						for (int l = -kernelSize / 2; l <= kernelSize / 2; l++)
						{
							if (j + l < frameHeight && j + l >= 0)
							{
								int num4 = i + k + (j + l) * frameHeight + num2;
								int num5 = k + kernelSize / 2 + (l + kernelSize / 2) * kernelSize;
								output[(num3)] += (int)((double)data[(num4)] * kernel[(num5)]);
							}
						}
					}
				}
			}
		}
		for (int m = num * frameHeight * frameWidth; m < (num + 1) * frameHeight * frameWidth; m++)
		{
			if (output[(m)] < 0)
			{
				output[(m)] = 0;
			}
		}
	}
}
