
// KernelDevelopment.findMaxima
extern "C" __global__  void run( int* data, int dataLen0, int frameWidth, int frameHeight, int windowWidth, int minLevel, double sqDistance, int minPosPixel, int sizeCenter,  int* Center, int CenterLen0);

// KernelDevelopment.findMaxima
extern "C" __global__  void run( int* data, int dataLen0, int frameWidth, int frameHeight, int windowWidth, int minLevel, double sqDistance, int minPosPixel, int sizeCenter,  int* Center, int CenterLen0)
{
	int num = blockIdx.x + gridDim.x * blockIdx.y;
	if (num < dataLen0 / (frameWidth * frameHeight))
	{
		int num2 = 0;
		int i = 0;
		int num3 = 0;
		bool flag = true;
		int j = num * (frameWidth * frameHeight) + windowWidth / 2 * frameWidth + windowWidth / 2;
		for (i = num * sizeCenter; i < (num + 1) * sizeCenter; i++)
		{
			Center[(i)] = 0;
		}
		i = 0;
		int num4 = 0;
		while (j < (num + 1) * (frameWidth * frameHeight) - windowWidth / 2 * frameWidth)
		{
			if (data[(j)] > minLevel)
			{
				i = 0;
				num3 = j - windowWidth / 2 * (frameWidth + 1);
				flag = true;
				num4 = 0;
				while (num3 <= j + windowWidth / 2 * (frameWidth + 1) && flag)
				{
					if (data[(num3)] > data[(j)])
					{
						flag = false;
					}
					if (data[(num3)] > 0)
					{
						i++;
					}
					num3++;
					num4++;
					if (num4 == windowWidth)
					{
						num3 += frameWidth - windowWidth;
						num4 = 0;
					}
				}
				if (i < minPosPixel)
				{
					flag = false;
				}
				if (flag)
				{
					Center[(num * sizeCenter + num2)] = j;
					num2++;
				}
			}
			j++;
			if (j % frameWidth == frameWidth - windowWidth / 2)
			{
				j += windowWidth - 1;
			}
		}
	}
}
