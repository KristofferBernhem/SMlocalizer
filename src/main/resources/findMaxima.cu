
// KernelDevelopment.findMaxima
extern "C" __global__  void run(int n,  int* data, int dataLen0, int frameWidth, int frameHeight, int windowWidth,  int* minLevel, int minLevelLen0, int minPosPixel, int sizeCenter,  int* Center, int CenterLen0);

// KernelDevelopment.findMaxima
extern "C" __global__  void run(int n,  int* data, int dataLen0, int frameWidth, int frameHeight, int windowWidth,  int* minLevel, int minLevelLen0, int minPosPixel, int sizeCenter,  int* Center, int CenterLen0)
{
	int num = blockIdx.x * blockDim.x + threadIdx.x;
	int num2 = blockDim.x * gridDim.x;
	for (int i = num; i < n; i += num2)
	{
		int num3 = 0;
		int j = 0;
		int num4 = 0;
		bool flag = true;
		int k = i * (frameWidth * frameHeight);
		j = i * sizeCenter;
		if (minLevel[(i)] == 0)
		{
			double num5 = 0.0;
			double num6 = 0.0;
			int num7 = 0;
			while (k < (i + 1) * (frameWidth * frameHeight))
			{
				if (data[(k)] > 0)
				{
					num5 += (double)data[(k)];
					num7++;
				}
				k++;
			}
			if (num7 > 0)
			{
				num5 /= (double)num7;
				for (k = i * (frameWidth * frameHeight); k < (i + 1) * (frameWidth * frameHeight); k++)
				{
					num6 += ((double)data[(k)] - num5) * ((double)data[(k)] - num5);
				}
				num6 /= (double)num7;
				num6 = sqrt(num6);
				minLevel[(i)] = (int)(num5 * 2.0 + 3.0 * num6);
			}
			else
			{
				minLevel[(i)] = 64000;
			}
		}
		int num8 = (int)(0.3 * (double)minLevel[(i)]);
		k = i * (frameWidth * frameHeight) + windowWidth / 2 * frameWidth + windowWidth / 2;
		while (j < (i + 1) * sizeCenter)
		{
			Center[(j)] = 0;
			j++;
		}
		j = 0;
		int num9 = 0;
		while (k < (i + 1) * (frameWidth * frameHeight) - windowWidth / 2 * frameWidth)
		{
			if (data[(k)] > minLevel[(i)])
			{
				j = 0;
				num4 = k - windowWidth / 2 * (frameWidth + 1);
				flag = true;
				num9 = 0;
				while (num4 <= k + windowWidth / 2 * (frameWidth + 1) && flag)
				{
					if (data[(num4)] > data[(k)])
					{
						flag = false;
					}
					if (data[(num4)] > 0)
					{
						j++;
					}
					num4++;
					num9++;
					if (num9 == windowWidth)
					{
						num4 += frameWidth - windowWidth;
						num9 = 0;
					}
				}
				if (j < minPosPixel)
				{
					flag = false;
				}
				if (flag && data[(k + 1)] > num8 && data[(k - 1)] > num8 && data[(k + frameWidth)] > num8 && data[(k - frameWidth)] > num8)
				{
					Center[(i * sizeCenter + num3)] = k;
					num3++;
				}
			}
			k++;
			if (k % frameWidth == frameWidth - windowWidth / 2)
			{
				k += windowWidth - 1;
			}
		}
	}
}
