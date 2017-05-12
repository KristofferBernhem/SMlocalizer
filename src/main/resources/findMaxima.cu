
// KernelDevelopment.findMaxima
extern "C" __global__  void run( int* data, int dataLen0, int frameWidth, int frameHeight, int windowWidth,  int* minLevel, int minLevelLen0, int minPosPixel, int sizeCenter,  int* Center, int CenterLen0);

// KernelDevelopment.findMaxima
extern "C" __global__  void run( int* data, int dataLen0, int frameWidth, int frameHeight, int windowWidth,  int* minLevel, int minLevelLen0, int minPosPixel, int sizeCenter,  int* Center, int CenterLen0)
{
	int num = blockIdx.x + gridDim.x * blockIdx.y;
	if (num < dataLen0 / (frameWidth * frameHeight))
	{
		int num2 = 0;
		int i = 0;
		int num3 = 0;
		bool flag = true;
		int j = num * (frameWidth * frameHeight);
		i = num * sizeCenter;
		if (minLevel[(num)] == 0)
		{
			double num4 = 0.0;
			double num5 = 0.0;
			int num6 = 0;
			while (j < (num + 1) * (frameWidth * frameHeight))
			{
				if (data[(j)] > 0)
				{
					num4 += (double)data[(j)];
					num6++;
				}
				j++;
			}
			if (num6 > 0)
			{
				num4 /= (double)num6;
				for (j = num * (frameWidth * frameHeight); j < (num + 1) * (frameWidth * frameHeight); j++)
				{
					num5 += ((double)data[(j)] - num4) * ((double)data[(j)] - num4);
				}
				num5 /= (double)num6;
				num5 = sqrt(num5);
				minLevel[(num)] = (int)(num4 * 2.0 + 3.0 * num5);
			}
			else
			{
				minLevel[(num)] = 64000;
			}
		}
		int num7 = (int)(0.3 * (double)minLevel[(num)]);
		j = num * (frameWidth * frameHeight) + windowWidth / 2 * frameWidth + windowWidth / 2;
		while (i < (num + 1) * sizeCenter)
		{
			Center[(i)] = 0;
			i++;
		}
		i = 0;
		int num8 = 0;
		while (j < (num + 1) * (frameWidth * frameHeight) - windowWidth / 2 * frameWidth)
		{
			if (data[(j)] > minLevel[(num)])
			{
				i = 0;
				num3 = j - windowWidth / 2 * (frameWidth + 1);
				flag = true;
				num8 = 0;
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
					num8++;
					if (num8 == windowWidth)
					{
						num3 += frameWidth - windowWidth;
						num8 = 0;
					}
				}
				if (i < minPosPixel)
				{
					flag = false;
				}
				if (flag && data[(j + 1)] > num7 && data[(j - 1)] > num7 && data[(j + frameWidth)] > num7 && data[(j - frameWidth)] > num7)
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
