
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
					i = 0;
					flag = true;
					while (i < num2)
					{
						if ((double)((Center[(num * sizeCenter + i)] % frameWidth - j % frameWidth) * (Center[(num * sizeCenter + i)] % frameWidth - j % frameWidth) + (Center[(num * sizeCenter + i)] / frameWidth % frameHeight - j / frameWidth % frameHeight) * (Center[(num * sizeCenter + i)] / frameWidth % frameHeight - j / frameWidth % frameHeight)) < sqDistance)
						{
							Center[(num * sizeCenter + i)] = 0;
							flag = false;
						}
						i++;
					}
					if (flag)
					{
						Center[(num * sizeCenter + num2)] = j;
						num2++;
					}
					else
					{
						i = num * sizeCenter;
						num4 = num2;
						while (i < num * sizeCenter + num4)
						{
							flag = true;
							if (Center[(i)] == 0)
							{
								num3 = i + 1;
								while (num3 < num * sizeCenter + num2 && flag)
								{
									if (Center[(num3)] > 0)
									{
										Center[(i)] = Center[(num3)];
										Center[(num3)] = 0;
										flag = false;
										num2--;
									}
									num3++;
								}
							}
							i++;
						}
					}
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
