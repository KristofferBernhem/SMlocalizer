
// KernelDevelopment.medianFilteringInterpolateSecond
extern "C" __global__  void medianKernel(int windowWidth,  float* filterWindow, int filterWindowLen0, int depth,  float* inputVector, int inputVectorLen0,  float* meanVector, int meanVectorLen0, int nStep,  int* answer, int answerLen0);

// KernelDevelopment.medianFilteringInterpolateSecond
extern "C" __global__  void medianKernel(int windowWidth,  float* filterWindow, int filterWindowLen0, int depth,  float* inputVector, int inputVectorLen0,  float* meanVector, int meanVectorLen0, int nStep,  int* answer, int answerLen0)
{
	int y = blockIdx.y;
	int x = blockIdx.x;
	int num = x + y * gridDim.x;
	if (num < inputVectorLen0 / depth)
	{
		int num2 = num * (2 * windowWidth + 1);
		int num3 = num2 + windowWidth;
		int num4 = 0;
		int i = num * depth;
		int j = num2;
		int num5 = num;
		int num6 = inputVectorLen0 / depth;
		float num7 = 0.0f;
		while (i < (num + 1) * depth)
		{
			inputVector[(i)] /= meanVector[(num4)];
			i++;
			num4++;
		}
		num4 = 0;
		i = num * depth;
		while (j <= num3)
		{
			filterWindow[(j)] = inputVector[(i)];
			j++;
			i += nStep;
		}
		num3++;
		bool flag = true;
		int k = 0;
		while (flag)
		{
			flag = false;
			k++;
			for (int l = num2; l < num3 - k; l++)
			{
				if (filterWindow[(l)] > filterWindow[(l + 1)])
				{
					float num8 = filterWindow[(l)];
					filterWindow[(l)] = filterWindow[(l + 1)];
					filterWindow[(l + 1)] = num8;
					flag = true;
				}
			}
		}
		num3 = windowWidth + 1;
		i = num * depth;
		answer[(num5)] = (int)(meanVector[(num4)] * (inputVector[(i)] - filterWindow[(num2 + num3 / 2 - 1)]));
		num7 = filterWindow[(num2 + num3 / 2 - 1)];
		if (answer[(num5)] < 0)
		{
			answer[(num5)] = 0;
		}
		num5 += num6 * nStep;
		i += nStep;
		num4 += nStep;
		float num9;
		while (j < num2 + 2 * windowWidth + 1)
		{
			filterWindow[(j)] = inputVector[(i + windowWidth * nStep)];
			flag = false;
			k = j;
			while (!flag)
			{
				if (filterWindow[(k - 1)] > filterWindow[(k)])
				{
					float num8 = filterWindow[(k)];
					filterWindow[(k)] = filterWindow[(k - 1)];
					filterWindow[(k - 1)] = num8;
				}
				else
				{
					flag = true;
				}
				k--;
				if (k == num2)
				{
					flag = true;
				}
			}
			answer[(num5)] = (int)(meanVector[(num4)] * (inputVector[(i)] - (filterWindow[(num2 + num3 / 2 - 1)] + (float)(num3 % 2) * filterWindow[(num2 + num3 % 2 + num3 / 2 - 1)]) / (float)(1 + num3 % 2)));
			if (answer[(num5)] < 0)
			{
				answer[(num5)] = 0;
			}
			num9 = (filterWindow[(num2 + num3 / 2 - 1)] + (float)(num3 % 2) * filterWindow[(num2 + num3 % 2 + num3 / 2 - 1)]) / (float)(1 + num3 % 2) - num7;
			num9 /= (float)nStep;
			for (int m = 1; m < nStep; m++)
			{
				answer[(num5 - nStep * num6 + m * num6)] = (int)(meanVector[(num4 - nStep + m)] * (inputVector[(i - nStep + m)] - num7 + num9 * (float)m));
				if (answer[(num5 - nStep * num6 + m * num6)] < 0)
				{
					answer[(num5 - nStep * num6 + m * num6)] = 0;
				}
			}
			num7 = (filterWindow[(num2 + num3 / 2)] + (float)(num3 % 2) * filterWindow[(num2 + num3 % 2 + num3 / 2)]) / (float)(1 + num3 % 2);
			num3++;
			i += nStep;
			j++;
			num5 += num6 * nStep;
			num4 += nStep;
		}
		while (i < (num + 1) * depth - windowWidth * nStep)
		{
			k = num2;
			flag = false;
			while (!flag)
			{
				if (inputVector[(i - (windowWidth + 1) * nStep)] == filterWindow[(k)])
				{
					filterWindow[(k)] = inputVector[(i + windowWidth * nStep)];
					flag = true;
				}
				else
				{
					k++;
				}
			}
			flag = false;
			if (k != num2 && k != num2 + 2 * windowWidth && (filterWindow[(k)] >= filterWindow[(k + 1)] || filterWindow[(k)] <= filterWindow[(k - 1)]))
			{
				if (filterWindow[(k)] > filterWindow[(k + 1)])
				{
					while (!flag)
					{
						float num8 = filterWindow[(k + 1)];
						filterWindow[(k + 1)] = filterWindow[(k)];
						filterWindow[(k)] = num8;
						k++;
						if (filterWindow[(k)] < filterWindow[(k + 1)])
						{
							flag = true;
						}
						if (k == num2 + 2 * windowWidth)
						{
							flag = true;
						}
					}
				}
				else
				{
					if (filterWindow[(k)] < filterWindow[(k - 1)])
					{
						while (!flag)
						{
							float num8 = filterWindow[(k - 1)];
							filterWindow[(k - 1)] = filterWindow[(k)];
							filterWindow[(k)] = num8;
							k--;
							if (filterWindow[(k)] > filterWindow[(k - 1)])
							{
								flag = true;
							}
							else
							{
								if (k == num2)
								{
									flag = true;
								}
							}
						}
					}
				}
			}
			answer[(num5)] = (int)(meanVector[(num4)] * (inputVector[(i)] - filterWindow[(num2 + num3 / 2)]));
			if (answer[(num5)] < 0)
			{
				answer[(num5)] = 0;
			}
			num9 = filterWindow[(num2 + num3 / 2)] - num7;
			num9 /= (float)nStep;
			for (int n = 1; n < nStep; n++)
			{
				answer[(num5 - nStep * num6 + n * num6)] = (int)(meanVector[(num4 - nStep + n)] * (inputVector[(i - nStep + n)] - num7 + num9 * (float)n));
				if (answer[(num5 - nStep * num6 + n * num6)] < 0)
				{
					answer[(num5 - nStep * num6 + n * num6)] = 0;
				}
			}
			num7 = filterWindow[(num2 + num3 / 2)];
			i += nStep;
			num5 += num6 * nStep;
			num4 += nStep;
		}
		num3--;
		while (i < (num + 1) * depth)
		{
			k = num2;
			flag = false;
			while (!flag)
			{
				if (inputVector[(i - (windowWidth - 1) * nStep)] == filterWindow[(k)])
				{
					while (k < num2 + num3)
					{
						float num8 = filterWindow[(k + 1)];
						filterWindow[(k + 1)] = filterWindow[(k)];
						filterWindow[(k)] = num8;
						k++;
					}
					flag = true;
				}
				else
				{
					k++;
				}
			}
			answer[(num5)] = (int)(meanVector[(num4)] * (inputVector[(i)] - (filterWindow[(num2 + num3 / 2)] + (float)(num3 % 2) * filterWindow[(num2 + num3 % 2 + num3 / 2)]) / (float)(1 + num3 % 2)));
			if (answer[(num5)] < 0)
			{
				answer[(num5)] = 0;
			}
			num9 = (filterWindow[(num2 + num3 / 2)] + (float)(num3 % 2) * filterWindow[(num2 + 1 + num3 / 2)]) / (float)(1 + num3 % 2) - num7;
			num9 /= (float)nStep;
			for (int num10 = 1; num10 < nStep; num10++)
			{
				answer[(num5 - nStep * num6 + num10 * num6)] = (int)(meanVector[(num4 - nStep + num10)] * (inputVector[(i - nStep + num10)] - num7 + num9 * (float)num10));
				if (answer[(num5 - nStep * num6 + num10 * num6)] < 0)
				{
					answer[(num5 - nStep * num6 + num10 * num6)] = 0;
				}
			}
			num7 = (filterWindow[(num2 + num3 / 2)] + (float)(num3 % 2) * filterWindow[(num2 + 1 + num3 / 2)]) / (float)(1 + num3 % 2);
			num3--;
			i += nStep;
			num5 += num6 * nStep;
			num4 += nStep;
		}
		i -= nStep;
		num4 -= nStep;
		num5 -= num6 * nStep;
		nStep = depth - num4;
		answer[(num5 + nStep * num6)] = (int)(meanVector[(num * depth - 1)] * (inputVector[(num * depth - 1)] - (filterWindow[(num2 + num3 / 2)] + (float)(num3 % 2) * filterWindow[(num2 + num3 % 2 + num3 / 2)]) / (float)(1 + num3 % 2)));
		num9 = (filterWindow[(num2 + num3 / 2)] + (float)(num3 % 2) * filterWindow[(num2 + num3 % 2 + num3 / 2)]) / (float)(1 + num3 % 2) - num7;
		num9 /= (float)nStep;
		for (int num11 = 1; num11 < nStep; num11++)
		{
			answer[(num5 + num11 * num6)] = (int)(meanVector[(num4 + num11)] * (inputVector[(i + num11)] - num7 - num9 * (float)num11));
			if (answer[(num5 + num11 * num6)] < 0)
			{
				answer[(num5 + num11 * num6)] = 0;
			}
		}
	}
}
