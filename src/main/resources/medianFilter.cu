
// KernelDevelopment.medianFilteringInterpolateSecond
extern "C" __global__  void medianKernel(int n, int windowWidth,  float* filterWindow, int filterWindowLen0, int depth,  float* inputVector, int inputVectorLen0,  float* meanVector, int meanVectorLen0, int nStep,  int* answer, int answerLen0);

// KernelDevelopment.medianFilteringInterpolateSecond
extern "C" __global__  void medianKernel(int n, int windowWidth,  float* filterWindow, int filterWindowLen0, int depth,  float* inputVector, int inputVectorLen0,  float* meanVector, int meanVectorLen0, int nStep,  int* answer, int answerLen0)
{
	int num = blockIdx.x * blockDim.x + threadIdx.x;
	int num2 = blockDim.x * gridDim.x;
	for (int i = num; i < n; i += num2)
	{
		int num3 = i * (2 * windowWidth + 1);
		int num4 = num3 + windowWidth;
		int num5 = 0;
		int j = i * depth;
		int k = num3;
		int num6 = i;
		int num7 = inputVectorLen0 / depth;
		float num8 = 0.0f;
		while (j < (i + 1) * depth)
		{
			inputVector[(j)] /= meanVector[(num5)];
			j++;
			num5++;
		}
		num5 = 0;
		j = i * depth;
		while (k <= num4)
		{
			filterWindow[(k)] = inputVector[(j)];
			k++;
			j += nStep;
		}
		num4++;
		bool flag = true;
		int l = 0;
		while (flag)
		{
			flag = false;
			l++;
			for (int m = num3; m < num4 - l; m++)
			{
				if (filterWindow[(m)] > filterWindow[(m + 1)])
				{
					float num9 = filterWindow[(m)];
					filterWindow[(m)] = filterWindow[(m + 1)];
					filterWindow[(m + 1)] = num9;
					flag = true;
				}
			}
		}
		num4 = windowWidth + 1;
		j = i * depth;
		answer[(num6)] = (int)(meanVector[(num5)] * (inputVector[(j)] - filterWindow[(num3 + num4 / 2 - 1)]));
		num8 = filterWindow[(num3 + num4 / 2 - 1)];
		if (answer[(num6)] < 0)
		{
			answer[(num6)] = 0;
		}
		num6 += num7 * nStep;
		j += nStep;
		num5 += nStep;
		float num10;
		while (k < num3 + 2 * windowWidth + 1)
		{
			filterWindow[(k)] = inputVector[(j + windowWidth * nStep)];
			flag = false;
			l = k;
			while (!flag)
			{
				if (filterWindow[(l - 1)] > filterWindow[(l)])
				{
					float num9 = filterWindow[(l)];
					filterWindow[(l)] = filterWindow[(l - 1)];
					filterWindow[(l - 1)] = num9;
				}
				else
				{
					flag = true;
				}
				l--;
				if (l == num3)
				{
					flag = true;
				}
			}
			answer[(num6)] = (int)(meanVector[(num5)] * (inputVector[(j)] - (filterWindow[(num3 + num4 / 2 - 1)] + (float)(num4 % 2) * filterWindow[(num3 + num4 % 2 + num4 / 2 - 1)]) / (float)(1 + num4 % 2)));
			if (answer[(num6)] < 0)
			{
				answer[(num6)] = 0;
			}
			num10 = (filterWindow[(num3 + num4 / 2 - 1)] + (float)(num4 % 2) * filterWindow[(num3 + num4 % 2 + num4 / 2 - 1)]) / (float)(1 + num4 % 2) - num8;
			num10 /= (float)nStep;
			for (int num11 = 1; num11 < nStep; num11++)
			{
				answer[(num6 - nStep * num7 + num11 * num7)] = (int)(meanVector[(num5 - nStep + num11)] * (inputVector[(j - nStep + num11)] - num8 + num10 * (float)num11));
				if (answer[(num6 - nStep * num7 + num11 * num7)] < 0)
				{
					answer[(num6 - nStep * num7 + num11 * num7)] = 0;
				}
			}
			num8 = (filterWindow[(num3 + num4 / 2)] + (float)(num4 % 2) * filterWindow[(num3 + num4 % 2 + num4 / 2)]) / (float)(1 + num4 % 2);
			num4++;
			j += nStep;
			k++;
			num6 += num7 * nStep;
			num5 += nStep;
		}
		while (j < (i + 1) * depth - windowWidth * nStep)
		{
			l = num3;
			float num12 = 100.0f;
			int num13 = 0;
			while (l < num3 + 2 * windowWidth)
			{
				if (abs(inputVector[(j - windowWidth * nStep)] - filterWindow[(l)]) < num12)
				{
					num12 = abs(inputVector[(j - windowWidth * nStep)] - filterWindow[(l)]);
					num13 = l;
				}
				l++;
			}
			l = num13;
			filterWindow[(l)] = inputVector[(j + windowWidth * nStep)];
			flag = false;
			if (l == num3)
			{
				while (l < num3 + 2 * windowWidth)
				{
					if (filterWindow[(l)] <= filterWindow[(l + 1)])
					{
						break;
					}
					float num9 = filterWindow[(l + 1)];
					filterWindow[(l + 1)] = filterWindow[(l)];
					filterWindow[(l)] = num9;
					l++;
				}
			}
			else
			{
				if (l == num3 + 2 * windowWidth)
				{
					while (l > num3 + 1)
					{
						if (filterWindow[(l)] >= filterWindow[(l - 1)])
						{
							break;
						}
						float num9 = filterWindow[(l - 1)];
						filterWindow[(l - 1)] = filterWindow[(l)];
						filterWindow[(l)] = num9;
						l--;
					}
				}
				else
				{
					if (filterWindow[(l)] >= filterWindow[(l + 1)] || filterWindow[(l)] <= filterWindow[(l - 1)])
					{
						if (filterWindow[(l)] > filterWindow[(l + 1)])
						{
							while (!flag)
							{
								float num9 = filterWindow[(l + 1)];
								filterWindow[(l + 1)] = filterWindow[(l)];
								filterWindow[(l)] = num9;
								l++;
								if (l == num3 + 2 * windowWidth)
								{
									flag = true;
								}
								else
								{
									if (filterWindow[(l)] < filterWindow[(l + 1)])
									{
										flag = true;
									}
								}
							}
						}
						else
						{
							if (filterWindow[(l)] < filterWindow[(l - 1)])
							{
								while (l > num3 + 1 && filterWindow[(l)] < filterWindow[(l - 1)])
								{
									float num9 = filterWindow[(l - 1)];
									filterWindow[(l - 1)] = filterWindow[(l)];
									filterWindow[(l)] = num9;
									l--;
								}
							}
						}
					}
				}
			}
			answer[(num6)] = (int)(meanVector[(num5)] * (inputVector[(j)] - filterWindow[(num3 + num4 / 2)]));
			if (answer[(num6)] < 0)
			{
				answer[(num6)] = 0;
			}
			num10 = filterWindow[(num3 + num4 / 2)] - num8;
			num10 /= (float)nStep;
			for (int num14 = 1; num14 < nStep; num14++)
			{
				answer[(num6 - nStep * num7 + num14 * num7)] = (int)(meanVector[(num5 - nStep + num14)] * (inputVector[(j - nStep + num14)] - num8 + num10 * (float)num14));
				if (answer[(num6 - nStep * num7 + num14 * num7)] < 0)
				{
					answer[(num6 - nStep * num7 + num14 * num7)] = 0;
				}
			}
			num8 = filterWindow[(num3 + num4 / 2)];
			j += nStep;
			num6 += num7 * nStep;
			num5 += nStep;
		}
		num4--;
		while (j < (i + 1) * depth)
		{
			l = num3;
			flag = false;
			float num15 = 100.0f;
			int num16 = 0;
			while (l < num3 + num4 - 1)
			{
				if (abs(inputVector[(j - windowWidth * nStep)] - filterWindow[(l)]) < num15)
				{
					num15 = abs(inputVector[(j - windowWidth * nStep)] - filterWindow[(l)]);
					num16 = l;
				}
				l++;
			}
			for (l = num16; l < num3 + 2 * windowWidth; l++)
			{
				float num9 = filterWindow[(l + 1)];
				filterWindow[(l + 1)] = filterWindow[(l)];
				filterWindow[(l)] = num9;
			}
			answer[(num6)] = (int)(meanVector[(num5)] * (inputVector[(j)] - (filterWindow[(num3 + num4 / 2)] + (float)(num4 % 2) * filterWindow[(num3 + num4 % 2 + num4 / 2)]) / (float)(1 + num4 % 2)));
			if (answer[(num6)] < 0)
			{
				answer[(num6)] = 0;
			}
			num10 = (filterWindow[(num3 + num4 / 2)] + (float)(num4 % 2) * filterWindow[(num3 + 1 + num4 / 2)]) / (float)(1 + num4 % 2) - num8;
			num10 /= (float)nStep;
			for (int num17 = 1; num17 < nStep; num17++)
			{
				answer[(num6 - nStep * num7 + num17 * num7)] = (int)(meanVector[(num5 - nStep + num17)] * (inputVector[(j - nStep + num17)] - num8 + num10 * (float)num17));
				if (answer[(num6 - nStep * num7 + num17 * num7)] < 0)
				{
					answer[(num6 - nStep * num7 + num17 * num7)] = 0;
				}
			}
			num8 = (filterWindow[(num3 + num4 / 2)] + (float)(num4 % 2) * filterWindow[(num3 + 1 + num4 / 2)]) / (float)(1 + num4 % 2);
			num4--;
			j += nStep;
			num6 += num7 * nStep;
			num5 += nStep;
		}
		j -= nStep;
		num5 -= nStep;
		num6 -= num7 * nStep;
		nStep = depth - num5 - 1;
		answer[(num6 + nStep * num7)] = (int)(meanVector[(depth - 1)] * (inputVector[((i + 1) * depth - 1)] - (filterWindow[(num3 + num4 / 2)] + (float)(num4 % 2) * filterWindow[(num3 + 1 + num4 / 2)]) / (float)(1 + num4 % 2)));
		num10 = (filterWindow[(num3 + num4 / 2)] + (float)(num4 % 2) * filterWindow[(num3 + 1 + num4 / 2)]) / (float)(1 + num4 % 2) - num8;
		num10 = 1.0f;
		num10 /= (float)nStep;
		int num18 = 1;
		while (num18 < nStep && num6 + num18 * num7 < answerLen0)
		{
			answer[(num6 + num18 * num7)] = (int)(meanVector[(num5 + num18)] * (inputVector[(j + num18)] - num8 - num10 * (float)num18));
			if (answer[(num6 + num18 * num7)] < 0)
			{
				answer[(num6 + num18 * num7)] = 0;
			}
			num18++;
		}
	}
}
