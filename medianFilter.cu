
// KernelDevelopment.medianFilteringInterpolate
extern "C" __global__  void medianKernel(int windowWidth,  float* filterWindow, int filterWindowLen0, int depth,  float* inputVector, int inputVectorLen0,  float* meanVector, int meanVectorLen0, int nStep,  int* answer, int answerLen0);

// KernelDevelopment.medianFilteringInterpolate
extern "C" __global__  void medianKernel(int windowWidth,  float* filterWindow, int filterWindowLen0, int depth,  float* inputVector, int inputVectorLen0,  float* meanVector, int meanVectorLen0, int nStep,  int* answer, int answerLen0)
{
	int y = blockIdx.y;
	int x = blockIdx.x;
	int num = x + y * gridDim.x;
	if (num < inputVectorLen0 / depth)
	{
		int num2 = 0;
		int i = num * depth;
		int num3 = num * (2 * windowWidth + 1);
		bool flag = true;
		double num4 = 0.0;
		int j = num;
		while (i < (num + 1) * depth)
		{
			inputVector[(i)] /= meanVector[(num2)];
			num2++;
			i++;
		}
		num2 = 0;
		i = num * depth;
		for (int k = num * depth; k < num * depth + (windowWidth + 1) * nStep; k += nStep)
		{
			filterWindow[(num3)] = inputVector[(k)];
			num3++;
		}
		int num5 = num * (2 * windowWidth + 1);
		for (int l = num5 + 1; l < num5 + windowWidth + 1; l++)
		{
			for (int m = num5; m < num5 + windowWidth - l; m++)
			{
				if (filterWindow[(m)] > filterWindow[(m + 1)])
				{
					float num6 = filterWindow[(m)];
					filterWindow[(m)] = filterWindow[(m + 1)];
					filterWindow[(m + 1)] = num6;
				}
			}
		}
		if (windowWidth % 2 == 0)
		{
			answer[(j)] = (int)(meanVector[(num2)] * (inputVector[(i)] - filterWindow[(num5 + windowWidth / 2)]));
			if (answer[(j)] < 0)
			{
				answer[(j)] = 0;
			}
			num4 = (double)filterWindow[(num5 + windowWidth / 2)];
			flag = false;
		}
		else
		{
			answer[(j)] = (int)((double)meanVector[(num2)] * ((double)inputVector[(i)] - (double)(filterWindow[(num5 + (windowWidth - 1) / 2)] + filterWindow[(num5 + (windowWidth - 1) / 2 + 1)]) / 2.0));
			if (answer[(j)] < 0)
			{
				answer[(j)] = 0;
			}
			num4 = (double)(filterWindow[(num5 + (windowWidth - 1) / 2)] + filterWindow[(num5 + (windowWidth - 1) / 2 + 1)]) / 2.0;
			flag = true;
		}
		num2 += nStep;
		if (flag)
		{
			num5 = num * (2 * windowWidth + 1) + (windowWidth - 1) / 2 + 1;
		}
		else
		{
			num5 = num * (2 * windowWidth + 1) + windowWidth / 2 + 1;
		}
		i += nStep;
		j += inputVectorLen0 / depth * nStep;
		double num9;
		for (int n = num * depth + windowWidth + 1; n < num * depth + (2 * windowWidth + 1) * nStep; n += nStep)
		{
			filterWindow[(num3)] = inputVector[(n)];
			for (int num7 = num * (2 * windowWidth + 1); num7 < num3; num7++)
			{
				for (int num8 = num * (2 * windowWidth + 1); num8 < num3 - num7; num8++)
				{
					if (filterWindow[(num8)] > filterWindow[(num8 + 1)])
					{
						float num6 = filterWindow[(num8)];
						filterWindow[(num8)] = filterWindow[(num8 + 1)];
						filterWindow[(num8 + 1)] = num6;
					}
				}
			}
			if (flag)
			{
				answer[(j)] = (int)(meanVector[(num2)] * (inputVector[(i)] - filterWindow[(num5)]));
				if (answer[(j)] < 0)
				{
					answer[(j)] = 0;
				}
				num9 = ((double)filterWindow[(num5)] - num4) / (double)nStep;
				for (int num10 = 1; num10 < nStep; num10++)
				{
					answer[(j - inputVectorLen0 / depth * nStep + inputVectorLen0 / depth * num10)] = (int)((double)meanVector[(num2 - nStep + num10)] * ((double)inputVector[(i - nStep + num10)] - (num4 + num9 * (double)num10)));
					if (answer[(j - inputVectorLen0 / depth * nStep + inputVectorLen0 / depth * num10)] < 0)
					{
						answer[(j - inputVectorLen0 / depth * nStep + inputVectorLen0 / depth * num10)] = 0;
					}
				}
				flag = false;
				num4 = (double)filterWindow[(num5)];
				num5++;
			}
			else
			{
				answer[(j)] = (int)((double)meanVector[(num2)] * ((double)inputVector[(i)] - (double)(filterWindow[(num5)] + filterWindow[(num5 - 1)]) / 2.0));
				if (answer[(j)] < 0)
				{
					answer[(j)] = 0;
				}
				num9 = ((double)(filterWindow[(num5)] + filterWindow[(num5 - 1)]) / 2.0 - num4) / (double)nStep;
				for (int num11 = 1; num11 < nStep; num11++)
				{
					answer[(j - inputVectorLen0 / depth * nStep + inputVectorLen0 / depth * num11)] = (int)((double)meanVector[(num2 - nStep + num11)] * ((double)inputVector[(i - nStep + num11)] - (num4 + num9 * (double)num11)));
					if (answer[(j - inputVectorLen0 / depth * nStep + inputVectorLen0 / depth * num11)] < 0)
					{
						answer[(j - inputVectorLen0 / depth * nStep + inputVectorLen0 / depth * num11)] = 0;
					}
				}
				flag = true;
				num4 = (double)(filterWindow[(num5)] + filterWindow[(num5 + 1)]) / 2.0;
			}
			num2 += nStep;
			num3++;
			i += nStep;
			j += inputVectorLen0 / depth * nStep;
		}
		num3 = num * (2 * windowWidth + 1);
		num5 = num3 + windowWidth;
		int num12 = num5 + windowWidth;
		bool flag2 = false;
		int num13 = 0;
		while (i < (num + 1) * depth - windowWidth)
		{
			flag2 = false;
			num13 = num3;
			while (!flag2 && num13 < num12 + 1)
			{
				if (filterWindow[(num13)] == inputVector[(i - windowWidth - 1)])
				{
					flag2 = true;
					filterWindow[(num13)] = inputVector[(i + windowWidth)];
					if ((num13 != num3 || filterWindow[(num13)] >= filterWindow[(num13 + 1)]) && (num13 != num12 || filterWindow[(num13)] <= filterWindow[(num13 - 1)]))
					{
						if (num13 == num3 && filterWindow[(num13)] > filterWindow[(num13 + 1)])
						{
							while (filterWindow[(num13)] > filterWindow[(num13 + 1)])
							{
								if (num13 >= num12)
								{
									break;
								}
								float num6 = filterWindow[(num13 + 1)];
								filterWindow[(num13 + 1)] = filterWindow[(num13)];
								filterWindow[(num13)] = num6;
								num13++;
							}
						}
						else
						{
							if (num13 == num12 && filterWindow[(num13)] < filterWindow[(num13 - 1)])
							{
								while (filterWindow[(num13)] < filterWindow[(num13 - 1)])
								{
									if (num13 <= num3)
									{
										break;
									}
									float num6 = filterWindow[(num13 - 1)];
									filterWindow[(num13 - 1)] = filterWindow[(num13)];
									filterWindow[(num13)] = num6;
									num13--;
								}
							}
							else
							{
								if (filterWindow[(num13)] != filterWindow[(num13 + 1)] && filterWindow[(num13)] != filterWindow[(num13 - 1)])
								{
									if (filterWindow[(num13)] > filterWindow[(num13 + 1)])
									{
										while (filterWindow[(num13)] > filterWindow[(num13 + 1)])
										{
											if (num13 >= num12)
											{
												break;
											}
											float num6 = filterWindow[(num13 + 1)];
											filterWindow[(num13 + 1)] = filterWindow[(num13)];
											filterWindow[(num13)] = num6;
											num13++;
										}
									}
									else
									{
										while (filterWindow[(num13)] < filterWindow[(num13 - 1)] && num13 > num3)
										{
											float num6 = filterWindow[(num13 - 1)];
											filterWindow[(num13 - 1)] = filterWindow[(num13)];
											filterWindow[(num13)] = num6;
											num13--;
										}
									}
								}
							}
						}
					}
				}
				num13++;
			}
			answer[(j)] = (int)(meanVector[(num2)] * (inputVector[(i)] - filterWindow[(num5)]));
			if (answer[(j)] < 0)
			{
				answer[(j)] = 0;
			}
			num9 = ((double)filterWindow[(num5)] - num4) / (double)nStep;
			for (int num14 = 1; num14 < nStep; num14++)
			{
				answer[(j - inputVectorLen0 / depth * nStep + inputVectorLen0 / depth * num14)] = (int)((double)meanVector[(num2 - nStep + num14)] * ((double)inputVector[(i - nStep + num14)] - (num4 + num9 * (double)num14)));
				if (answer[(j - inputVectorLen0 / depth * nStep + inputVectorLen0 / depth * num14)] < 0)
				{
					answer[(j - inputVectorLen0 / depth * nStep + inputVectorLen0 / depth * num14)] = 0;
				}
			}
			num4 = (double)filterWindow[(num5)];
			num2 += nStep;
			i += nStep;
			j += inputVectorLen0 / depth * nStep;
		}
		flag = false;
		while (i < (num + 1) * depth)
		{
			flag2 = false;
			num13 = num3;
			while (!flag2 && num13 < num12 + 1)
			{
				if (filterWindow[(num13)] == inputVector[(i - windowWidth - 1)] && num13 < num12)
				{
					while (num13 < num12 + 1)
					{
						float num6 = filterWindow[(num13)];
						filterWindow[(num13)] = filterWindow[(num13 + 1)];
						filterWindow[(num13 + 1)] = num6;
						num13++;
					}
				}
				num13++;
			}
			num12--;
			if (flag)
			{
				answer[(j)] = (int)(meanVector[(num2)] * (inputVector[(i)] - filterWindow[(num5)]));
				if (answer[(j)] < 0)
				{
					answer[(j)] = 0;
				}
				num9 = ((double)filterWindow[(num5)] - num4) / (double)nStep;
				for (int num15 = 1; num15 < nStep; num15++)
				{
					answer[(j - inputVectorLen0 / depth * nStep + inputVectorLen0 / depth * num15)] = (int)((double)meanVector[(num2 - nStep + num15)] * ((double)inputVector[(i - nStep + num15)] - (num4 + num9 * (double)num15)));
					if (answer[(j - inputVectorLen0 / depth * nStep + inputVectorLen0 / depth * num15)] < 0)
					{
						answer[(j - inputVectorLen0 / depth * nStep + inputVectorLen0 / depth * num15)] = 0;
					}
				}
				flag = false;
				num4 = (double)filterWindow[(num5)];
				num5++;
			}
			else
			{
				answer[(j)] = (int)((double)meanVector[(num2)] * ((double)inputVector[(i)] - (double)(filterWindow[(num5)] + filterWindow[(num5 - 1)]) / 2.0));
				if (answer[(j)] < 0)
				{
					answer[(j)] = 0;
				}
				num9 = ((double)(filterWindow[(num5)] + filterWindow[(num5 - 1)]) / 2.0 - num4) / (double)nStep;
				for (int num16 = 1; num16 < nStep; num16++)
				{
					answer[(j - inputVectorLen0 / depth * nStep + inputVectorLen0 / depth * num16)] = (int)((double)meanVector[(num2 - nStep + num16)] * ((double)inputVector[(i - nStep + num16)] - (num4 + num9 * (double)num16)));
					if (answer[(j - inputVectorLen0 / depth * nStep + inputVectorLen0 / depth * num16)] < 0)
					{
						answer[(j - inputVectorLen0 / depth * nStep + inputVectorLen0 / depth * num16)] = 0;
					}
				}
				flag = true;
				num4 = (double)(filterWindow[(num5)] + filterWindow[(num5 + 1)]) / 2.0;
				num5--;
			}
			num2 += nStep;
			i += nStep;
			j += inputVectorLen0 / depth * nStep;
		}
		num2 -= nStep;
		i -= nStep;
		j -= inputVectorLen0 / depth * nStep;
		nStep = meanVectorLen0 - num2;
		num9 = ((double)filterWindow[(num5)] - num4) / (double)nStep;
		while (j < answerLen0)
		{
			answer[(j)] = (int)(meanVector[(num2)] * (inputVector[(i)] - filterWindow[(num5)]));
			if (answer[(j)] < 0)
			{
				answer[(j)] = 0;
			}
			j += inputVectorLen0 / depth;
			num2++;
			i++;
		}
	}
}
