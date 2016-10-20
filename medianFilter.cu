
// KernelDevelopment.medianFiltering
extern "C" __global__  void medianKernel(int windowWidth,  float* filterWindow, int filterWindowLen0, int depth,  float* inputVector, int inputVectorLen0,  float* meanVector, int meanVectorLen0,  int* answer, int answerLen0);
// KernelDevelopment.medianFiltering
extern "C" __global__  void medianKernelInterpolate(int windowWidth,  float* filterWindow, int filterWindowLen0, int depth,  float* inputVector, int inputVectorLen0,  float* meanVector, int meanVectorLen0, int nStep,  int* answer, int answerLen0);

// KernelDevelopment.medianFiltering
extern "C" __global__  void medianKernel(int windowWidth,  float* filterWindow, int filterWindowLen0, int depth,  float* inputVector, int inputVectorLen0,  float* meanVector, int meanVectorLen0,  int* answer, int answerLen0)
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
		int num4 = num;
		while (i < (num + 1) * depth)
		{
			inputVector[(i)] /= meanVector[(num2)];
			num2++;
			i++;
		}
		num2 = 0;
		i = num * depth;
		for (int j = num * depth; j < num * depth + windowWidth + 1; j++)
		{
			filterWindow[(num3)] = inputVector[(j)];
			num3++;
		}
		int num5 = num * (2 * windowWidth + 1);
		for (int k = num5 + 1; k < num5 + windowWidth + 1; k++)
		{
			for (int l = num5; l < num5 + windowWidth - k; l++)
			{
				if (filterWindow[(l)] > filterWindow[(l + 1)])
				{
					float num6 = filterWindow[(l)];
					filterWindow[(l)] = filterWindow[(l + 1)];
					filterWindow[(l + 1)] = num6;
				}
			}
		}
		if (windowWidth % 2 == 0)
		{
			answer[(num4)] = (int)(meanVector[(num2)] * (inputVector[(i)] - filterWindow[(num5 + windowWidth / 2)]));
			if (answer[(num4)] < 0)
			{
				answer[(num4)] = 0;
			}
			flag = false;
		}
		else
		{
			answer[(num4)] = (int)((double)meanVector[(num2)] * ((double)inputVector[(i)] - (double)(filterWindow[(num5 + (windowWidth - 1) / 2)] + filterWindow[(num5 + (windowWidth - 1) / 2 + 1)]) / 2.0));
			if (answer[(num4)] < 0)
			{
				answer[(num4)] = 0;
			}
			flag = true;
		}
		num2++;
		if (flag)
		{
			num5 = num * (2 * windowWidth + 1) + (windowWidth - 1) / 2 + 1;
		}
		else
		{
			num5 = num * (2 * windowWidth + 1) + windowWidth / 2 + 1;
		}
		i++;
		num4 += inputVectorLen0 / depth;
		for (int m = num * depth + windowWidth + 1; m < num * depth + 2 * windowWidth + 1; m++)
		{
			filterWindow[(num3)] = inputVector[(m)];
			for (int n = num * (2 * windowWidth + 1); n < num3; n++)
			{
				for (int num7 = num * (2 * windowWidth + 1); num7 < num3 - n; num7++)
				{
					if (filterWindow[(num7)] > filterWindow[(num7 + 1)])
					{
						float num6 = filterWindow[(num7)];
						filterWindow[(num7)] = filterWindow[(num7 + 1)];
						filterWindow[(num7 + 1)] = num6;
					}
				}
			}
			if (flag)
			{
				answer[(num4)] = (int)(meanVector[(num2)] * (inputVector[(i)] - filterWindow[(num5)]));
				if (answer[(num4)] < 0)
				{
					answer[(num4)] = 0;
				}
				flag = false;
				num5++;
			}
			else
			{
				answer[(num4)] = (int)((double)meanVector[(num2)] * ((double)inputVector[(i)] - (double)(filterWindow[(num5)] + filterWindow[(num5 - 1)]) / 2.0));
				if (answer[(num4)] < 0)
				{
					answer[(num4)] = 0;
				}
				flag = true;
			}
			num2++;
			num3++;
			i++;
			num4 += inputVectorLen0 / depth;
		}
		num3 = num * (2 * windowWidth + 1);
		num5 = num3 + windowWidth;
		int num8 = num5 + windowWidth;
		bool flag2 = false;
		int num9 = 0;
		while (i < (num + 1) * depth - windowWidth)
		{
			flag2 = false;
			num9 = num3;
			while (!flag2 && num9 < num8 + 1)
			{
				if (filterWindow[(num9)] == inputVector[(i - windowWidth - 1)])
				{
					flag2 = true;
					filterWindow[(num9)] = inputVector[(i + windowWidth)];
					if ((num9 != num3 || filterWindow[(num9)] >= filterWindow[(num9 + 1)]) && (num9 != num8 || filterWindow[(num9)] <= filterWindow[(num9 - 1)]))
					{
						if (num9 == num3 && filterWindow[(num9)] > filterWindow[(num9 + 1)])
						{
							while (filterWindow[(num9)] > filterWindow[(num9 + 1)])
							{
								if (num9 >= num8)
								{
									break;
								}
								float num6 = filterWindow[(num9 + 1)];
								filterWindow[(num9 + 1)] = filterWindow[(num9)];
								filterWindow[(num9)] = num6;
								num9++;
							}
						}
						else
						{
							if (num9 == num8 && filterWindow[(num9)] < filterWindow[(num9 - 1)])
							{
								while (filterWindow[(num9)] < filterWindow[(num9 - 1)])
								{
									if (num9 <= num3)
									{
										break;
									}
									float num6 = filterWindow[(num9 - 1)];
									filterWindow[(num9 - 1)] = filterWindow[(num9)];
									filterWindow[(num9)] = num6;
									num9--;
								}
							}
							else
							{
								if (filterWindow[(num9)] != filterWindow[(num9 + 1)] && filterWindow[(num9)] != filterWindow[(num9 - 1)])
								{
									if (filterWindow[(num9)] > filterWindow[(num9 + 1)])
									{
										while (filterWindow[(num9)] > filterWindow[(num9 + 1)])
										{
											if (num9 >= num8)
											{
												break;
											}
											float num6 = filterWindow[(num9 + 1)];
											filterWindow[(num9 + 1)] = filterWindow[(num9)];
											filterWindow[(num9)] = num6;
											num9++;
										}
									}
									else
									{
										while (filterWindow[(num9)] < filterWindow[(num9 - 1)] && num9 > num3)
										{
											float num6 = filterWindow[(num9 - 1)];
											filterWindow[(num9 - 1)] = filterWindow[(num9)];
											filterWindow[(num9)] = num6;
											num9--;
										}
									}
								}
							}
						}
					}
				}
				num9++;
			}
			answer[(num4)] = (int)(meanVector[(num2)] * (inputVector[(i)] - filterWindow[(num5)]));
			if (answer[(num4)] < 0)
			{
				answer[(num4)] = 0;
			}
			num2++;
			i++;
			num4 += inputVectorLen0 / depth;
		}
		flag = false;
		while (i < (num + 1) * depth)
		{
			if (flag)
			{
				answer[(num4)] = (int)(meanVector[(num2)] * (inputVector[(i)] - filterWindow[(num5)]));
				if (answer[(num4)] < 0)
				{
					answer[(num4)] = 0;
				}
				flag = false;
			}
			else
			{
				answer[(num4)] = (int)((double)meanVector[(num2)] * ((double)inputVector[(i)] - (double)(filterWindow[(num5)] + filterWindow[(num5 + 1)]) / 2.0));
				if (answer[(num4)] < 0)
				{
					answer[(num4)] = 0;
				}
				flag = true;
				num5++;
			}
			num2++;
			i++;
			num4 += inputVectorLen0 / depth;
		}
	}
}
// KernelDevelopment.medianFiltering
extern "C" __global__  void medianKernelInterpolate(int windowWidth,  float* filterWindow, int filterWindowLen0, int depth,  float* inputVector, int inputVectorLen0,  float* meanVector, int meanVectorLen0, int nStep,  int* answer, int answerLen0)
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
		int num4 = num;
		while (i < (num + 1) * depth)
		{
			inputVector[(i)] /= meanVector[(num2)];
			num2++;
			i++;
		}
		num2 = 0;
		i = num * depth;
		for (int j = num * depth; j < num * depth + windowWidth + 1; j += nStep)
		{
			filterWindow[(num3)] = inputVector[(j)];
			num3++;
		}
		int num5 = num * (2 * windowWidth + 1);
		for (int k = num5 + 1; k < num5 + windowWidth + 1; k++)
		{
			for (int l = num5; l < num5 + windowWidth - k; l++)
			{
				if (filterWindow[(l)] > filterWindow[(l + 1)])
				{
					float num6 = filterWindow[(l)];
					filterWindow[(l)] = filterWindow[(l + 1)];
					filterWindow[(l + 1)] = num6;
				}
			}
		}
		if (windowWidth % 2 == 0)
		{
			answer[(num4)] = (int)(meanVector[(num2)] * (inputVector[(i)] - filterWindow[(num5 + windowWidth / 2)]));
			if (answer[(num4)] < 0)
			{
				answer[(num4)] = 0;
			}
			flag = false;
		}
		else
		{
			answer[(num4)] = (int)((double)meanVector[(num2)] * ((double)inputVector[(i)] - (double)(filterWindow[(num5 + (windowWidth - 1) / 2)] + filterWindow[(num5 + (windowWidth - 1) / 2 + 1)]) / 2.0));
			if (answer[(num4)] < 0)
			{
				answer[(num4)] = 0;
			}
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
		num4 += inputVectorLen0 / depth * nStep;
		for (int m = num * depth + windowWidth + 1; m < num * depth + 2 * windowWidth + 1; m += nStep)
		{
			filterWindow[(num3)] = inputVector[(m)];
			for (int n = num * (2 * windowWidth + 1); n < num3; n++)
			{
				for (int num7 = num * (2 * windowWidth + 1); num7 < num3 - n; num7++)
				{
					if (filterWindow[(num7)] > filterWindow[(num7 + 1)])
					{
						float num6 = filterWindow[(num7)];
						filterWindow[(num7)] = filterWindow[(num7 + 1)];
						filterWindow[(num7 + 1)] = num6;
					}
				}
			}
			if (flag)
			{
				answer[(num4)] = (int)(meanVector[(num2)] * (inputVector[(i)] - filterWindow[(num5)]));
				if (answer[(num4)] < 0)
				{
					answer[(num4)] = 0;
				}
				flag = false;
				num5++;
			}
			else
			{
				answer[(num4)] = (int)((double)meanVector[(num2)] * ((double)inputVector[(i)] - (double)(filterWindow[(num5)] + filterWindow[(num5 - 1)]) / 2.0));
				if (answer[(num4)] < 0)
				{
					answer[(num4)] = 0;
				}
				flag = true;
			}
			float num8 = (float)(answer[(num4)] - answer[(num4 - inputVectorLen0 / depth * nStep)]);
			for (int num9 = 1; num9 < nStep; num9++)
			{
				answer[(num4 + inputVectorLen0 / depth * (nStep + num9))] = (int)((float)answer[(num4 - inputVectorLen0 / depth * nStep)] + (float)num9 * num8);
			}
			num2 += nStep;
			num3 += nStep;
			i += nStep;
			num4 += inputVectorLen0 / depth * nStep;
		}
		num3 = num * (2 * windowWidth + 1);
		num5 = num3 + windowWidth;
		int num10 = num5 + windowWidth;
		bool flag2 = false;
		int num11 = 0;
		while (i < (num + 1) * depth - windowWidth)
		{
			flag2 = false;
			num11 = num3;
			while (!flag2 && num11 < num10 + 1)
			{
				if (filterWindow[(num11)] == inputVector[(i - windowWidth - 1)])
				{
					flag2 = true;
					filterWindow[(num11)] = inputVector[(i + windowWidth)];
					if ((num11 != num3 || filterWindow[(num11)] >= filterWindow[(num11 + 1)]) && (num11 != num10 || filterWindow[(num11)] <= filterWindow[(num11 - 1)]))
					{
						if (num11 == num3 && filterWindow[(num11)] > filterWindow[(num11 + 1)])
						{
							while (filterWindow[(num11)] > filterWindow[(num11 + 1)])
							{
								if (num11 >= num10)
								{
									break;
								}
								float num6 = filterWindow[(num11 + 1)];
								filterWindow[(num11 + 1)] = filterWindow[(num11)];
								filterWindow[(num11)] = num6;
								num11++;
							}
						}
						else
						{
							if (num11 == num10 && filterWindow[(num11)] < filterWindow[(num11 - 1)])
							{
								while (filterWindow[(num11)] < filterWindow[(num11 - 1)])
								{
									if (num11 <= num3)
									{
										break;
									}
									float num6 = filterWindow[(num11 - 1)];
									filterWindow[(num11 - 1)] = filterWindow[(num11)];
									filterWindow[(num11)] = num6;
									num11--;
								}
							}
							else
							{
								if (filterWindow[(num11)] != filterWindow[(num11 + 1)] && filterWindow[(num11)] != filterWindow[(num11 - 1)])
								{
									if (filterWindow[(num11)] > filterWindow[(num11 + 1)])
									{
										while (filterWindow[(num11)] > filterWindow[(num11 + 1)])
										{
											if (num11 >= num10)
											{
												break;
											}
											float num6 = filterWindow[(num11 + 1)];
											filterWindow[(num11 + 1)] = filterWindow[(num11)];
											filterWindow[(num11)] = num6;
											num11++;
										}
									}
									else
									{
										while (filterWindow[(num11)] < filterWindow[(num11 - 1)] && num11 > num3)
										{
											float num6 = filterWindow[(num11 - 1)];
											filterWindow[(num11 - 1)] = filterWindow[(num11)];
											filterWindow[(num11)] = num6;
											num11--;
										}
									}
								}
							}
						}
					}
				}
				num11++;
			}
			answer[(num4)] = (int)(meanVector[(num2)] * (inputVector[(i)] - filterWindow[(num5)]));
			if (answer[(num4)] < 0)
			{
				answer[(num4)] = 0;
			}
			float num12 = (float)(answer[(num4)] - answer[(num4 - inputVectorLen0 / depth * nStep)]);
			for (int num13 = 1; num13 < nStep; num13++)
			{
				answer[(num4 + inputVectorLen0 / depth * (nStep + num13))] = (int)((float)answer[(num4 - inputVectorLen0 / depth * nStep)] + (float)num13 * num12);
			}
			num2 += nStep;
			num3 += nStep;
			i += nStep;
			num4 += inputVectorLen0 / depth * nStep;
		}
		flag = false;
		while (i < (num + 1) * depth)
		{
			if (flag)
			{
				answer[(num4)] = (int)(meanVector[(num2)] * (inputVector[(i)] - filterWindow[(num5)]));
				if (answer[(num4)] < 0)
				{
					answer[(num4)] = 0;
				}
				flag = false;
			}
			else
			{
				answer[(num4)] = (int)((double)meanVector[(num2)] * ((double)inputVector[(i)] - (double)(filterWindow[(num5)] + filterWindow[(num5 + 1)]) / 2.0));
				if (answer[(num4)] < 0)
				{
					answer[(num4)] = 0;
				}
				flag = true;
				num5++;
			}
			float num14 = (float)(answer[(num4)] - answer[(num4 - inputVectorLen0 / depth * nStep)]);
			for (int num15 = 1; num15 < nStep; num15++)
			{
				answer[(num4 + inputVectorLen0 / depth * (nStep + num15))] = (int)((float)answer[(num4 - inputVectorLen0 / depth * nStep)] + (float)num15 * num14);
			}
			num2 += nStep;
			num3 += nStep;
			i += nStep;
			num4 += inputVectorLen0 / depth * nStep;
		}
		num4 -= 2 * (inputVectorLen0 / depth) * nStep;
		i -= nStep;
		num2 -= nStep;
		if (i != (num + 1) * depth - 1)
		{
			if (flag)
			{
				answer[(num + (depth - 1) * (inputVectorLen0 / depth))] = (int)(meanVector[(num2)] * (inputVector[(i)] - filterWindow[(num5)]));
				if (answer[((num + 1) * depth - 1)] < 0)
				{
					answer[((num + 1) * depth - 1)] = 0;
				}
			}
			else
			{
				answer[(num + (depth - 1) * (inputVectorLen0 / depth))] = (int)((double)meanVector[(num2)] * ((double)inputVector[(i)] - (double)(filterWindow[(num5)] + filterWindow[(num5 + 1)]) / 2.0));
				if (answer[((num + 1) * depth - 1)] < 0)
				{
					answer[((num + 1) * depth - 1)] = 0;
				}
			}
			float num16 = (float)(answer[(num + (depth - 1) * (inputVectorLen0 / depth))] - answer[(num4)]);
			for (int num17 = 1; num17 < (num + 1) * depth - 1 - i; num17++)
			{
				answer[(num4 + inputVectorLen0 / depth * num17)] = (int)((float)answer[(num4)] + (float)num17 * num16);
			}
		}
	}
}
