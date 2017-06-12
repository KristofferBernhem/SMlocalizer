
// KernelDevelopment.gaussFit
extern "C" __global__  void gaussFitter(int n,  int* gaussVector, int gaussVectorLen0,  double* P, int PLen0, unsigned short windowWidth,  double* bounds, int boundsLen0,  double* stepSize, int stepSizeLen0, double convCriteria, int maxIterations);

// KernelDevelopment.gaussFit
extern "C" __global__  void gaussFitter(int n,  int* gaussVector, int gaussVectorLen0,  double* P, int PLen0, unsigned short windowWidth,  double* bounds, int boundsLen0,  double* stepSize, int stepSizeLen0, double convCriteria, int maxIterations)
{
	int num = blockIdx.x * blockDim.x + threadIdx.x;
	int num2 = blockDim.x * gridDim.x;
	for (int i = num; i < n; i += num2)
	{
		int num3 = 7 * i;
		int num4 = (int)(windowWidth * windowWidth) * i;
		double num5 = 0.0;
		double num6 = 0.0;
		double num7 = 0.0;
		for (int j = 0; j < (int)(windowWidth * windowWidth); j++)
		{
			num7 += (double)gaussVector[(num4 + j)];
			num5 += (double)(j % (int)windowWidth * gaussVector[(num4 + j)]);
			num6 += (double)(j / (int)windowWidth * gaussVector[(num4 + j)]);
		}
		P[(num3 + 1)] = num5 / num7;
		P[(num3 + 2)] = num6 / num7;
		num7 /= (double)(windowWidth * windowWidth);
		double num8 = 0.0;
		for (int k = 0; k < (int)(windowWidth * windowWidth); k++)
		{
			num8 += ((double)gaussVector[(num4 + k)] - num7) * ((double)gaussVector[(num4 + k)] - num7);
		}
		bool flag = true;
		int num9 = 0;
		double num10 = 1.0;
		double num11 = num10;
		int num12 = 0;
		double num13 = 0.0;
		double num14 = 0.0;
		double num15 = 0.0;
		double num16 = 0.0;
		int l = 0;
		double num17 = 0.0;
		double num18 = P[(num3)] * bounds[(0)];
		double num19 = P[(num3)] * bounds[(1)];
		double num20 = P[(num3)] * bounds[(12)];
		double num21 = P[(num3)] * bounds[(13)];
		stepSize[(num3)] *= P[(num3)];
		stepSize[(num3 + 6)] *= P[(num3)];
		double num22 = bounds[(6)];
		double num23 = bounds[(8)];
		while (num22 <= bounds[(7)])
		{
			num13 = 1.0 / (2.0 * num22 * num22);
			while (num23 <= bounds[(9)])
			{
				num15 = 1.0 / (2.0 * num23 * num23);
				num16 = 0.0;
				for (l = 0; l < (int)(windowWidth * windowWidth); l++)
				{
					int num24 = l % (int)windowWidth;
					int num25 = l / (int)windowWidth;
					double num26 = P[(num3)] * exp(-(num13 * ((double)num24 - P[(num3 + 1)]) * ((double)num24 - P[(num3 + 1)]) + num15 * ((double)num25 - P[(num3 + 2)]) * ((double)num25 - P[(num3 + 2)]))) - (double)gaussVector[(num4 + l)];
					num16 += num26 * num26;
				}
				num16 /= num8;
				if (num16 < num10)
				{
					num10 = num16;
					P[(num3 + 3)] = num22;
					P[(num3 + 4)] = num23;
				}
				num23 += stepSize[(num3 + 4)];
			}
			num22 += stepSize[(num3 + 3)];
			num23 = bounds[(8)];
		}
		num10 = 1.0;
		num13 = 1.0 / (2.0 * P[(num3 + 3)] * P[(num3 + 3)]);
		num14 = 0.0;
		num15 = 1.0 / (2.0 * P[(num3 + 4)] * P[(num3 + 4)]);
		while (flag)
		{
			if (num12 == 0)
			{
				num11 = num10;
				if (P[(num3 + num12)] + stepSize[(num3 + num12)] > num18 && P[(num3 + num12)] + stepSize[(num3 + num12)] < num19)
				{
					P[(num3 + num12)] += stepSize[(num3 + num12)];
					num16 = 0.0;
					for (l = 0; l < (int)(windowWidth * windowWidth); l++)
					{
						int num24 = l % (int)windowWidth;
						int num25 = l / (int)windowWidth;
						double num26 = P[(num3)] * exp(-(num13 * ((double)num24 - P[(num3 + 1)]) * ((double)num24 - P[(num3 + 1)]) - 2.0 * num14 * ((double)num24 - P[(num3 + 1)]) * ((double)num25 - P[(num3 + 2)]) + num15 * ((double)num25 - P[(num3 + 2)]) * ((double)num25 - P[(num3 + 2)]))) + P[(num3 + 6)] - (double)gaussVector[(num4 + l)];
						num16 += num26 * num26;
					}
					num16 /= num8;
					if (num16 < num10)
					{
						num10 = num16;
					}
					else
					{
						P[(num3 + num12)] -= stepSize[(num3 + num12)];
						if (stepSize[(num3 + num12)] < 0.0)
						{
							if (num9 < 20)
							{
								stepSize[(num3 + num12)] *= -0.3;
							}
							else
							{
								stepSize[(num3 + num12)] *= -0.7;
							}
						}
						else
						{
							stepSize[(num3 + num12)] *= -1.0;
						}
					}
				}
				else
				{
					if (stepSize[(num3 + num12)] < 0.0)
					{
						if (num9 < 20)
						{
							stepSize[(num3 + num12)] *= -0.3;
						}
						else
						{
							stepSize[(num3 + num12)] *= -0.7;
						}
					}
					else
					{
						stepSize[(num3 + num12)] *= -1.0;
					}
				}
			}
			else
			{
				if (num12 == 6)
				{
					if (P[(num3 + num12)] + stepSize[(num3 + num12)] > num20 && P[(num3 + num12)] + stepSize[(num3 + num12)] < num21)
					{
						P[(num3 + num12)] += stepSize[(num3 + num12)];
						num16 = 0.0;
						for (l = 0; l < (int)(windowWidth * windowWidth); l++)
						{
							int num24 = l % (int)windowWidth;
							int num25 = l / (int)windowWidth;
							double num26 = P[(num3)] * exp(-(num13 * ((double)num24 - P[(num3 + 1)]) * ((double)num24 - P[(num3 + 1)]) - 2.0 * num14 * ((double)num24 - P[(num3 + 1)]) * ((double)num25 - P[(num3 + 2)]) + num15 * ((double)num25 - P[(num3 + 2)]) * ((double)num25 - P[(num3 + 2)]))) + P[(num3 + 6)] - (double)gaussVector[(num4 + l)];
							num16 += num26 * num26;
						}
						num16 /= num8;
						if (num16 < num10)
						{
							num10 = num16;
						}
						else
						{
							P[(num3 + num12)] -= stepSize[(num3 + num12)];
							if (stepSize[(num3 + num12)] < 0.0)
							{
								if (num9 < 20)
								{
									stepSize[(num3 + num12)] *= -0.3;
								}
								else
								{
									stepSize[(num3 + num12)] *= -0.7;
								}
							}
							else
							{
								stepSize[(num3 + num12)] *= -1.0;
							}
						}
					}
					else
					{
						if (stepSize[(num3 + num12)] < 0.0)
						{
							if (num9 < 20)
							{
								stepSize[(num3 + num12)] *= -0.3;
							}
							else
							{
								stepSize[(num3 + num12)] *= -0.7;
							}
						}
						else
						{
							stepSize[(num3 + num12)] *= -1.0;
						}
					}
				}
				else
				{
					if (flag)
					{
						if (P[(num3 + num12)] + stepSize[(num3 + num12)] > bounds[(2 * num12)] && P[(num3 + num12)] + stepSize[(num3 + num12)] < bounds[(2 * num12 + 1)])
						{
							P[(num3 + num12)] += stepSize[(num3 + num12)];
							num13 = cos(P[(num3 + 5)]) * cos(P[(num3 + 5)]) / (2.0 * P[(num3 + 3)] * P[(num3 + 3)]) + sin(P[(num3 + 5)]) * sin(P[(num3 + 5)]) / (2.0 * P[(num3 + 4)] * P[(num3 + 4)]);
							num14 = -sin(2.0 * P[(num3 + 5)]) / (4.0 * P[(num3 + 3)] * P[(num3 + 3)]) + sin(2.0 * P[(num3 + 5)]) / (4.0 * P[(num3 + 4)] * P[(num3 + 4)]);
							num15 = sin(P[(num3 + 5)]) * sin(P[(num3 + 5)]) / (2.0 * P[(num3 + 3)] * P[(num3 + 3)]) + cos(P[(num3 + 5)]) * cos(P[(num3 + 5)]) / (2.0 * P[(num3 + 4)] * P[(num3 + 4)]);
							num16 = 0.0;
							for (l = 0; l < (int)(windowWidth * windowWidth); l++)
							{
								int num24 = l % (int)windowWidth;
								int num25 = l / (int)windowWidth;
								double num26 = P[(num3)] * exp(-(num13 * ((double)num24 - P[(num3 + 1)]) * ((double)num24 - P[(num3 + 1)]) - 2.0 * num14 * ((double)num24 - P[(num3 + 1)]) * ((double)num25 - P[(num3 + 2)]) + num15 * ((double)num25 - P[(num3 + 2)]) * ((double)num25 - P[(num3 + 2)]))) + P[(num3 + 6)] - (double)gaussVector[(num4 + l)];
								num16 += num26 * num26;
							}
							num16 /= num8;
							if (num16 < num10)
							{
								num10 = num16;
							}
							else
							{
								P[(num3 + num12)] -= stepSize[(num3 + num12)];
								num13 = cos(P[(num3 + 5)]) * cos(P[(num3 + 5)]) / (2.0 * P[(num3 + 3)] * P[(num3 + 3)]) + sin(P[(num3 + 5)]) * sin(P[(num3 + 5)]) / (2.0 * P[(num3 + 4)] * P[(num3 + 4)]);
								num14 = -sin(2.0 * P[(num3 + 5)]) / (4.0 * P[(num3 + 3)] * P[(num3 + 3)]) + sin(2.0 * P[(num3 + 5)]) / (4.0 * P[(num3 + 4)] * P[(num3 + 4)]);
								num15 = sin(P[(num3 + 5)]) * sin(P[(num3 + 5)]) / (2.0 * P[(num3 + 3)] * P[(num3 + 3)]) + cos(P[(num3 + 5)]) * cos(P[(num3 + 5)]) / (2.0 * P[(num3 + 4)] * P[(num3 + 4)]);
								if (stepSize[(num3 + num12)] < 0.0)
								{
									if (num9 < 20)
									{
										stepSize[(num3 + num12)] *= -0.3;
									}
									else
									{
										stepSize[(num3 + num12)] *= -0.7;
									}
								}
								else
								{
									stepSize[(num3 + num12)] *= -1.0;
								}
							}
						}
						else
						{
							if (stepSize[(num3 + num12)] < 0.0)
							{
								if (num9 < 20)
								{
									stepSize[(num3 + num12)] *= -0.3;
								}
								else
								{
									stepSize[(num3 + num12)] *= -0.7;
								}
							}
							else
							{
								stepSize[(num3 + num12)] *= -1.0;
							}
						}
					}
				}
			}
			num12++;
			num9++;
			if (num12 > 6)
			{
				if (num9 > 250 && num11 - num10 < convCriteria)
				{
					flag = false;
				}
				num12 = 0;
			}
			if (num9 > maxIterations)
			{
				flag = false;
			}
		}
		num13 = cos(P[(num3 + 5)]) * cos(P[(num3 + 5)]) / (2.0 * P[(num3 + 3)] * P[(num3 + 3)]) + sin(P[(num3 + 5)]) * sin(P[(num3 + 5)]) / (2.0 * P[(num3 + 4)] * P[(num3 + 4)]);
		num14 = -sin(2.0 * P[(num3 + 5)]) / (4.0 * P[(num3 + 3)] * P[(num3 + 3)]) + sin(2.0 * P[(num3 + 5)]) / (4.0 * P[(num3 + 4)] * P[(num3 + 4)]);
		num15 = sin(P[(num3 + 5)]) * sin(P[(num3 + 5)]) / (2.0 * P[(num3 + 3)] * P[(num3 + 3)]) + cos(P[(num3 + 5)]) * cos(P[(num3 + 5)]) / (2.0 * P[(num3 + 4)] * P[(num3 + 4)]);
		num16 = 0.0;
		for (l = 0; l < (int)(windowWidth * windowWidth); l++)
		{
			int num24 = l % (int)windowWidth;
			int num25 = l / (int)windowWidth;
			double num26 = P[(num3)] * exp(-(num13 * ((double)num24 - P[(num3 + 1)]) * ((double)num24 - P[(num3 + 1)]) - 2.0 * num14 * ((double)num24 - P[(num3 + 1)]) * ((double)num25 - P[(num3 + 2)]) + num15 * ((double)num25 - P[(num3 + 2)]) * ((double)num25 - P[(num3 + 2)]))) + P[(num3 + 6)];
			num17 += num26;
			num26 -= (double)gaussVector[(num4 + l)];
			num16 += num26 * num26;
		}
		num16 /= num8;
		P[(num3)] = num17;
		P[(num3 + 6)] = 1.0 - num16;
	}
}
