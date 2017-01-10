
// KernelDevelopment.gaussFit
extern "C" __global__  void gaussFitter( int* gaussVector, int gaussVectorLen0,  double* P, int PLen0, unsigned short windowWidth,  double* bounds, int boundsLen0,  double* stepSize, int stepSizeLen0, double convCriteria, int maxIterations);

// KernelDevelopment.gaussFit
extern "C" __global__  void gaussFitter( int* gaussVector, int gaussVectorLen0,  double* P, int PLen0, unsigned short windowWidth,  double* bounds, int boundsLen0,  double* stepSize, int stepSizeLen0, double convCriteria, int maxIterations)
{
	int x = blockIdx.x;
	int y = blockIdx.y;
	int num = x + gridDim.x * y;
	if (num < gaussVectorLen0 / (int)(windowWidth * windowWidth))
	{
		int num2 = 7 * num;
		int num3 = (int)(windowWidth * windowWidth) * num;
		double num4 = 0.0;
		double num5 = 0.0;
		double num6 = 0.0;
		for (int i = 0; i < (int)(windowWidth * windowWidth); i++)
		{
			num6 += (double)gaussVector[(num3 + i)];
			num4 += (double)(i % (int)windowWidth * gaussVector[(num3 + i)]);
			num5 += (double)(i / (int)windowWidth * gaussVector[(num3 + i)]);
		}
		P[(num2 + 1)] = num4 / num6;
		P[(num2 + 2)] = num5 / num6;
		num6 /= (double)(windowWidth * windowWidth);
		double num7 = 0.0;
		for (int j = 0; j < (int)(windowWidth * windowWidth); j++)
		{
			num7 += ((double)gaussVector[(num3 + j)] - num6) * ((double)gaussVector[(num3 + j)] - num6);
		}
		bool flag = true;
		int num8 = 0;
		double num9 = 1.0;
		double num10 = num9;
		int num11 = 0;
		double num12 = 0.0;
		double num13 = 0.0;
		double num14 = 0.0;
		double num15 = 0.0;
		int k = 0;
		double num16 = 0.0;
		double num17 = P[(num2)] * bounds[(0)];
		double num18 = P[(num2)] * bounds[(1)];
		double num19 = P[(num2)] * bounds[(12)];
		double num20 = P[(num2)] * bounds[(13)];
		stepSize[(num2)] *= P[(num2)];
		stepSize[(num2 + 6)] *= P[(num2)];
		double num21 = bounds[(6)];
		double num22 = bounds[(8)];
		while (num21 <= bounds[(7)])
		{
			num12 = 1.0 / (2.0 * num21 * num21);
			while (num22 <= bounds[(9)])
			{
				num14 = 1.0 / (2.0 * num22 * num22);
				num15 = 0.0;
				for (k = 0; k < (int)(windowWidth * windowWidth); k++)
				{
					int num23 = k % (int)windowWidth;
					int num24 = k / (int)windowWidth;
					double num25 = P[(num2)] * exp(-(num12 * ((double)num23 - P[(num2 + 1)]) * ((double)num23 - P[(num2 + 1)]) + num14 * ((double)num24 - P[(num2 + 2)]) * ((double)num24 - P[(num2 + 2)]))) - (double)gaussVector[(num3 + k)];
					num15 += num25 * num25;
				}
				num15 /= num7;
				if (num15 < num9)
				{
					num9 = num15;
					P[(num2 + 3)] = num21;
					P[(num2 + 4)] = num22;
				}
				num22 += stepSize[(num2 + 4)];
			}
			num21 += stepSize[(num2 + 3)];
			num22 = bounds[(8)];
		}
		num9 = 1.0;
		num12 = 1.0 / (2.0 * P[(num2 + 3)] * P[(num2 + 3)]);
		num13 = 0.0;
		num14 = 1.0 / (2.0 * P[(num2 + 4)] * P[(num2 + 4)]);
		while (flag)
		{
			if (num11 == 0)
			{
				num10 = num9;
				if (P[(num2 + num11)] + stepSize[(num2 + num11)] > num17 && P[(num2 + num11)] + stepSize[(num2 + num11)] < num18)
				{
					P[(num2 + num11)] += stepSize[(num2 + num11)];
					num15 = 0.0;
					for (k = 0; k < (int)(windowWidth * windowWidth); k++)
					{
						int num23 = k % (int)windowWidth;
						int num24 = k / (int)windowWidth;
						double num25 = P[(num2)] * exp(-(num12 * ((double)num23 - P[(num2 + 1)]) * ((double)num23 - P[(num2 + 1)]) - 2.0 * num13 * ((double)num23 - P[(num2 + 1)]) * ((double)num24 - P[(num2 + 2)]) + num14 * ((double)num24 - P[(num2 + 2)]) * ((double)num24 - P[(num2 + 2)]))) + P[(num2 + 6)] - (double)gaussVector[(num3 + k)];
						num15 += num25 * num25;
					}
					num15 /= num7;
					if (num15 < num9)
					{
						num9 = num15;
					}
					else
					{
						P[(num2 + num11)] -= stepSize[(num2 + num11)];
						if (stepSize[(num2 + num11)] < 0.0)
						{
							stepSize[(num2 + num11)] *= -0.6667;
						}
						else
						{
							stepSize[(num2 + num11)] *= -1.0;
						}
					}
				}
				else
				{
					if (stepSize[(num2 + num11)] < 0.0)
					{
						stepSize[(num2 + num11)] *= -0.6667;
					}
					else
					{
						stepSize[(num2 + num11)] *= -1.0;
					}
				}
			}
			else
			{
				if (num11 == 6)
				{
					if (P[(num2 + num11)] + stepSize[(num2 + num11)] > num19 && P[(num2 + num11)] + stepSize[(num2 + num11)] < num20)
					{
						P[(num2 + num11)] += stepSize[(num2 + num11)];
						num15 = 0.0;
						for (k = 0; k < (int)(windowWidth * windowWidth); k++)
						{
							int num23 = k % (int)windowWidth;
							int num24 = k / (int)windowWidth;
							double num25 = P[(num2)] * exp(-(num12 * ((double)num23 - P[(num2 + 1)]) * ((double)num23 - P[(num2 + 1)]) - 2.0 * num13 * ((double)num23 - P[(num2 + 1)]) * ((double)num24 - P[(num2 + 2)]) + num14 * ((double)num24 - P[(num2 + 2)]) * ((double)num24 - P[(num2 + 2)]))) + P[(num2 + 6)] - (double)gaussVector[(num3 + k)];
							num15 += num25 * num25;
						}
						num15 /= num7;
						if (num15 < num9)
						{
							num9 = num15;
						}
						else
						{
							P[(num2 + num11)] -= stepSize[(num2 + num11)];
							if (stepSize[(num2 + num11)] < 0.0)
							{
								stepSize[(num2 + num11)] *= -0.6667;
							}
							else
							{
								stepSize[(num2 + num11)] *= -1.0;
							}
						}
					}
					else
					{
						if (stepSize[(num2 + num11)] < 0.0)
						{
							stepSize[(num2 + num11)] *= -0.6667;
						}
						else
						{
							stepSize[(num2 + num11)] *= -1.0;
						}
					}
				}
				else
				{
					if (flag)
					{
						if (P[(num2 + num11)] + stepSize[(num2 + num11)] > bounds[(2 * num11)] && P[(num2 + num11)] + stepSize[(num2 + num11)] < bounds[(2 * num11 + 1)])
						{
							P[(num2 + num11)] += stepSize[(num2 + num11)];
							num12 = cos(P[(num2 + 5)]) * cos(P[(num2 + 5)]) / (2.0 * P[(num2 + 3)] * P[(num2 + 3)]) + sin(P[(num2 + 5)]) * sin(P[(num2 + 5)]) / (2.0 * P[(num2 + 4)] * P[(num2 + 4)]);
							num13 = -sin(2.0 * P[(num2 + 5)]) / (4.0 * P[(num2 + 3)] * P[(num2 + 3)]) + sin(2.0 * P[(num2 + 5)]) / (4.0 * P[(num2 + 4)] * P[(num2 + 4)]);
							num14 = sin(P[(num2 + 5)]) * sin(P[(num2 + 5)]) / (2.0 * P[(num2 + 3)] * P[(num2 + 3)]) + cos(P[(num2 + 5)]) * cos(P[(num2 + 5)]) / (2.0 * P[(num2 + 4)] * P[(num2 + 4)]);
							num15 = 0.0;
							for (k = 0; k < (int)(windowWidth * windowWidth); k++)
							{
								int num23 = k % (int)windowWidth;
								int num24 = k / (int)windowWidth;
								double num25 = P[(num2)] * exp(-(num12 * ((double)num23 - P[(num2 + 1)]) * ((double)num23 - P[(num2 + 1)]) - 2.0 * num13 * ((double)num23 - P[(num2 + 1)]) * ((double)num24 - P[(num2 + 2)]) + num14 * ((double)num24 - P[(num2 + 2)]) * ((double)num24 - P[(num2 + 2)]))) + P[(num2 + 6)] - (double)gaussVector[(num3 + k)];
								num15 += num25 * num25;
							}
							num15 /= num7;
							if (num15 < num9)
							{
								num9 = num15;
							}
							else
							{
								P[(num2 + num11)] -= stepSize[(num2 + num11)];
								num12 = cos(P[(num2 + 5)]) * cos(P[(num2 + 5)]) / (2.0 * P[(num2 + 3)] * P[(num2 + 3)]) + sin(P[(num2 + 5)]) * sin(P[(num2 + 5)]) / (2.0 * P[(num2 + 4)] * P[(num2 + 4)]);
								num13 = -sin(2.0 * P[(num2 + 5)]) / (4.0 * P[(num2 + 3)] * P[(num2 + 3)]) + sin(2.0 * P[(num2 + 5)]) / (4.0 * P[(num2 + 4)] * P[(num2 + 4)]);
								num14 = sin(P[(num2 + 5)]) * sin(P[(num2 + 5)]) / (2.0 * P[(num2 + 3)] * P[(num2 + 3)]) + cos(P[(num2 + 5)]) * cos(P[(num2 + 5)]) / (2.0 * P[(num2 + 4)] * P[(num2 + 4)]);
								if (stepSize[(num2 + num11)] < 0.0)
								{
									stepSize[(num2 + num11)] *= -0.6667;
								}
								else
								{
									stepSize[(num2 + num11)] *= -1.0;
								}
							}
						}
						else
						{
							if (stepSize[(num2 + num11)] < 0.0)
							{
								stepSize[(num2 + num11)] *= -0.6667;
							}
							else
							{
								stepSize[(num2 + num11)] *= -1.0;
							}
						}
					}
				}
			}
			num11++;
			num8++;
			if (num11 > 6)
			{
				if (num8 > 50 && num10 - num9 < convCriteria)
				{
					flag = false;
				}
				num11 = 0;
			}
			if (num8 > maxIterations)
			{
				flag = false;
			}
		}
		num12 = cos(P[(num2 + 5)]) * cos(P[(num2 + 5)]) / (2.0 * P[(num2 + 3)] * P[(num2 + 3)]) + sin(P[(num2 + 5)]) * sin(P[(num2 + 5)]) / (2.0 * P[(num2 + 4)] * P[(num2 + 4)]);
		num13 = -sin(2.0 * P[(num2 + 5)]) / (4.0 * P[(num2 + 3)] * P[(num2 + 3)]) + sin(2.0 * P[(num2 + 5)]) / (4.0 * P[(num2 + 4)] * P[(num2 + 4)]);
		num14 = sin(P[(num2 + 5)]) * sin(P[(num2 + 5)]) / (2.0 * P[(num2 + 3)] * P[(num2 + 3)]) + cos(P[(num2 + 5)]) * cos(P[(num2 + 5)]) / (2.0 * P[(num2 + 4)] * P[(num2 + 4)]);
		num15 = 0.0;
		for (k = 0; k < (int)(windowWidth * windowWidth); k++)
		{
			int num23 = k % (int)windowWidth;
			int num24 = k / (int)windowWidth;
			double num25 = P[(num2)] * exp(-(num12 * ((double)num23 - P[(num2 + 1)]) * ((double)num23 - P[(num2 + 1)]) - 2.0 * num13 * ((double)num23 - P[(num2 + 1)]) * ((double)num24 - P[(num2 + 2)]) + num14 * ((double)num24 - P[(num2 + 2)]) * ((double)num24 - P[(num2 + 2)]))) + P[(num2 + 6)];
			num16 += num25;
			num25 -= (double)gaussVector[(num3 + k)];
			num15 += num25 * num25;
		}
		num15 /= num7;
		P[(num2)] = num16;
		P[(num2 + 6)] = 1.0 - num15;
	}
}
