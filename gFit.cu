
// KernelDevelopment.gaussFitShort
extern "C" __global__  void gaussFitter( int* gaussVector, int gaussVectorLen0,  float* P, int PLen0, int windowWidth,  float* bounds, int boundsLen0,  float* stepSize, int stepSizeLen0, double convCriteria, int maxIterations);

// KernelDevelopment.gaussFitShort
extern "C" __global__  void gaussFitter( int* gaussVector, int gaussVectorLen0,  float* P, int PLen0, int windowWidth,  float* bounds, int boundsLen0,  float* stepSize, int stepSizeLen0, double convCriteria, int maxIterations)
{
	int num = blockIdx.x + gridDim.x * blockIdx.y;
	if (num < gaussVectorLen0 / (windowWidth * windowWidth))
	{
		int num2 = 7 * num;
		int num3 = windowWidth * windowWidth * num;
		float num4 = 0.0f;
		float num5 = 0.0f;
		float num6 = 0.0f;
		for (int i = 0; i < windowWidth * windowWidth; i++)
		{
			num6 += (float)gaussVector[(num3 + i)];
			num4 += (float)(i % windowWidth * gaussVector[(num3 + i)]);
			num5 += (float)(i / windowWidth * gaussVector[(num3 + i)]);
		}
		P[(num2 + 1)] = num4 / num6;
		P[(num2 + 2)] = num5 / num6;
		num6 /= (float)(windowWidth * windowWidth);
		float num7 = 0.0f;
		for (int j = 0; j < windowWidth * windowWidth; j++)
		{
			num7 += ((float)gaussVector[(num3 + j)] - num6) * ((float)gaussVector[(num3 + j)] - num6);
		}
		bool flag = true;
		int num8 = 0;
		int num9 = 0;
		float num10 = 1.0f;
		float num11 = num10;
		float num12 = 0.0f;
		float num13 = 0.0f;
		float num14 = 0.0f;
		float num15 = 0.0f;
		int k = 0;
		float num16 = 0.0f;
		stepSize[(num2)] *= P[(num2)];
		stepSize[(num2 + 6)] *= P[(num2)];
		float num17 = P[(num2)] * bounds[(0)];
		float num18 = P[(num2)] * bounds[(1)];
		float num19 = P[(num2)] * bounds[(12)];
		float num20 = P[(num2)] * bounds[(13)];
		bool flag2 = false;
		for (float num21 = P[(num2 + 3)] - 3.0f * stepSize[(num2 + 3)]; num21 <= P[(num2 + 3)] + 2.0f * stepSize[(num2 + 3)]; num21 += stepSize[(num2 + 3)])
		{
			num12 = 1.0f / (2.0f * num21 * num21);
			for (float num22 = P[(num2 + 4)] - 3.0f * stepSize[(num2 + 4)]; num22 <= P[(num2 + 4)] + 2.0f * stepSize[(num2 + 4)]; num22 += stepSize[(num2 + 4)])
			{
				num14 = 1.0f / (2.0f * num22 * num22);
				num15 = 0.0f;
				for (k = 0; k < windowWidth * windowWidth; k++)
				{
					int num23 = k % windowWidth;
					int num24 = k / windowWidth;
					float num25 = (float)((double)P[(num2)] * exp((double)(-(double)(num12 * ((float)num23 - P[(num2 + 1)]) * ((float)num23 - P[(num2 + 1)]) - 2.0f * num13 * ((float)num23 - P[(num2 + 1)]) * ((float)num24 - P[(num2 + 2)]) + num14 * ((float)num24 - P[(num2 + 2)]) * ((float)num24 - P[(num2 + 2)]))))) - (float)gaussVector[(num3 + k)];
					num15 += num25 * num25;
				}
				num15 /= num7;
				if (num15 < num10)
				{
					num10 = num15;
					num17 = num21;
					num18 = num22;
				}
			}
		}
		P[(num2 + 3)] = num17;
		P[(num2 + 4)] = num18;
		num17 = P[(num2)] * bounds[(0)];
		num18 = P[(num2)] * bounds[(1)];
		num10 = 1.0f;
		num12 = 1.0f / (2.0f * P[(num2 + 3)] * P[(num2 + 3)]);
		num13 = 0.0f;
		num14 = 1.0f / (2.0f * P[(num2 + 4)] * P[(num2 + 4)]);
		while (flag)
		{
			flag2 = false;
			if (num9 == 0)
			{
				num11 = num10;
			}
			if (num9 == 0)
			{
				if (P[(num2)] + stepSize[(num2)] > num17 && P[(num2)] + stepSize[(num2)] < num18)
				{
					flag2 = true;
				}
			}
			else
			{
				if (num9 == 6)
				{
					if (P[(num2 + 6)] + stepSize[(num2 + 6)] > num19 && P[(num2 + 6)] + stepSize[(num2 + 6)] < num20)
					{
						flag2 = true;
					}
				}
				else
				{
					if (P[(num2 + num9)] + stepSize[(num2 + num9)] > bounds[(num9 * 2)] && P[(num2 + num9)] + stepSize[(num2 + num9)] < bounds[(num9 * 2 + 1)])
					{
						flag2 = true;
					}
				}
			}
			if (flag2)
			{
				P[(num2 + num9)] += stepSize[(num2 + num9)];
				if (num9 == 3 || num9 == 4 || num9 == 5)
				{
					num12 = (float)(cos((double)P[(num2 + 5)]) * cos((double)P[(num2 + 5)]) / (double)(2.0f * P[(num2 + 3)] * P[(num2 + 3)]) + sin((double)P[(num2 + 5)]) * sin((double)P[(num2 + 5)]) / (double)(2.0f * P[(num2 + 4)] * P[(num2 + 4)]));
					num13 = (float)(-(float)sin((double)(2.0f * P[(num2 + 5)])) / (double)(4.0f * P[(num2 + 3)] * P[(num2 + 3)]) + sin((double)(2.0f * P[(num2 + 5)])) / (double)(4.0f * P[(num2 + 4)] * P[(num2 + 4)]));
					num14 = (float)(sin((double)P[(num2 + 5)]) * sin((double)P[(num2 + 5)]) / (double)(2.0f * P[(num2 + 3)] * P[(num2 + 3)]) + cos((double)P[(num2 + 5)]) * cos((double)P[(num2 + 5)]) / (double)(2.0f * P[(num2 + 4)] * P[(num2 + 4)]));
				}
				num15 = 0.0f;
				for (k = 0; k < windowWidth * windowWidth; k++)
				{
					int num23 = k % windowWidth;
					int num24 = k / windowWidth;
					float num25 = (float)((double)P[(num2)] * exp((double)(-(double)(num12 * ((float)num23 - P[(num2 + 1)]) * ((float)num23 - P[(num2 + 1)]) - 2.0f * num13 * ((float)num23 - P[(num2 + 1)]) * ((float)num24 - P[(num2 + 2)]) + num14 * ((float)num24 - P[(num2 + 2)]) * ((float)num24 - P[(num2 + 2)]))))) + P[(num2 + 6)] - (float)gaussVector[(num3 + k)];
					num15 += num25 * num25;
				}
				num15 /= num7;
				if (num15 < num10)
				{
					num10 = num15;
				}
				else
				{
					P[(num2 + num9)] -= stepSize[(num2 + num9)];
					if (stepSize[(num2 + num9)] < 0.0f)
					{
						stepSize[(num2 + num9)] /= -1.5f;
					}
					else
					{
						stepSize[(num2 + num9)] *= -1.0f;
					}
				}
			}
			else
			{
				if (stepSize[(num2 + num9)] < 0.0f)
				{
					stepSize[(num2 + num9)] /= -1.5f;
				}
				else
				{
					stepSize[(num2 + num9)] *= -1.0f;
				}
			}
			num9++;
			if (num9 > 6)
			{
				if (num8 > 40 && (double)(num11 - num10) < convCriteria)
				{
					flag = false;
				}
				num9 = 0;
			}
			num8++;
			if (num8 > maxIterations)
			{
				flag = false;
			}
		}
		num12 = (float)(cos((double)P[(num2 + 5)]) * cos((double)P[(num2 + 5)]) / (double)(2.0f * P[(num2 + 3)] * P[(num2 + 3)]) + sin((double)P[(num2 + 5)]) * sin((double)P[(num2 + 5)]) / (double)(2.0f * P[(num2 + 4)] * P[(num2 + 4)]));
		num13 = (float)(-(float)sin((double)(2.0f * P[(num2 + 5)])) / (double)(4.0f * P[(num2 + 3)] * P[(num2 + 3)]) + sin((double)(2.0f * P[(num2 + 5)])) / (double)(4.0f * P[(num2 + 4)] * P[(num2 + 4)]));
		num14 = (float)(sin((double)P[(num2 + 5)]) * sin((double)P[(num2 + 5)]) / (double)(2.0f * P[(num2 + 3)] * P[(num2 + 3)]) + cos((double)P[(num2 + 5)]) * cos((double)P[(num2 + 5)]) / (double)(2.0f * P[(num2 + 4)] * P[(num2 + 4)]));
		num15 = 0.0f;
		for (k = 0; k < windowWidth * windowWidth; k++)
		{
			int num23 = k % windowWidth;
			int num24 = k / windowWidth;
			float num25 = (float)((double)P[(num2)] * exp((double)(-(double)(num12 * ((float)num23 - P[(num2 + 1)]) * ((float)num23 - P[(num2 + 1)]) - 2.0f * num13 * ((float)num23 - P[(num2 + 1)]) * ((float)num24 - P[(num2 + 2)]) + num14 * ((float)num24 - P[(num2 + 2)]) * ((float)num24 - P[(num2 + 2)]))))) + P[(num2 + 6)];
			num16 += num25;
			num25 -= (float)gaussVector[(num3 + k)];
			num15 += num25 * num25;
		}
		num15 /= num7;
		P[(num2)] = num16;
		P[(num2 + 6)] = 1.0f - num15;
	}
}
