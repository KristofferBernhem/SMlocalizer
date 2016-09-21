/* Copyright 2016 Kristoffer Bernhem.
 * This file is part of SMLocalizer.
 *
 *  SMLocalizer is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  SMLocalizer is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with SMLocalizer.  If not, see <http://www.gnu.org/licenses/>.
 */

public class Gauss2Dfit {
	int[] inputdata;
	int width;
	int size;
	double totalSumOfSquares;

	public Gauss2Dfit(int[] inputdata, int window){ // Setup
		this.inputdata 	= inputdata;
		this.width 		= window;
		this.size 		= inputdata.length;
		double InputMean = 0;
		for (int i =0; i< size;i++){
			InputMean += inputdata[i];
		}
		InputMean /=  size;
		for (int i = 0; i < inputdata.length; i++){
			this.totalSumOfSquares += (inputdata[i] -InputMean)*(inputdata[i] -InputMean);
		}	
	}

	public double Eval_photons(double[] P){ // Evaluate gaussian with parameters from P:
		/*
		 * P[0]: Amplitude
		 * P[1]: x0
		 * P[2]: y0
		 * P[3]: sigma x
		 * P[4]: sigma y
		 * P[5]: theta
		 * P[6]: offset 
		 */		
		double ThetaA = Math.cos(P[5])*Math.cos(P[5])/(2*P[3]*P[3]) + Math.sin(P[5])*Math.sin(P[5])/(2*P[4]*P[4]); 
		double ThetaB = -Math.sin(2*P[5])/(4*P[3]*P[3]) + Math.sin(2*P[5])/(4*P[4]*P[4]); 
		double ThetaC = Math.sin(P[5])*Math.sin(P[5])/(2*P[3]*P[3]) + Math.cos(P[5])*Math.cos(P[5])/(2*P[4]*P[4]);

		double photons = 0;
		for (int i = 0; i < size; i++){
			int xi = i % width;
			int yi = i / width;	
			photons += P[0]*Math.exp(-(ThetaA*(xi - P[1])*(xi - P[1]) - 
					2*ThetaB*(xi - P[1])*(yi - P[2]) +
					ThetaC*(yi - P[2])*(yi - P[2])
					)) + P[6];

		}

		return photons;
	}

	public static void main(final String... args){
		/*		double[] inputdata = {
				10,  40,  70,  40, 10,
				40, 160, 260, 160, 40,
				70, 260, 810, 260, 70,
				40, 160, 260, 160, 40,
				10,  40,  70,  40, 10};*/
		int[] testdataFirst ={ // slize 45 SingleBead2
				3888, 3984,  6192,   4192, 3664,  3472, 3136,
				6384, 8192,  12368, 12720, 6032,  5360, 3408, 
				6192, 13760, 21536, 20528, 9744,  6192, 2896,
				6416, 15968, 25600, 28080, 12288, 4496, 2400,
				4816, 11312, 15376, 14816, 8016,  4512, 3360,
				2944, 4688,  7168,   5648, 5824,  3456, 2912,
				2784, 3168,  4512,   4192, 3472,  2768, 2912
		};

		int[] testdataSecond = {
				3296, 4544,  5600,  5536,  5248,  4448, 3328,
				3760, 5344,  8240,  9680, 10592,  7056, 3328,
				3744, 6672, 14256, 24224, 20256, 11136, 5248,
				3696, 7504, 16944, 26640, 21680, 10384, 5008,
				2992, 6816, 10672, 15536, 14464,  7792, 4016,
				2912, 3872,  4992,  6560,  6448,  4896, 3392,
				3088, 3248,  3552, 	3504,  4144,  4512, 2944  
		};


		/*		int[] inputdata = {
				0  ,12 ,25 ,12 ,0  ,
				12 ,89 ,153,89 ,12 ,
				25 ,153,255,153,25 ,
				12 ,89 ,153,89 ,12 ,
				0  ,12 ,25 ,12 ,0  ,
		};*/	
		int frame 			= 1;		// include in optimize call.
		int channel 		= 1;		// include in optimize call.
		int[] center 		= {5,5};	// include in optimize call.	
		int inpwidth 		= 7;
		int pixelSize 		= 100;
		long start, stop, stop2;
		start = System.nanoTime();	

		stop = System.nanoTime();

		Gauss2Dfit fit 		= new Gauss2Dfit(testdataFirst, inpwidth);
		Particle First = fit.optimizeAdaptive(frame, channel, center, pixelSize);
		Gauss2Dfit fit2 		= new Gauss2Dfit(testdataSecond, inpwidth);
		Particle Second = fit2.optimizeAdaptive(frame, channel, center, pixelSize);
		//fit.optimizeAdatpive(frame,channel, center, pixelSize);

		center[0] = 4;
		//Gauss2Dfit fitSecond 		= new Gauss2Dfit(testdataSecond, inpwidth);

		stop2 = System.nanoTime();
		System.out.println((stop-start)/1000000 + " ms vs " + (stop2-stop)/1000000 + " ms");	
		//fitSecond.optimizeRsquare(frame,channel, center, pixelSize);

		//	Particle Second = fitSecond.optimizeRsquare(frame, channel, center, pixelSize);

		System.out.println(First.x + " x " + First.y);
		System.out.println(Second.x + " x " + Second.y);

	}



	public Particle optimizeAdaptive(int frame, int channel, int[] center, int pixelSize)
	{
		// starting guess.		
		double mx = 0; // moment in x (first order).
		double my = 0; // moment in y (first order).
		double m0 = 0; // 0 order moment.

		for (int i = 0; i < inputdata.length; i++)
		{
			int x = i % width;
			int y = i / width; 
			mx += x*inputdata[i];
			my += y*inputdata[i];
			m0 += inputdata[i];
		}
		double[] weightedCentroid = {mx/m0, my/m0};
		double[] P = {inputdata[width*(width-1)/2 + (width-1)/2], 	// Amplitude.
				weightedCentroid[0], 												// x center, weighted centroid.
				weightedCentroid[1], 												// y center, weighted centroid.
				(width-1)/4.0, 										// sigma x.
				(width-1)/4.0, 										// sigma y.
				0, 													// theta.
				0 													// offset.
		};		
		double[] bounds = {
				0.8*P[0],      1.2*P[0],   						// amplitude, should be close to center pixel value. Add +/-20 % of center pixel, not critical for performance.
				weightedCentroid[0]-1, weightedCentroid[0]+1, 	// x coordinate. Center has to be around the center pixel if gaussian distributed.
				weightedCentroid[1]-1, weightedCentroid[1]+1, 	// y coordinate. Center has to be around the center pixel if gaussian distributed.
				width/9.0,  	 width/2.0,   						// sigma x. Based on window size.
				width/9.0,  	 width/2.0,   						// sigma y. Based on window size.
				0, 	   Math.PI/4,   							// Theta. Any larger and the same result can be gained by swapping sigma x and y, symmetry yields only positive theta relevant.
				-P[0]*0.1,  	    P[0]*0.1};  				// offset, best estimate, not critical for performance.

		// steps is the most critical for processing time. Final step is 1/25th of these values. 
		double[] stepSize = {
				P[0]*.125,        // amplitude, make final step 0.5% of max signal.
				0.25*100/pixelSize,        // x step, final step = 0.2 nm.
				0.25*100/pixelSize,        // y step, final step = 0.2 nm.
				0.5*100/pixelSize,        // sigma x step, final step = 0.4 nm.
				0.5*100/pixelSize,        // sigma y step, final step = 0.4 nm.
				0.19625,        // theta step, final step = 0.00785 radians. Start value == 25% of bounds.
				P[0]*.0125};       // offset, make final step 0.05% of signal.

		///////////////////////////////////////////////////////////////////
		//////////////////// intitate variables. //////////////////////////
		///////////////////////////////////////////////////////////////////
		Boolean optimize    = true;
		double z0 		  	= 0;
		double sigma_z 	  	= 0;
		int loopcounter     = 0;
		int xi              = 0;
		int yi              = 0;
		double residual     = 0;
		double Rsquare      = 1;
		double ThetaA       = 0;
		double ThetaB       = 0;
		double ThetaC       = 0;
		double inputRsquare = 0;
		double tempRsquare  = 0;
		double ampStep      = stepSize[0];
		double xStep        = stepSize[1];
		double yStep        = stepSize[2];
		double sigmaxStep   = stepSize[3];
		double sigmayStep   = stepSize[4];
		double thetaStep    = stepSize[5];                
		double offsetStep   = stepSize[6];
		double sigmax2      = P[3]*P[3];
		double sigmay2      = P[4]*P[4];
		double sigmax       = 0;
		double sigmay       = 0;
		double theta        = 0;
		double x            = 0;
		double y            = 0;
		int xyIndex         = 0;		
		int improvedStep 	= 0;

		///////////////////////////////////////////////////////////////////
		/////// optimize x, y, sigma x, sigma y and theta in parallel. /////
		///////////////////////////////////////////////////////////////////

		ThetaA = 1 / (2 * sigmax2);
		ThetaB = 0;
		ThetaC = 1 / (2 * sigmay2);
		for (xyIndex = 0; xyIndex < width * width; xyIndex++)
		{
			xi = xyIndex % width;
			yi = xyIndex / width;
			residual = P[0] * Math.exp(-(ThetaA * (xi - P[1]) * (xi - P[1]) -
					2 * ThetaB * (xi - P[1]) * (yi - P[2]) +
					ThetaC * (yi - P[2]) * (yi - P[2])
					)) + P[6] - inputdata[xyIndex];

			Rsquare += residual * residual;
		}
		Rsquare = (Rsquare / totalSumOfSquares);  // normalize.
		while (optimize) 
		{
			inputRsquare = Rsquare; // before loop.                         
			for (sigmax = P[3] - sigmaxStep; sigmax <= P[3] + sigmaxStep; sigmax += sigmaxStep)                        
			{
				sigmax2 = sigmax * sigmax; // calulating this at this point saves computation time.
				for (sigmay = P[4] - sigmayStep; sigmay <= P[4] + sigmayStep; sigmay += sigmayStep)                            
				{
					sigmay2 = sigmay * sigmay; // calulating this at this point saves computation time.
					if (sigmax != sigmay)
					{											
						for (theta = P[5] - thetaStep; theta <= P[5] + thetaStep; theta += thetaStep)
						{
							if (theta >= bounds[10] && theta <= bounds[11] && // Check that the current parameters are within the allowed range.
									sigmax >= bounds[6] && sigmax <= bounds[7] &&
									sigmay >= bounds[8] && sigmay <= bounds[9])
							{
								// calulating these at this point saves computation time.
								ThetaA = Math.cos(theta) * Math.cos(theta) / (2 * sigmax2) + Math.sin(theta) * Math.sin(theta) / (2 * sigmay2);
								ThetaB = -Math.sin(2 * theta) / (4 * sigmax2) + Math.sin(2 * theta) / (4 * sigmay2);
								ThetaC = Math.sin(theta) * Math.sin(theta) / (2 * sigmax2) + Math.cos(theta) * Math.cos(theta) / (2 * sigmay2);

								for (x = P[1] - xStep; x <= P[1] + xStep; x += xStep)
								{
									for (y = P[2] - yStep; y <= P[2] + yStep; y += yStep)
									{
										if (sigmax == P[3] && sigmay == P[4] && x == P[1] && y == P[2]) // no need to calculate center again.
										{
										} else{
											if (x >= bounds[2] && x <= bounds[3] && // Check that the current parameters are within the allowed range.
													y >= bounds[4] && y <= bounds[5])
											{
												// Calculate residual for this set of parameters.
												tempRsquare = 0; // reset.
												for (xyIndex = 0; xyIndex < width * width; xyIndex++)
												{
													xi = xyIndex % width;
													yi = xyIndex / width;
													residual = P[0] * Math.exp(-(ThetaA * (xi - x) * (xi - x) -
															2 * ThetaB * (xi - x) * (yi - y) +
															ThetaC * (yi - y) * (yi - y)
															)) - inputdata[xyIndex];
													
/*													residual = P[0] * Math.exp(-(ThetaA * (xi - x) * (xi - x) -
															2 * ThetaB * (xi - x) * (yi - y) +
															ThetaC * (yi - y) * (yi - y)
															)) + P[6] - inputdata[xyIndex];
*/
													tempRsquare += residual * residual;
																					
													
												}
												//System.out.println("count: " + count + "sigma: " + sigmax + " x " + sigmay + " theta: " + theta + " xy: " + x + " x " + y);
											//	System.out.println("sigma step " + sigmaxStep + " x " + sigmayStep + "xy steps: " + xStep + " x " + yStep + " theta: " + thetaStep);
											//	count++;
												tempRsquare = (tempRsquare / totalSumOfSquares);  // normalize.
												if (tempRsquare < 0.99 * Rsquare)                // If improved, update variables.
												{
													Rsquare = tempRsquare;
													P[1] = x;
													P[2] = y;
													P[3] = sigmax;
													P[4] = sigmay;
													P[5] = theta;

												} // update parameters
											}// bounds check.
										}
									} // y loop.
								} // x loop.
							} // Theta check.
						} //  theta loop.
					} else // if sigmax and sigmay are the same, theta = 0 as the gaussian is perfectly circular. 
					{
						//theta = 0;
						if (sigmax >= bounds[6] && sigmax <= bounds[7] && // Check that the current parameters are within the allowed range.
								sigmay >= bounds[8] && sigmay <= bounds[9])
						{
							// calulating these at this point saves computation time.
							ThetaA = 1 / (2 * sigmax2);
							ThetaB = 0;
							ThetaC = 1 / (2 * sigmay2);

							for (x = P[1] - xStep; x <= P[1] + xStep; x += xStep)
							{
								for (y = P[2] - yStep; y <= P[2] + yStep; y += yStep)
								{
									if (sigmax == P[3] && sigmay == P[4] && x == P[1] && y == P[2]) // no need to calculate center again.
									{
									} else{
									
										if (x >= bounds[2] && x <= bounds[3] && // Check that the current parameters are within the allowed range.
												y >= bounds[4] && y <= bounds[5])
										{
											// Calculate residual for this set of parameters.
											tempRsquare = 0; // reset.
											for (xyIndex = 0; xyIndex < width * width; xyIndex++)
											{
												xi = xyIndex % width;
												yi = xyIndex / width;
												residual = P[0] * Math.exp(-(ThetaA * (xi - x) * (xi - x) -
														2 * ThetaB * (xi - x) * (yi - y) +
														ThetaC * (yi - y) * (yi - y)
														)) - inputdata[xyIndex];

/*												residual = P[0] * Math.exp(-(ThetaA * (xi - x) * (xi - x) -
														2 * ThetaB * (xi - x) * (yi - y) +
														ThetaC * (yi - y) * (yi - y)
														)) + P[6] - inputdata[xyIndex];
*/
												tempRsquare += residual * residual;
												
											}
											//System.out.println("count: " + count + "sigma: " + sigmax + " x " + sigmay + " theta: " + theta + " xy: " + x + " x " + y);
										//	System.out.println("sigma step " + sigmaxStep + " x " + sigmayStep + "xy steps: " + xStep + " x " + yStep);
										//	count++;
											tempRsquare = (tempRsquare / totalSumOfSquares);  // normalize.
											if (tempRsquare < 0.99	 * Rsquare)                // If improved, update variables.
											{
												Rsquare = tempRsquare;
												P[1] = x;
												P[2] = y;
												P[3] = sigmax;
												P[4] = sigmay;
										//		P[5] = theta;

											} // update parameters
										}// bounds check.
									}
								} // y loop.
							} // x loop.
						} // Theta check.
					} // sigma x and sigma y limit check.
				} // sigma y loop.
			} // sigmax loop.
			loopcounter++;
			if (inputRsquare == Rsquare) // if no improvement was made.
			{
				if (improvedStep < 3) // if stepsize has not been decreased 6 times already. Final stepsize = 1/128th of start.
				{
					xStep           = xStep         / 5;
					yStep           = yStep         / 5;
					sigmaxStep      = sigmaxStep    / 5;
					sigmayStep      = sigmayStep    / 5;
					thetaStep       = thetaStep     / 5;
					improvedStep++;					
				}
				else
					optimize = false; // exit.
			}
			if (loopcounter > 500) // exit.
				optimize = false;
		} // optimize while loop.
//System.out.println(count);
		/////////////////////////////////////////////////////////////////////////////
		////// optimize  amplitude and offset. Only used for photon estimate. ///////
		/////////////////////////////////////////////////////////////////////////////


		// no need to recalculate these for offset and amplitude:
		ThetaA 		= Math.cos(P[5]) * Math.cos(P[5]) / (2 * P[3] * P[3]) + Math.sin(P[5]) * Math.sin(P[5]) / (2 * P[4] * P[4]);
		ThetaB      = -Math.sin(2 * P[5]) / (4 * P[3] * P[3]) + Math.sin(2 * P[5]) / (4 * P[4] * P[4]);
		ThetaC      = Math.sin(P[5]) * Math.sin(P[5]) / (2 * P[3] * P[3]) + Math.cos(P[5]) * Math.cos(P[5]) / (2 * P[4] * P[4]);
		optimize    = true; // reset.	
		loopcounter = 0; // reset.
		improvedStep = 0; // reset.
		while (optimize) // optimize amplitude and offset.
		{
			inputRsquare = Rsquare; // before loop.
			for (double amp = P[0] - ampStep; amp <= P[0] + ampStep; amp = amp + ampStep)
			{
				for (double offset = P[6] - offsetStep; offset <= P[6] + offsetStep; offset = offset + offsetStep)
				{
					if (amp == P[0] && offset == P[6]) // no need to rerun the center calculation. 
					{
					}else {
						tempRsquare = 0;
						if (amp > bounds[0] && amp < bounds[1] &&
								offset > bounds[12] && offset < bounds[13])
						{
							for (xyIndex = 0; xyIndex < width * width; xyIndex++)
							{
								xi = xyIndex % width;
								yi = xyIndex / width;
								residual = amp * Math.exp(-(ThetaA * (xi - P[1]) * (xi - P[1]) -
										2 * ThetaB * (xi - P[1]) * (yi - P[2]) +
										ThetaC * (yi - P[2]) * (yi - P[2])
										)) + offset - inputdata[xyIndex];

								tempRsquare += residual * residual;
							}
							tempRsquare = (tempRsquare / totalSumOfSquares);  // normalize.
							if (tempRsquare < 0.99 * Rsquare)// If improved, update variables.
							{
								Rsquare     = tempRsquare;
								P[0]     = amp;
								P[6] = offset;


							} // update parameters
						}// Check if within bounds.
					}
				} // offset loop.
			} // amplitude loop.

			loopcounter++;
			if (inputRsquare == Rsquare) // if no improvement was made.
			{
				if (improvedStep < 3) // if stepsize has not been decreased 6 times already. Final stepsize = 1/64 of start.
				{
					ampStep = ampStep / 2;
					offsetStep = offsetStep / 2;
					improvedStep++;
				}
				else
					optimize = false; // exit.
			}
			if (loopcounter > 50) // exit.
				optimize = false;
		}// optimize while loop

		///////////////////////////////////////////////////////////////////
		///////////////////////// Final output: ///////////////////////////
		///////////////////////////////////////////////////////////////////

		Particle Localized 		= new Particle(); // Create new Particle and include fit parameters.
		Localized.include 		= 1;
		Localized.channel 		= channel;
		Localized.frame   		= frame;
		Localized.r_square 		= 1-Rsquare;
		Localized.x				= pixelSize*(P[1] + center[0] - Math.round((width)/2));
		Localized.y				= pixelSize*(P[2] + center[1] - Math.round((width)/2));
		Localized.z				= pixelSize*z0;
		Localized.sigma_x		= pixelSize*P[3];
		Localized.sigma_y		= pixelSize*P[4];
		Localized.sigma_z		= pixelSize*sigma_z;
		Localized.photons		= (int) Eval_photons(P);
		Localized.precision_x 	= Localized.sigma_x/Math.sqrt(Localized.photons);
		Localized.precision_y 	= Localized.sigma_y/Math.sqrt(Localized.photons);
		Localized.precision_z 	= Localized.sigma_z/Math.sqrt(Localized.photons);		

		return Localized;
	}
}