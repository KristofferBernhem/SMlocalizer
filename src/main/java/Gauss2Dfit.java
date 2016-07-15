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
	double[] inputdata;
	int width;
	int size;


	public Gauss2Dfit(double[] inputdata, int window){ // Setup
		this.inputdata = inputdata;
		this.width = window;
		this.size = inputdata.length;
	}

	public double Eval(double[] P){ // Evaluate gaussian with parameters from P:
		/*
		 * P[0]: Amplitude
		 * P[1]: x0
		 * P[2]: y0
		 * P[3]: sigma x
		 * P[4]: sigma y
		 * P[5]: theta
		 * P[6]: offset 
		 */
		double ChiSquare = 0;
		double ThetaA = Math.cos(P[5])*Math.cos(P[5])/(2*P[3]*P[3]) + Math.sin(P[5])*Math.sin(P[5])/(2*P[4]*P[4]); 
		double ThetaB = -Math.sin(2*P[5])/(4*P[3]*P[3]) + Math.sin(2*P[5])/(4*P[4]*P[4]); 
		double ThetaC = Math.sin(P[5])*Math.sin(P[5])/(2*P[3]*P[3]) + Math.cos(P[5])*Math.cos(P[5])/(2*P[4]*P[4]);

//				double SigmaX2 = 2*P[3]*P[3];
//				double SigmaY2 = 2*P[4]*P[4];
		for (int i = 0; i < size; i++){
			int xi = i % width;
			int yi = i / width;	
			double residual = P[0]*Math.exp(-(ThetaA*(xi - P[1])*(xi - P[1]) - 
					2*ThetaB*(xi - P[1])*(yi - P[2]) +
					ThetaC*(yi - P[2])*(yi - P[2])
					)) + P[6] - inputdata[i];

			//			double xprime = (xi - P[1])*Math.cos(P[5]) - (yi - P[2])*Math.sin(P[5]);
				//		double yprime = (xi - P[1])*Math.sin(P[5]) + (yi - P[2])*Math.cos(P[5]);
				//double residual = P[0]*Math.exp(-(xprime*xprime/SigmaX2 + yprime*yprime/SigmaY2)) + P[6] - inputdata[i];
			ChiSquare += residual*residual/(inputdata[i]+1); // handle case where inputdata[i] == 0;
		}

		return ChiSquare;
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

		//		double SigmaX2 = 2*P[3]*P[3];
		//		double SigmaY2 = 2*P[4]*P[4];
		double photons = 0;
		for (int i = 0; i < size; i++){
			int xi = i % width;
			int yi = i / width;	
			photons += P[0]*Math.exp(-(ThetaA*(xi - P[1])*(xi - P[1]) - 
					2*ThetaB*(xi - P[1])*(yi - P[2]) +
					ThetaC*(yi - P[2])*(yi - P[2])
					)) + P[6];

			//			double xprime = (xi - P[1])*Math.cos(P[5]) - (yi - P[2])*Math.sin(P[5]);
			//			double yprime = (xi - P[1])*Math.sin(P[5]) + (yi - P[2])*Math.cos(P[5]);
			//	double residual[i]  = P[0]*Math.exp(-(xprime*xprime/SigmaX2 + yprime*yprime/SigmaY2)) + P[6];			
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
		
		
		/*double[] inputdata = {
				0  ,12 ,25 ,12 ,0  ,
				12 ,89 ,153,89 ,12 ,
				25 ,153,255,153,25 ,
				12 ,89 ,153,89 ,12 ,
				0  ,12 ,25 ,12 ,0  ,
		};	
		int frame 			= 1;		// include in optimize call.
		int channel 		= 1;		// include in optimize call.
		int[] center 		= {3,3};	// include in optimize call.	
		int inpwidth 		= 5;
		int pixelSize 		= 100;
		Gauss2Dfit fit 		= new Gauss2Dfit(inputdata, inpwidth);		
		Particle Localized 	= fit.optimize(frame,channel, center, pixelSize);		*/
	}
	
	
	
	public Particle optimize(int frame, int channel, int[] center, int pixelSize){	
		double z0 = 0;
		double sigma_z = 0;			
		
		double[] P = {inputdata[width*(width-1)/2 + (width-1)/2],
				(width-1)/2,
				(width-1)/2,
				(width-1)/4,
				(width-1)/4,
				0,
				0
		};

		double[] lb = {0.7*P[0],
				0,
				0,
				width/7,
				width/7,
				-Math.PI/2,
				-P[0]
		};
		double[] ub = {1.5*P[0],
				width,
				width,
				width/2,
				width/2,
				Math.PI/2,
				P[0]
		};

//		long start = System.nanoTime();
		double[] stepfraction = new double[7];
		stepfraction[0] = (ub[0] - lb[0])/20;			// Amplitude.
		stepfraction[1] = (ub[1] - lb[1])/20;			// x.
		stepfraction[2] = (ub[2] - lb[2])/20;			// y.		
		stepfraction[3] = (ub[3] - lb[3])/20;			// sigma x.
		stepfraction[4] = (ub[4] - lb[4])/20;			// sigma y.
		stepfraction[5] = (ub[5] - lb[5])/10;			// theta.
		stepfraction[6] = (ub[6] - lb[6])/100;			// offset.
		int iterations	 		= 0;					// Loopcounter	
		int completeCirc 		= 0;					// Keeps track of complete loops over all parameters. 
		int currentP 			= 0; 					// Which parameter from parameterOrder to currently modify.
		int[] Porder 			= {3,4,1,2,5,6,0};  	// sigma, x-y, theta, offset,amp
		double Order 			= 1; 					// +/- 1, direction of change to parameter.
		double lastRoundChi 	= Eval(P);				// Keep track if improvements are being made, start with startparameters resulting value.
		double chisquare 		= lastRoundChi; 		// Input;


		// Logical loop for LS fitting.
		while(iterations < 10000 && completeCirc<100){ 														// Max loop counter and max complete turns of all parameters.
			P[Porder[currentP]] += Order*stepfraction[Porder[currentP]]; 									// Add (or subtract depending on value of Order) one step according to stepfraction for the current parameter.
			if(P[Porder[currentP]] > lb[Porder[currentP]] && 												// Check that the new value falls within parameter limits.
					P[Porder[currentP]] < ub[Porder[currentP]]){											// Check that the new value falls within parameter limits.				
				double newChi = Eval(P); 																	// Fit with new parameter.
				if (newChi < chisquare){ 																	// If fit was improved, keep on.
					chisquare = newChi;																		// Update chisquare
				}else{																						// If the new fit was not an improvement.
					if (Order == 1){																		// If we were increasing the current parameters. 
						Order = -1; 																		// Step in other direction.						
						P[Porder[currentP]] += Order*stepfraction[Porder[currentP]]; 						// Return parameter to value from last loop.
					}else{																					// If we've already stepped both up and down to find an optimum.						
						P[Porder[currentP]] -= Order*stepfraction[Porder[currentP]]; 						// Return parameter to value from last loop.
						currentP++;																			// Step to the next parameter.
						Order = 1;																			// Start by stepping up.
						if ( currentP == 7){																// If we've cycled throguh all parameters.
							completeCirc++;																	// Update counter.
							currentP = 0;																	// Reset parameter.
							if(chisquare == lastRoundChi){													// If we did not improve chisquare at all compared to last complete turn.
								completeCirc = 1000;														// Push counter outside loop check, exiting.
							}else{																			// If we did improve upon the fit this round, keep going.
								lastRoundChi = chisquare;													// Update tracker of this rounds best estimate of chisquare.
							}								
						}

					}						
				}
			}else{																							// If we stepped outside the parameter limit.
				if (Order == 1){																			// If we were stepping up.
					Order = -1; 																			// Step in other direction.									
					P[Porder[currentP]] += Order*stepfraction[Porder[currentP]]; 							// Reset parameter value to last iteration.
				}else{																						// If we've already stepped both up and down to find an optimum.	
					P[Porder[currentP]] -= Order*stepfraction[Porder[currentP]]; 							// Return parameter to value from last loop.				
					currentP++;																				// Step to the next parameter.
					Order = 1;																				// Start by stepping up.
					if ( currentP == 7){																	// If we've cycled throguh all parameters.
						completeCirc++;																		// Update counter.
						currentP = 0;																		// Reset parameter.
						if(chisquare == lastRoundChi){														// If we did not improve chisquare at all compared to last complete turn.
							completeCirc = 1000;															// Push counter outside loop check, exiting.
						}else{																				// If we did improve upon the fit this round, keep going.
							lastRoundChi = chisquare;														// Update tracker of this rounds best estimate of chisquare.
						}
					}

				}
			}			
			iterations++;																					// Update loop counter.
		} // End loop
	//	int lastIt = iterations - 1;
		iterations 		= 0; 																				// Reset.
		currentP 		= 0; 																				// Reset.
		Order	 		= 1; 																				// Reset.
		completeCirc 	= 0; 																				// Reset.
		// Finer stepsize.
		stepfraction[0] = (ub[0] - lb[0])/100;			// Amplitude.
		stepfraction[1] = (ub[1] - lb[1])/500;			// x.
		stepfraction[2] = (ub[2] - lb[2])/500;			// y.		
		stepfraction[3] = (ub[3] - lb[3])/200;			// sigma x.
		stepfraction[4] = (ub[4] - lb[4])/200;			// sigma y.
		stepfraction[5] = (ub[5] - lb[5])/100;			// theta.
		stepfraction[6] = (ub[6] - lb[6])/1000;			// offset.


		// Logical loop for LS fitting.
		while(iterations < 10000 && completeCirc<100){ 														// Max loop counter and max complete turns of all parameters.
			P[Porder[currentP]] += Order*stepfraction[Porder[currentP]]; 									// Add (or subtract depending on value of Order) one step according to stepfraction for the current parameter.
			if(P[Porder[currentP]] > lb[Porder[currentP]] && 												// Check that the new value falls within parameter limits.
					P[Porder[currentP]] < ub[Porder[currentP]]){											// Check that the new value falls within parameter limits.				
				double newChi = Eval(P); 																	// Fit with new parameter.
				if (newChi < chisquare){ 																	// If fit was improved, keep on.
					chisquare = newChi;																		// Update chisquare
				}else{																						// If the new fit was not an improvement.
					if (Order == 1){																		// If we were increasing the current parameters. 
						Order = -1; 																		// Step in other direction.						
						P[Porder[currentP]] += Order*stepfraction[Porder[currentP]]; 						// Return parameter to value from last loop.
					}else{																					// If we've already stepped both up and down to find an optimum.						
						P[Porder[currentP]] -= Order*stepfraction[Porder[currentP]]; 						// Return parameter to value from last loop.
						currentP++;																			// Step to the next parameter.
						Order = 1;																			// Start by stepping up.
						if ( currentP == 7){																// If we've cycled throguh all parameters.
							completeCirc++;																	// Update counter.
							currentP = 0;																	// Reset parameter.
							if(chisquare == lastRoundChi){													// If we did not improve chisquare at all compared to last complete turn.
								completeCirc = 1000;														// Push counter outside loop check, exiting.
							}else{																			// If we did improve upon the fit this round, keep going.
								lastRoundChi = chisquare;													// Update tracker of this rounds best estimate of chisquare.
							}								
						}

					}						
				}
			}else{																							// If we stepped outside the parameter limit.
				if (Order == 1){																			// If we were stepping up.
					Order = -1; 																			// Step in other direction.									
					P[Porder[currentP]] += Order*stepfraction[Porder[currentP]];			 				// Reset parameter value to last iteration.
				}else{																						// If we've already stepped both up and down to find an optimum.	
					P[Porder[currentP]] -= Order*stepfraction[Porder[currentP]]; 							// Return parameter to value from last loop.				
					currentP++;																				// Step to the next parameter.
					Order = 1;																				// Start by stepping up.
					if ( currentP == 7){																	// If we've cycled throguh all parameters.
						completeCirc++;																		// Update counter.
						currentP = 0;																		// Reset parameter.
						if(chisquare == lastRoundChi){														// If we did not improve chisquare at all compared to last complete turn.
							completeCirc = 1000;															// Push counter outside loop check, exiting.
						}else{																				// If we did improve upon the fit this round, keep going.
							lastRoundChi = chisquare;														// Update tracker of this rounds best estimate of chisquare.
						}
					}

				}
			}			
			iterations++;																					// Update loop counter.
		} // End loop



		chisquare = Eval(P);
/*		long stop = System.nanoTime();
	System.out.println("P[0]: " + P[0] +"\n"+ 
				"P[1]: " + P[1] +"\n"+
				"P[2]: " + P[2] +"\n"+
				"P[3]: " + P[3] +"\n"+
				"P[4]: " + P[4] +"\n"+
				"P[5]: " + P[5] +"\n"+
				"P[6]: " + P[6] +"\n" + 
				"in " + (lastIt + iterations) + " iterations" + "\n" + 
				"in " + (stop-start)/1000000.0 + " ms");
/*		double sum=0;
		for(int i = 0; i< inputdata.length; i++){
			sum += inputdata[i];
		}*/
		//	System.out.println(sum + " vs " + Eval(startparameters)/sum);
		Particle Localized 		= new Particle(); // Create new Particle and include fit parameters.
		Localized.include 		= 1;
		Localized.channel 		= channel;
		Localized.frame   		= frame;
		Localized.chi_square 	= chisquare;
		Localized.x				= pixelSize*(P[1] + center[0] - Math.round((width-1)/2));
		Localized.y				= pixelSize*(P[2] + center[1] - Math.round((width-1)/2));
		Localized.z				= pixelSize*z0;
		Localized.sigma_x		= pixelSize*P[3];
		Localized.sigma_y		= pixelSize*P[4];
		Localized.sigma_z		= pixelSize*sigma_z;
		Localized.photons		= Eval_photons(P);
		Localized.precision_x 	= Localized.sigma_x/Localized.photons;
		Localized.precision_y 	= Localized.sigma_y/Localized.photons;
		Localized.precision_z 	= Localized.sigma_z/Localized.photons;
			
		return Localized; // Return fit.
	} // optimize.

}