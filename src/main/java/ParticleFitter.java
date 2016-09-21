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

import java.util.ArrayList;

import org.apache.commons.math3.fitting.leastsquares.LeastSquaresOptimizer.Optimum;

//import org.apache.commons.math3.fitting.leastsquares.LeastSquaresOptimizer.Optimum;
//import org.apache.commons.math3.fitting.leastsquares.LeastSquaresOptimizer.Optimum;

import ij.process.ImageProcessor;


/*
 * ParticleFitter.Fitter returns an arraylist of particles that has been fitted. Input is a ImageProcessor for a frame of interest, an int array list of center coordinates of interest,
 * Window width of square centered on these coordinates in pixels and frame number.
 */

public class ParticleFitter {

	public static ArrayList<Particle> Fitter(float[][] InpArray, ArrayList<int[]> Center, int Window, int Frame, int Channel, int pixelSize){				
		//		double z0 			= 0; 													// Fitter does not support 3D fitting at this time.
		//		double sigma_z  	= 0;													// Fitter does not support 3D fitting at this time.
		//		double precision_z 	= 0;													// Fitter does not support 3D fitting at this time.
		//		int Offcenter = Math.round((Window-1)/2) +1;								// How far from 0 the center pixel is. Used to modify output to match the underlying image.
		ArrayList<Particle> Results = new ArrayList<Particle>(); 					// Create output arraylist.
		//	int CenterArray = Window*(Window-1)/2 + (Window-1)/2;				
		for (int Event = 0; Event < Center.size(); Event++){ 						// Pull out data based on each entry into Center.
			int[] dataFit = new int[Window*Window];							// Container for data to be fitted.
			int[] Coord = Center.get(Event);										// X and Y coordiantes for center pixels to be fitted.			
			int count = 0;	
			//			double ExpectedValue = 0; 												// Total value within the region, to be compared to calculated gaussian.

			for (int i = Coord[0]-(Window-1)/2; i< Coord[0] + (Window-1)/2 + 1; i++){ 	// Get all pixels for the region.
				for (int j = Coord[1]-(Window-1)/2; j< Coord[1] + (Window-1)/2 + 1; j++){					
					dataFit[count] = (int) InpArray[i][j];								// Pull out data.					
					//		ExpectedValue += dataFit[count];								// Add data to total.
					count++;
				}
			}

			// Fit pulled out data.

			/*			double[] startParameters = {
					dataFit[CenterArray], 						// Amplitude.
					(Window-1)/2,								// X center.
					(Window-1)/2,								// Y center.
					Window/3.0,									// Sigma x.
					Window/3.0,									// Sigma y.
					0,											// Offset.
					0											// Theta, angle in radians away from y axis.
			};*/
			
			Gauss2Dfit gfit = new Gauss2Dfit(dataFit,Window);
			Results.add(gfit.optimizeAdaptive(Frame, Channel, Coord, pixelSize));
			/*
			Eval tdgp = new Eval(dataFit, startParameters, Window, new int[] {10000,100}); // Create fit object.

			try{
				//do LevenbergMarquardt optimization and get optimized parameters
				Optimum opt = tdgp.fit2dGauss();
				final double[] optimalValues = opt.getPoint().toArray();
				optimalValues[3] = Math.abs(optimalValues[3]);
				optimalValues[4] = Math.abs(optimalValues[4]);
				double photons = Evaluate(optimalValues,dataFit.length,Window);
				double ChiSquare = (photons-ExpectedValue)*(photons-ExpectedValue)/ExpectedValue;

				Results.add( new Particle(									// Add results to output list.
						pixelSize*(optimalValues[1] + Coord[0] - Offcenter),// Fitted x coordinate.
						pixelSize*(optimalValues[2] + Coord[1] - Offcenter),// Fitted y coordinate.
						z0,													// Default value.
						Frame, 												// frame that the particle was identified.
						Channel,											// Currently default value.
						pixelSize*(optimalValues[3]), 						// fitted sigma in x direction.
						pixelSize*(optimalValues[4]), 						// fitted sigma in y direction.
						sigma_z,											// Default value.
						pixelSize*(optimalValues[3]/Math.sqrt(photons)),	// precision of fit for x coordinate.
						pixelSize*(optimalValues[4]/Math.sqrt(photons)),	// precision of fit for y coordinate.
						precision_z,										// Default value.
						ChiSquare, 											// Goodness of fit.
						photons, 											// Photon count based on gaussian fit.
						1));												// Include particle
			}

			catch (Exception e) {
				//	System.out.println(e.toString());
			}
			 */
		}

		return Results;
	}

	public static ArrayList<Particle> Fitter(ImageProcessor IP, ArrayList<int[]> Center, int Window, int Frame, double Channel, int pixelSize){
		// Temp input until fitting sorted.			
		/*		double z0 			= 0; 													// Fitter does not support 3D fitting at this time.
		double sigma_z  	= 0;													// Fitter does not support 3D fitting at this time.
		double precision_z 	= 0;													// Fitter does not support 3D fitting at this time.
		int Offcenter 		= Math.round((Window-1)/2);								// How far from 0 the center pixel is. Used to modify output to match the underlying image.
		 */
		ArrayList<Particle> Results = new ArrayList<Particle>(); 					// Create output arraylist.
		//		int CenterArray = Window*(Window-1)/2 + (Window-1)/2;
		for (int Event = 0; Event < Center.size(); Event++){ 						// Pull out data based on each entry into Center.
			int[] dataFit = new int[Window*Window];							// Container for data to be fitted.
			int[] Coord = Center.get(Event);										// X and Y coordinates for center pixels to be fitted.
	
			//	double ExpectedValue = 0; 												// Total value within the region, to be compared to calculated gaussian.
			for (int j = 0; j < Window*Window; j++)
			{
				int x =  Coord[0] - Math.round((Window)/2) +  (j % Window);
				int y =  Coord[1] - Math.round((Window)/2) +  (j / Window);
				dataFit[j] = (int) IP.getf(x,y);
			}


			/*
			 * Fit pulled out data.
			 */			
			/*		double[] startParameters = {
					dataFit[CenterArray], 						// Amplitude.
					(Window-1)/2+1,								// X center.
					(Window-1)/2+1,								// Y center.
					Window/3.0,									// Sigma x.
					Window/3.0,									// Sigma y.
					0,											// Offset.
					0											// Theta, angle in radians away from y axis.
			};*/
			Gauss2Dfit gfit = new Gauss2Dfit(dataFit,Window);
			Results.add(gfit.optimizeAdaptive(Frame, (int) Channel, Coord, pixelSize));

					
		/*	Eval tdgp = new Eval(dataFit, startParameters, Window, new int[] {1000,1000}); // Create fit object.

			try{									
				//do LevenbergMarquardt optimization and get optimized parameters
				Optimum opt = tdgp.fit2dGauss();				
				final double[] optimalValues = opt.getPoint().toArray();
				optimalValues[0] = Math.abs(optimalValues[0]);
				optimalValues[1] = Math.abs(optimalValues[1]);
				optimalValues[2] = Math.abs(optimalValues[2]);
				optimalValues[3] = Math.abs(optimalValues[3]);
				optimalValues[4] = Math.abs(optimalValues[4]);
				optimalValues[6] = Math.abs(optimalValues[6]);

				double photons = Evaluate(optimalValues,dataFit.length,Window);
				double ChiSquare = (photons-ExpectedValue)*(photons-ExpectedValue)/ExpectedValue;
				if (optimalValues[1] > 0 && optimalValues[1] < Window && // If nothing strange happened with the fitting.
						optimalValues[2] >0 && optimalValues[2] < Window){
					Results.add( new Particle(									// Add results to output list.
							pixelSize*(optimalValues[1] + Coord[0] - Offcenter),// Fitted x coordinate.
							pixelSize*(optimalValues[2] + Coord[1] - Offcenter),// Fitted y coordinate.
							z0,													// Default value.
							Frame, 												// frame that the particle was identified.
							Channel,											// Currently default value.
							pixelSize*(optimalValues[3]), 						// fitted sigma in x direction.
							pixelSize*(optimalValues[4]), 						// fitted sigma in y direction.
							sigma_z,											// Default value.
							pixelSize*(optimalValues[3]/Math.sqrt(photons)),	// precision of fit for x coordinate.
							pixelSize*(optimalValues[4]/Math.sqrt(photons)),	// precision of fit for y coordinate.
							precision_z,										// Default value.
							ChiSquare, 											// Goodness of fit.
							photons,											// Photon count based on gaussian fit.
							1));												// Include particle
				//	System.out.println("Coord: " + Coord[0] + "x" + Coord[1] + " fitted: " + (optimalValues[1]) + "x" +(optimalValues[2]) + " angle:" + optimalValues[6]);
				}
			}
			catch (Exception e) {
				//System.out.println(e.toString());
			}*/

		}

		return Results;
	}


	public static Particle Fitter(fitParameters fitThese){ // setup a single gaussian fit, return localized particle.
		double convergence	= 1E-8;	// stop optimizing once improvement is below this.
		int maxIteration 	= 1000;	// max number of iterations.
		GaussSolver Gsolver = new GaussSolver(
				fitThese.data, 		// data to be fitted.
				fitThese.windowWidth, // window used for data extraction.
				convergence,
				maxIteration, 
				fitThese.Center, 		// center coordianates for center pixel.
				fitThese.channel, 		// channel id.
				fitThese.pixelsize,		// pixelsize in nm.
				fitThese.frame,			// frame number.
				fitThese.totalGain);	// total gain, camera specific parameter giving relation between input photon to output pixel intensity.
		Particle Results 	= Gsolver.Fit();	// do fit.
		return Results;
	}
		public static Particle FitterLM(fitParameters fitThese){
		double[] startParameters = {
						26000,
						2.4,
						2.7,
						1.0,
						1.0,
						0,
						0
				};
				double[] data = new double[fitThese.data.length];
				for (int i = 0; i < fitThese.data.length; i++){
					data[i] = fitThese.data[i];
				}
				
				Eval tdgp = new Eval(data, startParameters, fitThese.windowWidth, new int[] {1000,1000}); // Create fit object.
				Particle Results = new Particle();	
		try{									
			//do LevenbergMarquardt optimization and get optimized parameters
			Optimum opt = tdgp.fit2dGauss();				
			final double[] optimalValues = opt.getPoint().toArray();
			optimalValues[0] = Math.abs(optimalValues[0]);
			optimalValues[1] = Math.abs(optimalValues[1]);
			optimalValues[2] = Math.abs(optimalValues[2]);
			optimalValues[3] = Math.abs(optimalValues[3]);
			optimalValues[4] = Math.abs(optimalValues[4]);
			optimalValues[6] = Math.abs(optimalValues[6]);
			Results.x = 100*(optimalValues[1] + fitThese.Center[0] - Math.round((fitThese.windowWidth)/2));
			Results.y = 100*(optimalValues[2] + fitThese.Center[1] - Math.round((fitThese.windowWidth)/2));
			//Results.frame = optimalValues[5];
			/*
			 * calculate Rsquare for comparison to adaptive method.
			 */
			/*
			double m0= 0;
			for (int i = 0; i < fitThese.data.length; i++) // get image moments.
			{
				m0 		+= data[i];
			}
			m0 /=  fitThese.data.length;// Mean value.		
			double totalSumOfSquares= 0;
			for (int i = 0; i < fitThese.data.length; i++){			// Calculate total sum of squares for R^2 calculations.
				totalSumOfSquares += (fitThese.data[i] -m0)*(fitThese.data[i] -m0);
			}
				
				double	ThetaA = Math.cos(optimalValues[6]) * Math.cos(optimalValues[6]) / (2 * optimalValues[3]*optimalValues[3]) + 
							Math.sin(optimalValues[6]) * Math.sin(optimalValues[6]) / (2 * optimalValues[4]*optimalValues[4]);
				double  ThetaB = -Math.sin(2 * optimalValues[5]) / (4 * optimalValues[3]*optimalValues[3]) + 
							Math.sin(2 * optimalValues[6]) / (4 * optimalValues[4]*optimalValues[4]);
				double	ThetaC = Math.sin(optimalValues[6]) * Math.sin(optimalValues[5]) / (2 * optimalValues[3]*optimalValues[3]) + 
							Math.cos(optimalValues[6]) * Math.cos(optimalValues[6]) / (2 * optimalValues[4]*optimalValues[4]);
				
				double	tempRsquare = 0; // reset.
				for (int xyIndex = 0; xyIndex < fitThese.windowWidth * fitThese.windowWidth; xyIndex++)
				{
					int xi = xyIndex % fitThese.windowWidth;
					int yi = xyIndex / fitThese.windowWidth;
					double residual = optimalValues[0] * Math.exp(-(ThetaA * (xi -  optimalValues[1]) * (xi -  optimalValues[1]) -
							2 * ThetaB * (xi -  optimalValues[1]) * (yi - optimalValues[2]) +
							ThetaC * (yi - optimalValues[2]) * (yi - optimalValues[2])
							)) + optimalValues[5] - fitThese.data[xyIndex];
					tempRsquare += residual * residual;
				}
				
				tempRsquare = (tempRsquare / totalSumOfSquares);  // normalize.
			System.out.println("LM: " +(1-tempRsquare));*/
	
		
			
		}
		catch (Exception e) {
		
		}
			

		return Results;
	}

	public static double Evaluate(double[] Parameters, int nPixels, int width) // Evaluate gaussian based on input values.
			throws IllegalArgumentException {
		double eval = 0;
		double SigmaX2 = 2*Parameters[3]*Parameters[3];
		double SigmaY2 = 2*Parameters[4]*Parameters[4];
		for (int i = 0; i < nPixels; i++){
			int xi = i % width;
			int yi = i / width;				
			double xprime = (xi - Parameters[1])*Math.cos(Parameters[6]) - (yi - Parameters[2])*Math.sin(Parameters[6]);
			double yprime = (xi - Parameters[1])*Math.sin(Parameters[6]) + (yi - Parameters[2])*Math.cos(Parameters[6]);
			eval += Parameters[0]*Math.exp(-(xprime*xprime/SigmaX2 + yprime*yprime/SigmaY2)) + Parameters[5];
		}
		return eval; // Return summed value of fitted gaussian.
	}
}
