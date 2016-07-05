
import java.util.ArrayList;

import org.apache.commons.math3.fitting.leastsquares.LeastSquaresOptimizer.Optimum;

import ij.process.ImageProcessor;


/*
 * ParticleFitter.Fitter returns an arraylist of particles that has been fitted. Input is a ImageProcessor for a frame of interest, an int array list of center coordinates of interest,
 * Window width of square centered on these coordinates in pixels and frame numer.
 */

public class ParticleFitter {

	public static ArrayList<Particle> Fitter(float[][] InpArray, ArrayList<int[]> Center, int Window, int Frame, double Channel, int pixelSize){				
		double z0 			= 0; 													// Fitter does not support 3D fitting at this time.
		double sigma_z  	= 0;													// Fitter does not support 3D fitting at this time.
		double precision_z 	= 0;													// Fitter does not support 3D fitting at this time.
		int Offcenter = Math.round((Window-1)/2) +1;								// How far from 0 the center pixel is. Used to modify output to match the underlying image.
		ArrayList<Particle> Results = new ArrayList<Particle>(); 					// Create output arraylist.
		int CenterArray = Window*(Window-1)/2 + (Window-1)/2;
		for (int Event = 0; Event < Center.size(); Event++){ 						// Pull out data based on each entry into Center.
			double[] dataFit = new double[Window*Window];							// Container for data to be fitted.
			int[] Coord = Center.get(Event);										// X and Y coordiantes for center pixels to be fitted.
			int count = 0;	
			double ExpectedValue = 0; 												// Total value within the region, to be compared to calculated gaussian.
			for (int i = Coord[0]-(Window-1)/2; i< Coord[0] + (Window-1)/2 + 1; i++){ 	// Get all pixels for the region.
				for (int j = Coord[1]-(Window-1)/2; j< Coord[1] + (Window-1)/2 + 1; j++){					
					dataFit[count] = InpArray[i][j];								// Pull out data.					
					ExpectedValue += dataFit[count];								// Add data to total.
					count++;
				}
			}

			/*
			 * Fit pulled out data.
			 */			
			double[] startParameters = {
					dataFit[CenterArray], 						// Amplitude.
					(Window-1)/2,								// X center.
					(Window-1)/2,								// Y center.
					Window/4.0,									// Sigma x.
					Window/4.0,									// Sigma y.
					0,											// Offset.
					0											// Theta, angle in radians away from y axis.
			};
			Eval tdgp = new Eval(dataFit, startParameters, Window, new int[] {1000,100}); // Create fit object.

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
						pixelSize*(optimalValues[3]/Math.sqrt(photons)),	// precision of fit for y coordinate.
						precision_z,										// Default value.
						ChiSquare, 											// Goodness of fit.
						photons, 											// Photon count based on gaussian fit.
						1));												// Include particle
			}

			catch (Exception e) {
				//	System.out.println(e.toString());
			}

		}

		return Results;
	}

	public static ArrayList<Particle> Fitter(ImageProcessor IP, ArrayList<int[]> Center, int Window, int Frame, double Channel, int pixelSize){
		// Temp input until fitting sorted.		

		double z0 			= 0; 													// Fitter does not support 3D fitting at this time.
		double sigma_z  	= 0;													// Fitter does not support 3D fitting at this time.
		double precision_z 	= 0;													// Fitter does not support 3D fitting at this time.
		int Offcenter 		= Math.round((Window-1)/2) +1;							// How far from 0 the center pixel is. Used to modify output to match the underlying image.
		ArrayList<Particle> Results = new ArrayList<Particle>(); 					// Create output arraylist.
		int CenterArray = Window*(Window-1)/2 + (Window-1)/2;
		for (int Event = 0; Event < Center.size(); Event++){ 						// Pull out data based on each entry into Center.
			double[] dataFit = new double[Window*Window];							// Container for data to be fitted.
			int[] Coord = Center.get(Event);										// X and Y coordinates for center pixels to be fitted.
			int count = 0;	
			double ExpectedValue = 0; 												// Total value within the region, to be compared to calculated gaussian.
			for (int i = Coord[0]-(Window-1)/2; i< Coord[0] + (Window-1)/2 + 1; i++){ 	// Get all pixels for the region.
				for (int j = Coord[1]-(Window-1)/2; j< Coord[1] + (Window-1)/2 + 1; j++){
					dataFit[count] = IP.getf(i, j);	
					// Pull out data.				
					ExpectedValue += dataFit[count];								// Add data to total.
					count++;
				}
			}

			/*
			 * Fit pulled out data.
			 */			
			double[] startParameters = {
					dataFit[CenterArray], 						// Amplitude.
					(Window-1)/2,								// X center.
					(Window-1)/2,								// Y center.
					Window/3.0,									// Sigma x.
					Window/3.0,									// Sigma y.
					0,											// Offset.
					0											// Theta, angle in radians away from y axis.
			};
			Eval tdgp = new Eval(dataFit, startParameters, Window, new int[] {1000,100}); // Create fit object.

			try{
				//do LevenbergMarquardt optimization and get optimized parameters
				Optimum opt = tdgp.fit2dGauss();
				final double[] optimalValues = opt.getPoint().toArray();
				optimalValues[3] = Math.abs(optimalValues[3]);
				optimalValues[4] = Math.abs(optimalValues[4]);
				double photons = Evaluate(optimalValues,dataFit.length,Window);
				double ChiSquare = (photons-ExpectedValue)*(photons-ExpectedValue)/ExpectedValue;
				if (optimalValues[1] >0 && optimalValues[1] < Window && // If nothing strange happened with the fitting.
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
							pixelSize*(optimalValues[3]/Math.sqrt(photons)),	// precision of fit for y coordinate.
							precision_z,										// Default value.
							ChiSquare, 											// Goodness of fit.
							photons,											// Photon count based on gaussian fit.
							1));												// Include particle
				}
			}
			catch (Exception e) {
				//System.out.println(e.toString());
			}

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
