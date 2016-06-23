package sm_localizer;

import java.util.ArrayList;

import org.apache.commons.math3.fitting.leastsquares.LeastSquaresOptimizer.Optimum;

import ij.process.ImageProcessor;


/*
 * ParticleFitter.Fitter returns an arraylist of particles that has been fitted. Input is a ImageProcessor for a frame of interest, an int array list of center coordinates of interest,
 * Window width of square centered on these coordinates in pixels and frame numer.
 */

public class ParticleFitter {
	public static ArrayList<Particle> Fitter(ImageProcessor IP, ArrayList<int[]> Center, int Window, int Frame){
		ArrayList<Particle> Results = new ArrayList<Particle>(); 					// Create output arraylist.
		for (int Event = 0; Event < Center.size(); Event++){ 						// Pull out data based on each entry into Center.
			double[] dataFit = new double[Window*Window];							// Container for data to be fitted.
			int[] Coord = Center.get(Event);										// X and Y coordiantes for center pixels to be fitted.
			int count = 0;	
			double ExpectedValue = 0; 												// Total value wihtin the region, to be compared to calculated gaussian.
			for (int i = Coord[0]-(Window-1)/2; i< Coord[0] + (Window-1)/2; i++){ 	// Get all pixels for the region.
				for (int j = Coord[1]-(Window-1)/2; j< Coord[1] + (Window-1)/2; j++){
					dataFit[count] = IP.getPixel(i, j);								// Pull out data.
					ExpectedValue += dataFit[count];								// Add data to total.
					count++;
				}
			}

			/*
			 * Fit pulled out data.
			 */

			double[] startParameters = {
					dataFit[IP.getPixel(Coord[0],Coord[1])], 	// Amplitude.
					(Window-1)/2 + 1,							// X center.
					(Window-1)/2 + 1,							// Y center.
					Window/4.0,									// Sigma x.
					Window/4.0,									// Sigma y.
					0,											// Offset..
					0											// Theta, angle in radians away from y axis.
			};
			Eval tdgp = new Eval(dataFit, startParameters, Window, new int[] {1000,100}); // Create fit object.

			try{
				//do LevenbergMarquardt optimization and get optimized parameters
				Optimum opt = tdgp.fit2dGauss();
				final double[] optimalValues = opt.getPoint().toArray();
				double photons = Evaluate(optimalValues,dataFit.length,Window);
				double ChiSquare = (photons-ExpectedValue)*(photons-ExpectedValue)/ExpectedValue;

				
				Results.add( new Particle(					// Add results to output list.
						optimalValues[1], 					// Fitted x coordinate.
						optimalValues[2], 					// Fitted y coordinate.
						Frame, 								// frame that the particle was identified.
						optimalValues[3], 					// fitted sigma in x direction.
						optimalValues[4], 					// fitted sigma in y direction.
						optimalValues[3]/Math.sqrt(photons),// precision of fit for x coordinate.
						optimalValues[3]/Math.sqrt(photons),// precision of fit for y coordinate.
						ChiSquare, 							// Goodness of fit.
						photons));							// Photon count based on gaussian fit.
			}

			catch (Exception e) {
				System.out.println(e.toString());
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
