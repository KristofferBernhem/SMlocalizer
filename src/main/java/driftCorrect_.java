import java.awt.Color;
import java.util.ArrayList;

import ij.gui.Plot;
import ij.plugin.PlugIn;

public class driftCorrect_ implements PlugIn {
	public void run(String arg0){

		/*
		 * User input that needs to fixed.
		 */

		double[] lb 				= {-250,						// Allowed lower range of x drift in nm, user input.
				-250,						// Allowed lower range of y drift in nm, user input.
				0,						// Allowed lower range of sigma_x in nm, user input.
				0,						// Allowed lower range of sigma_y in nm, user input.
				0,						// Allowed lower range of precision_x in nm, user input.
				0,						// Allowed lower range of precision_y in nm, user input.
				0,						// Allowed lower range of chi_square, user input.
				100							// Allowed lower range of photon count, user input.
		};  				
		double[] ub 				= {250,						// Allowed upper range of x drift in nm, user input.
				250,						// Allowed upper range of y drift in nm, user input.
				300,						// Allowed upper range of sigma_x in nm, user input.
				300,						// Allowed upper range of sigma_y in nm, user input.
				300,						// Allowed upper range of precision_x in nm, user input.
				300,						// Allowed upper range of precision_y in nm, user input.
				1.0,						// Allowed upper range of chi_square, user input.
				500000000					// Allowed upper range of photon count, user input.
		};  							
		double BinFrac				= 0.02;							// Fraction of total frames in each bin for drift corrrection. User input.
		int nParticles 				= 1000;							// Maximal number of particles to use for drift correction in each step, user input.
		int minNrParticles 			= 500;
		int[] stepSize 				= {5,5};						// Stepsize in nm, user input.
		
		
		ArrayList<Particle> locatedParticles = TableIO.Load(); // Get current table data.
		System.out.println("driftCorrect: " + locatedParticles.size());
		ArrayList<Particle> correctedResults = new ArrayList<Particle>(); // Output arraylist, will contain all particles that fit user input requirements after drift correction. 
		int[] lb_xy 		= {(int) lb[0],(int) lb[1]};	// Pull out lower boundry of x and y drift.
		int[] ub_xy 		= {(int) ub[0],(int) ub[1]};	// Pull out upper boundry of x and y drift.		
		int idx 			= locatedParticles.size() - 1;

		double frameBin = Math.round( 				// Bin size for drift correction based on total number of frames and user input fraction. 
				locatedParticles.get(idx).frame * 	// Last frame that was used.
				BinFrac);							// User input fraction.
		ArrayList<Particle> filteredResults =  new ArrayList<Particle>(); // output arraylist.
		int[] timeIndex = new int[(int) (Math.round(1.0/BinFrac)+1)];
		double[] timeIndexDouble = new double[(int) (Math.round(1.0/BinFrac)+1)];
		int count = 0;		

		// Check which particles are within user set parameters.
		for (int i = 0; i < locatedParticles.size(); i++){
			if (	locatedParticles.get(i).sigma_x > lb[2] && // Check that all parameters are within user defined limits.
					locatedParticles.get(i).sigma_x < ub[2] &&
					locatedParticles.get(i).sigma_y > lb[3] &&
					locatedParticles.get(i).sigma_y < ub[3] &&
					locatedParticles.get(i).precision_x > lb[4] &&
					locatedParticles.get(i).precision_x < ub[4] &&					
					locatedParticles.get(i).precision_y > lb[5] &&
					locatedParticles.get(i).precision_y < ub[5] &&
					locatedParticles.get(i).chi_square > lb[6] &&
					locatedParticles.get(i).chi_square < ub[6] &&						
					locatedParticles.get(i).photons > lb[7] &&
					locatedParticles.get(i).photons < ub[7]
					){
				filteredResults.add(locatedParticles.get(i)); 	// Add particles that match user set parameters.					

				if (filteredResults.get(filteredResults.size()-1).frame > frameBin*count){	// First time data from a new bin is added, register index.
					timeIndex[count] = filteredResults.size() - 1; // Get the index for the first entry with the new bin.
					timeIndexDouble[count] = filteredResults.get(filteredResults.size()-1).frame;
					count++;
				}
			}
		}

		timeIndex[timeIndex.length-1] =  filteredResults.size(); 									// Final entry.
		timeIndexDouble[timeIndex.length-1] = filteredResults.get(filteredResults.size()-1).frame; 	// Final entry.
		double[] lambdax = new double[(int) Math.round(1.0/BinFrac)];								// Holds drift estimate between all bins in x.
		double[] lambday = new double[(int) Math.round(1.0/BinFrac)];								// Holds drift estimate between all bins in y.
		lambdax[0] = 0;																				// Drift for first bin is 0.
		lambday[0] = 0;																				// Drift for first bin is 0.
		int maxTime =(int) timeIndexDouble[timeIndexDouble.length-1]; 								// Last frame included.
		double[][] lambda = new double[maxTime][2];													// Holds interpolated drift corrections in x and y.
		int okBins = 0;																				// If all bins are ok, this will still be 0.
		for (int i = 1; i < timeIndex.length ; i++){ 												// Loop over all bins.
			if ((timeIndex[i] - timeIndex[i-1])<minNrParticles){									// If the bin lacks enough points to meet user minimum criteria.
				okBins++;				
			}
		}
		if (okBins == 0){ 														// If all bins were ok.
			for (int i = 1; i < Math.round(1.0/BinFrac) ; i++){ 				// Loop over all bins.
				ArrayList<Particle> Data1 	= new ArrayList<Particle>(); 		// Target particles.			
				int addedFrames1 			= 0;								// Number of particles added to the bin.
				for (int j = timeIndex[i]; j < timeIndex[i+1];j++){
					if (addedFrames1 < nParticles &&
							filteredResults.get(j).frame < frameBin*(i+1)){
						Data1.add(filteredResults.get(j));
						addedFrames1++;
					}
				}			
				ArrayList<Particle> Data2 	= new ArrayList<Particle>(); 		// Change these particles so that the correlation function is maximized.
				int addedFrames2 			= 0;								// Number of particles added to the bin.
				for (int j = timeIndex[i-1]; j < timeIndex[i];j++){
					if (addedFrames2 < nParticles &&
							filteredResults.get(j).frame < frameBin*i ){
						Data2.add(filteredResults.get(j));
						addedFrames2++;
					}
				}

				int[] roughStepsize  	= {stepSize[0]*5,stepSize[1]*5}; // increase stepSize for a first round of optimization. 
				double[] roughlambda	= AutoCorrelation.getLambda(Data1,Data2,roughStepsize,lb_xy,ub_xy); // Get rough estimate of lambda, drift.			
				int[] fineLb 			= {(int) (roughlambda[0] - stepSize[0]),(int) (roughlambda[1] - stepSize[1])}; 	// Narrow lower boundry.
				int[] fineUb 			= {(int) (roughlambda[0] + stepSize[0]),(int) (roughlambda[1] + stepSize[1])}; 	// Narrow upper boundry.
				double[] tempLamda 		= AutoCorrelation.getLambda(Data1,Data2,stepSize ,fineLb ,fineUb); 				// Get drift.
				lambdax[i] 				= tempLamda[0] + lambdax[i-1];
				lambday[i] 				= tempLamda[1] + lambday[i-1];	
			}


			int countx = lambda.length-1;
			int county = lambda.length-1;

			for (int j =  (int) (Math.round(1.0/BinFrac) - 1); j >0; j--){
				double[] temp 			= interp(lambdax[j],lambdax[j-1],(int) frameBin);
				for (int k = 0; k < temp.length; k++){
					lambda[countx][0] = temp[k];
					countx--;
				}
				double[] temp2 			= interp(lambday[j],lambday[j-1],(int) frameBin);
				for (int k = 0; k < temp2.length; k++){
					lambda[county][1] = temp2[k];
					county--;
				}
			}
			int[] timeV = new int[lambda.length];
			for (int i = 0; i < timeV.length;i++){
				timeV[i] = i;
			}		

			for (int i = 0; i < filteredResults.size(); i++){

				Particle tempPart 	= new Particle();
				tempPart.frame	 	= filteredResults.get(i).frame;
				tempPart.chi_square = filteredResults.get(i).chi_square;
				tempPart.photons 	= filteredResults.get(i).photons;
				tempPart.precision_x= filteredResults.get(i).precision_x;
				tempPart.precision_y= filteredResults.get(i).precision_y;			
				tempPart.sigma_x 	= filteredResults.get(i).sigma_x;
				tempPart.sigma_y 	= filteredResults.get(i).sigma_x;

				tempPart.x = filteredResults.get(i).x - lambda[(int) tempPart.frame-1][0];
				tempPart.y = filteredResults.get(i).y - lambda[(int) tempPart.frame-1][1];
				correctedResults.add(tempPart);
			}
			double[] lx = new double[lambda.length];
			double[] ly = new double[lambda.length];
			for (int i 	= 0; i < lambda.length;i++){
				lx[i] 	= lambda[i][0];
				ly[i]	= lambda[i][1];
			}
			plot(lx,ly,timeV);
			TableIO.Store(correctedResults);
		}
		System.out.println("No drift correction possible, not enough particles in each bin.");
	}
	 
	public static double[] interp(double X1, double X2, int n){
		double[] extendedX 	= new double[n]; 
		extendedX[0] 		= X1;
		extendedX[n-1] 		= X2;

		double step 		= (X2-X1)/(n-2);
		for (int i = 1; i < n-1; i++){
			extendedX[i] = extendedX[i-1] + step;
		}

		return extendedX;
	}
	
	/*
	 * Supporting plot functions
	 */
	static void plot(double[] values) {
		double[] x = new double[values.length];
		for (int i = 0; i < x.length; i++)
			x[i] = i;
		Plot plot = new Plot("Plot window", "x", "values", x, values);
		plot.show();
	}
	static void plot(double[] values,int[] x_axis) {
		double[] x = new double[values.length];
		for (int i = 0; i < x.length; i++)
			x[i] = x_axis[i];
		Plot plot = new Plot("Plot window", "x", "values", x, values);
		plot.show();
	}
	static void plot(double[] values, double[] values2,int[] x_axis) {
		double[] x = new double[values.length];
		for (int i = 0; i < x.length; i++)
			x[i] = x_axis[i];
		Plot plot = new Plot("Plot window", "x", "values", x, values);
		plot.setColor(Color.GREEN);
		plot.draw();
		plot.addPoints(x, values, Plot.LINE);

		plot.setColor(Color.RED);
		plot.draw();
		plot.addPoints(x, values2, Plot.LINE);

		plot.addLegend("X: green" + "\n" + "Y: red");
		plot.show();
	}

}
