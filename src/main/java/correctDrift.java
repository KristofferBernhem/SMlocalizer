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
import java.awt.Color;
import java.util.ArrayList;
import ij.gui.Plot;


public class correctDrift {
	public static void run(int[] lb, int[] ub, double BinFrac, int nParticles, int minParticles, int[] stepSize){
		ArrayList<Particle> locatedParticles = TableIO.Load(); // Get current table data.		
		ArrayList<Particle> correctedResults = new ArrayList<Particle>(); // Output arraylist, will contain all particles that fit user input requirements after drift correction.
		if (locatedParticles.size() == 0){
			return;
		}
		double Channels = 1;
		for (int i = 0; i < correctedResults.size(); i ++){
			if (correctedResults.get(i).channel>Channels){
				Channels = correctedResults.get(i).channel;
			}
		}
		for(double Ch = 1; Ch <= Channels; Ch++){
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
				if (locatedParticles.get(i).include == 1 &&
						locatedParticles.get(i).channel == Ch){
					filteredResults.add(locatedParticles.get(i)); 	// Add particles that match user set parameters.										
					if (filteredResults.get(filteredResults.size()-1).frame > frameBin*count){	// First time data from a new bin is added, register index.						
						timeIndex[count] = filteredResults.size() - 1; // Get the index for the first entry with the new bin.
						timeIndexDouble[count] = filteredResults.get(filteredResults.size()-1).frame;
						count++;
					}
				}
			}
			if (count == 0){
				return;
			}
			timeIndex[timeIndex.length-1] =  filteredResults.size(); 									// Final entry.
			timeIndexDouble[timeIndex.length-1] = filteredResults.get(filteredResults.size()-1).frame; 	// Final entry.
			double[] lambdax = new double[(int) Math.round(1.0/BinFrac)];								// Holds drift estimate between all bins in x.
			double[] lambday = new double[(int) Math.round(1.0/BinFrac)];								// Holds drift estimate between all bins in y.
			double[] lambdaz = new double[(int) Math.round(1.0/BinFrac)];								// Holds drift estimate between all bins in y.
			lambdax[0] = 0;																				// Drift for first bin is 0.
			lambday[0] = 0;																				// Drift for first bin is 0.
			lambdaz[0] = 0;																				// Drift for first bin is 0.
			int maxTime =(int) timeIndexDouble[timeIndexDouble.length-1]; 								// Last frame included.
			double[][] lambda = new double[maxTime][3];													// Holds interpolated drift corrections in x and y.
			int okBins = 0;																				// If all bins are ok, this will still be 0.
			for (int i = 1; i < timeIndex.length ; i++){ 												// Loop over all bins.
				if ((timeIndex[i] - timeIndex[i-1])<minParticles){									// If the bin lacks enough points to meet user minimum criteria.
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
//					int[] newLb = {(int) (lb[0] +lambdax[i-1]), (int) (lb[1] +lambday[i-1]),(int) (lb[2] +lambdaz[i-1]) };
	//				int[] newUb = {(int) (ub[0] +lambdax[i-1]), (int) (ub[1] +lambday[i-1]),(int) (ub[2] +lambdaz[i-1]) };
					int[] roughStepsize  	= {stepSize[0]*5,stepSize[1]*5,stepSize[2]*5}; // increase stepSize for a first round of optimization.
		//			double[] roughlambda	= AutoCorrelation.getLambda(Data1,Data2,roughStepsize,newLb,newUb); // Get rough estimate of lambda, drift.
					double[] roughlambda	= AutoCorrelation.getLambda(Data1,Data2,roughStepsize,lb,ub); // Get rough estimate of lambda, drift.			
					int[] fineLb 			= {(int) (roughlambda[0] - stepSize[0]),(int) (roughlambda[1] - stepSize[1]),(int) (roughlambda[2] - stepSize[2])}; 	// Narrow lower boundry.
					int[] fineUb 			= {(int) (roughlambda[0] + stepSize[0]),(int) (roughlambda[1] + stepSize[1]),(int) (roughlambda[2] + stepSize[2])}; 	// Narrow upper boundry.
					for(int j = 0; j < lb.length;j++){
						if(lb[j] == 0){
							fineLb[j] = 0;
						}					
						if (ub[j] == 0){
							fineUb[j] = 0;
						}
					}
					double[] tempLamda 		= AutoCorrelation.getLambda(Data1,Data2,stepSize ,fineLb ,fineUb); 				// Get drift.
//					lambdax[i] 				= tempLamda[0];
//					lambday[i] 				= tempLamda[1];
//					lambdaz[i] 				= tempLamda[2];
					lambdax[i] 				= tempLamda[0] + lambdax[i-1];
					lambday[i] 				= tempLamda[1] + lambday[i-1];
					lambdaz[i] 				= tempLamda[2] + lambdaz[i-1];
				}


				int countx = lambda.length-1;
				int county = lambda.length-1;
				int countz = lambda.length-1;

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
					double[] temp3 			= interp(lambdaz[j],lambdaz[j-1],(int) frameBin);
					for (int k = 0; k < temp3.length; k++){
						lambda[countz][2] = temp3[k];
						countz--;
					}
				}
				double[] timeV = new double[lambda.length];
				for (int i = 0; i < timeV.length;i++){
					timeV[i] = i;
				}		
				//int index = 0;			
				for (int index = 0; index < locatedParticles.size(); index++ ){
					//while(locatedParticles.get(index).channel == Ch){
					if(locatedParticles.get(index).channel == Ch){
						Particle tempPart 	= new Particle();
						tempPart.frame	 	= locatedParticles.get(index).frame;
						tempPart.chi_square = locatedParticles.get(index).chi_square;
						tempPart.photons 	= locatedParticles.get(index).photons;
						tempPart.include 	= locatedParticles.get(index).include;
						tempPart.precision_x= locatedParticles.get(index).precision_x;
						tempPart.precision_y= locatedParticles.get(index).precision_y;
						tempPart.precision_z= locatedParticles.get(index).precision_z;
						tempPart.sigma_x 	= locatedParticles.get(index).sigma_x;
						tempPart.sigma_y 	= locatedParticles.get(index).sigma_y;
						tempPart.sigma_z 	= locatedParticles.get(index).sigma_z;
						tempPart.channel 	= locatedParticles.get(index).channel;

						tempPart.x = locatedParticles.get(index).x - lambda[(int) tempPart.frame-1][0];
						tempPart.y = locatedParticles.get(index).y - lambda[(int) tempPart.frame-1][1];
						tempPart.z = locatedParticles.get(index).z - lambda[(int) tempPart.frame-1][2];
						if(tempPart.x >= 0){
							if(tempPart.y >= 0){
								if(tempPart.z >= 0){
									correctedResults.add(tempPart);
								}
							}
						}
						
					}

				}
				double[] lx = new double[lambda.length];
				double[] ly = new double[lambda.length];
				double[] lz = new double[lambda.length];
				for (int i 	= 0; i < lambda.length;i++){
					lx[i] 	= lambda[i][0];
					ly[i]	= lambda[i][1];
					lz[i]	= lambda[i][2];
				}
				plot(lx,ly,timeV);
				
			}else
				System.out.println("No drift correction possible, not enough particles in each bin.");
			
		}// Channel loop ends.	
		
		if (correctedResults.size()== locatedParticles.size()){
			TableIO.Store(correctedResults);
			System.out.println("Drift corrections made");
		}
	}


	public static void ChannelAlign(int[] lb, int[] ub, int nParticles, int minParticles, int[] stepSize){
		ArrayList<Particle> locatedParticles = TableIO.Load(); // Get current table data.
		if (locatedParticles.size() == 0){ // If no particles.
			return;
		}

		double Channels = 1;
		for (int i = 0; i < locatedParticles.size(); i ++){
			if (locatedParticles.get(i).channel>Channels){
				Channels = locatedParticles.get(i).channel;
			}
		}
		if (Channels < 2){ // If only 1 channel.
			return;
		}
		for (double Ch = 2; Ch < Channels; Ch++){
			ArrayList<Particle> Data1 	= new ArrayList<Particle>(); 		// Target particles.			
			int addedFrames1 			= 0;								// Number of particles added to the bin.
			int index = 0;
			while (addedFrames1 < nParticles && index < locatedParticles.size()){
				if (locatedParticles.get(index).channel == Ch-1 &&
						locatedParticles.get(index).include == 1){
					Data1.add(locatedParticles.get(index));					
					addedFrames1++;
				}
				index++;
			}
			if(addedFrames1 < minParticles){
				return;
			}

			ArrayList<Particle> Data2 	= new ArrayList<Particle>(); 		// Change these particles so that the correlation function is maximized.
			int addedFrames2 			= 0;								// Number of particles added to the bin.
			index = 0;
			while (addedFrames2 < nParticles && index < locatedParticles.size()){
				if (locatedParticles.get(index).channel == Ch &&
						locatedParticles.get(index).include == 1){
					Data2.add(locatedParticles.get(index));					
					addedFrames2++;
				}
				index++;
			}

			if(addedFrames2 < minParticles){
				return;
			}


			int[] roughStepsize  	= {stepSize[0]*5,stepSize[1]*5,stepSize[2]*5}; // increase stepSize for a first round of optimization. 
			double[] roughlambda	= AutoCorrelation.getLambda(Data1,Data2,roughStepsize,lb,ub); // Get rough estimate of lambda, drift.			
			int[] fineLb 			= {(int) (roughlambda[0] - stepSize[0]),(int) (roughlambda[1] - stepSize[1]),(int) (roughlambda[2] - stepSize[2])}; 	// Narrow lower boundry.
			
			int[] fineUb 			= {(int) (roughlambda[0] + stepSize[0]),(int) (roughlambda[1] + stepSize[1]),(int) (roughlambda[2] + stepSize[2])}; 	// Narrow upper boundry.
			for(int i = 0; i < lb.length;i++){
				if(lb[i] == 0){
					fineLb[i] = 0;
				}					
				if (ub[i] == 0){
					fineUb[i] = 0;
				}
			}
			double[] lambdaCh 		= AutoCorrelation.getLambda(Data1,Data2,stepSize ,fineLb ,fineUb); 				// Get drift.

			for(int i = 0; i < locatedParticles.size(); i++){
				if (locatedParticles.get(i).channel == Ch){
					locatedParticles.get(i).x = locatedParticles.get(i).x - lambdaCh[0];
					locatedParticles.get(i).y = locatedParticles.get(i).y - lambdaCh[1];
					locatedParticles.get(i).z = locatedParticles.get(i).z - lambdaCh[2];
				}
			}		
		}		
		for (int i = locatedParticles.size()-1; i >=0; i--){
			if(locatedParticles.get(i).x < 0 ||
					locatedParticles.get(i).y < 0 ||
					locatedParticles.get(i).z < 0){
				locatedParticles.remove(i);
			}		
		}
		TableIO.Store(locatedParticles);
		System.out.println("Channels aligned.");
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
	static void plot(double[] values, double[] values2,double[] x_axis) {
		Plot newPlot = new Plot("Drift corrections","frame","drift [nm]");
		newPlot.setColor(Color.GREEN);
		newPlot.addPoints(x_axis,values, Plot.LINE);
		newPlot.setColor(Color.RED);
		newPlot.addPoints(x_axis,values2, Plot.LINE);
		newPlot.addLegend("X \n Y");
		newPlot.show();		
	}

}