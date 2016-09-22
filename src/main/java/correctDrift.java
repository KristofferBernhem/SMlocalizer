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
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import ij.gui.Plot;

// TODO: check drift correction application of correction. Appears to skip first bin!. 
// TODO: Go through function and error check.
public class correctDrift {
	
	public static void main(String[] args){ // test case.
		Particle P = new Particle();
		P.x = 100;
		P.y = 100;
		P.z = 50;		
		P.channel = 1;
		Particle Psecond = new Particle();
		Psecond.x = 1000;
		Psecond.y = 1000;
		Psecond.z = 500;
		Psecond.channel = 1;
		ArrayList<Particle> A = new ArrayList<Particle>();
		double drift = 0.20;
		for (double i = 1; i < 2000; i++){
			Particle P2 = new Particle();
			P2.x = P.x - i*drift;
			P2.y = P.y - i*drift;
			P2.z = P.z - 2*i*drift;
			P2.channel = 1;
			P2.include = 1;
			P2.frame = (int) i;

			A.add(P2);
			
			Particle P4 = new Particle();
			P4.x = Psecond.x - i*drift;
			P4.y = Psecond.y - i*drift;
			P4.z = Psecond.z - 2*i*drift;
			P4.frame = (int) i;
			P4.channel = 1;
			P4.include = 1;
			A.add(P4);
			
		}
		final ij.ImageJ ij = new ij.ImageJ();
		ij.setVisible(true);
		TableIO.Store(A);
		
		int[][] boundry = new int[3][10];
		int[] nBins = new int[10];
		int[] nParticles = new int[10];
		int[] minParticles = new int[10];
		for (int i = 0; i < 10;i++)
		{
			boundry[0][i] = 250;
			boundry[1][i] = 250;
			boundry[2][i] = 250;
			nBins[i] = 10;
			nParticles[i] = 1000;
			minParticles[i] = 10;					
			
		}
		run(boundry,nBins,nParticles,minParticles,false);
	}
	public static void run(int[][] boundry, int[] nBins, int[] nParticles, int[] minParticles,boolean GPU){
		int[] maxDistance = {2500,2500,2500}; // everything beyond 50 nm apart after shift will not affect outcome.
		ArrayList<Particle> locatedParticles = TableIO.Load(); // Get current table data.		
		ArrayList<Particle> correctedResults = new ArrayList<Particle>(); // Output arraylist, will contain all particles that fit user input requirements after drift correction.
		if (locatedParticles.size() == 0){
			return;
		}
		double Channels = locatedParticles.get(locatedParticles.size()-1).channel;
		
		int width = 0;
		int height = 0;
		int depth = 0;
		for (int i = 0; i < locatedParticles.size(); i ++){
			if (locatedParticles.get(i).channel>Channels){
				Channels = locatedParticles.get(i).channel;
			}
			if(locatedParticles.get(i).x > width){
				width = (int) Math.round(locatedParticles.get(i).x);
			}
			if(locatedParticles.get(i).y > height){
				height = (int) Math.round(locatedParticles.get(i).y);
			}
			if(locatedParticles.get(i).z > depth){
				depth = (int) Math.round(locatedParticles.get(i).z);
			}
		}
		
		int lastIndex = 0;
		int startIndex = 0;
		for (int Ch = 1; Ch <= Channels; Ch ++)
		{
			if(Ch > 1)
				startIndex = lastIndex+1;
			for (int i = startIndex; i <  locatedParticles.size(); i++) // locate last index of this series.
			{
				if (locatedParticles.get(i).channel == Ch)
					lastIndex = i;
				
			} // locate index interval for this channel.
			int processors 					= Runtime.getRuntime().availableProcessors();				// Number of processor cores on this system.
			ExecutorService exec 			= Executors.newFixedThreadPool(processors);					// Set up parallel computing using all cores.
			List<Callable<float[]>> tasks 	= new ArrayList<Callable<float[]>>();						// Preallocate.

			boolean enoughParticles = true;
			int binSize = Math.round((locatedParticles.get(lastIndex).frame - locatedParticles.get(startIndex).frame)/(nBins[Ch-1] + 1));
			int bin  = 0;
			while (bin < nBins[Ch-1]) // seperate out data.
			{
				if (enoughParticles)
					{
					ArrayList<Particle> A = new ArrayList<Particle>();
					ArrayList<Particle> B = new ArrayList<Particle>();
					for (int i = startIndex; i <= lastIndex; i ++){ // loop over the full range.
						if (locatedParticles.get(i).frame > bin*binSize &&
								locatedParticles.get(i).frame <= (bin+1)*binSize)
						{
							A.add(locatedParticles.get(i));
						}
						else if(locatedParticles.get(i).frame > (bin+1)*binSize &&
								locatedParticles.get(i).frame <= (bin+2)*binSize)
						{
							B.add(locatedParticles.get(i));
						}
						else if (bin== nBins[Ch-1] &&
								locatedParticles.get(i).frame > (bin+1)*binSize &&
								i <= lastIndex)
						{														
								B.add(locatedParticles.get(i));
							
						}
					}
					final ArrayList<Particle> Beta = hasNeighbors(A, B, (double) maxDistance[0]);
					final ArrayList<Particle> Alpha = hasNeighbors(Beta, A, (double) maxDistance[0]);
					if(Alpha.size() < minParticles[Ch-1] &&
							Beta.size() < minParticles[Ch-1]){
						ij.IJ.log("not enough particles, no shift correction possible");
						enoughParticles = false;
						bin = nBins[Ch-1];
					} else if (!GPU){
						final int[] boundryFinal = {boundry[0][Ch-1], boundry[1][Ch-1], boundry[2][Ch-1]};
						final int[] maxDistanceFinal = maxDistance;
						Callable<float[]> c = new Callable<float[]>() {													// Computation to be done.							
							@Override
							public float[] call() throws Exception {		
								return DriftCompensation.findDrift (Alpha, Beta, boundryFinal,  maxDistanceFinal);// Actual call for each parallel process.
							}
						};
						tasks.add(c);	
					} // Parallel CPU bound.
				}						
				bin++;
			} // while loop to set up correction calculations.
			double[] lambdax = new double[nBins[Ch-1]+1];								// Holds drift estimate between all bins in x.
			double[] lambday = new double[nBins[Ch-1]+1];								// Holds drift estimate between all bins in y.
			double[] lambdaz = new double[nBins[Ch-1]+1];								// Holds drift estimate between all bins in y.
			lambdax[0] = 0;															// Drift for first bin is 0.
			lambday[0] = 0;															// Drift for first bin is 0.
			lambdaz[0] = 0;															// Drift for first bin is 0.
			
			try {
				List<Future<float[]>> parallelComputeSmall = exec.invokeAll(tasks);		// Execute computation.
				float[] Corr;
				for (int i = 1; i <= parallelComputeSmall.size(); i++){							// Loop over and transfer results.
					try {
						Corr = parallelComputeSmall.get(i-1).get();	
						lambdax[i] = Corr[1] + lambdax[i - 1];											// Update best guess at x shift.
						lambday[i] = Corr[2] + lambday[i - 1];											// Update best guess at y shift.
						lambdaz[i] = Corr[3] + lambdaz[i - 1];											// Update best guess at z shift.				
					} catch (ExecutionException e) {
						e.printStackTrace();
					}
				}
			} catch (InterruptedException e) {
	
				e.printStackTrace();
			}
			finally {
				exec.shutdown();	// Shut down connection to cores.
			}	
		
			double[][] lambda = new double[locatedParticles.get(lastIndex).frame][3];			
			// apply drift compensation.
			int idx = binSize;
			for (int i = 1; i <= nBins[Ch-1]; i++)
			{
			
				double xStep = lambdax[i] - lambdax[i-1];
				xStep /= binSize;
				double yStep = lambday[i] - lambday[i-1];
				yStep /= binSize;
				double zStep = lambdaz[i] - lambdaz[i-1];
				zStep /= binSize;
				int stepIdx = 0;
				while(idx <= binSize*(i+1))
				{
					lambda[idx][0] = lambdax[i-1] + xStep*stepIdx;
					lambda[idx][1] = lambday[i-1] + yStep*stepIdx;
					lambda[idx][2] = lambdaz[i-1] + zStep*stepIdx;
					idx++;
					stepIdx++;
				}
				if (i == nBins[Ch-1])
				{
					while(idx <lambda.length)
					{
						lambda[idx][0] = lambdax[i-1] + xStep*stepIdx;
						lambda[idx][1] = lambday[i-1] + yStep*stepIdx;
						lambda[idx][2] = lambdaz[i-1] + zStep*stepIdx;
						idx++;
						stepIdx++;
					}
				}
			}
			
			idx = startIndex;			
			while (idx <= lastIndex)
			{				
				Particle tempPart 	= new Particle();
				tempPart.frame	 	= locatedParticles.get(idx).frame;
				tempPart.r_square 	= locatedParticles.get(idx).r_square;
				tempPart.photons 	= locatedParticles.get(idx).photons;
				tempPart.include 	= locatedParticles.get(idx).include;
				tempPart.precision_x= locatedParticles.get(idx).precision_x;
				tempPart.precision_y= locatedParticles.get(idx).precision_y;
				tempPart.precision_z= locatedParticles.get(idx).precision_z;
				tempPart.sigma_x 	= locatedParticles.get(idx).sigma_x;
				tempPart.sigma_y 	= locatedParticles.get(idx).sigma_y;
				tempPart.sigma_z 	= locatedParticles.get(idx).sigma_z;
				tempPart.channel 	= locatedParticles.get(idx).channel;

				tempPart.x = locatedParticles.get(idx).x - lambda[tempPart.frame-1][0];
				tempPart.y = locatedParticles.get(idx).y - lambda[tempPart.frame-1][1];
				tempPart.z = locatedParticles.get(idx).z - lambda[tempPart.frame-1][2];
				if(tempPart.x >= 0){
					if(tempPart.y >= 0){
						if(tempPart.z >= 0){
							correctedResults.add(tempPart);
						}
					}
				}				
				idx++;
			}
			

			double[] lx = new double[lambda.length];
			double[] ly = new double[lambda.length];
			double[] lz = new double[lambda.length];
			for (int i 	= 0; i < lambda.length;i++){
				lx[i] 	= lambda[i][0];
				ly[i]	= lambda[i][1];
				lz[i]	= lambda[i][2];
			}
			double[] timeV = new double[lambda.length];
			for (int i = 0; i < timeV.length;i++){
				timeV[i] = i;
			}
			plot(lx,ly,lz,timeV);
			if (Ch == Channels){
				TableIO.Store(correctedResults);
			}			
		}// channel loop.
		System.out.println("Success!");
	}
	//	public static void run(int[] lb, int[] ub, double BinFrac, int nParticles, int minParticles, int[] stepSize){
	public static void run2(int[][] boundry, int[] nBins, int[] nParticles, int[] minParticles,boolean GPU){
		int[] maxDistance = {2500,2500,2500}; // everything beyond 50 nm apart after shift will not affect outcome.
		ArrayList<Particle> locatedParticles = TableIO.Load(); // Get current table data.		
		ArrayList<Particle> correctedResults = new ArrayList<Particle>(); // Output arraylist, will contain all particles that fit user input requirements after drift correction.
		if (locatedParticles.size() == 0){
			return;
		}
		double Channels = locatedParticles.get(locatedParticles.size()-1).channel;
		
		int width = 0;
		int height = 0;
		int depth = 0;
		for (int i = 0; i < correctedResults.size(); i ++){
			if (correctedResults.get(i).channel>Channels){
				Channels = correctedResults.get(i).channel;
			}
			if(correctedResults.get(i).x > width){
				width = (int) Math.round(correctedResults.get(i).x);
			}
			if(correctedResults.get(i).y > height){
				height = (int) Math.round(correctedResults.get(i).y);
			}
			if(correctedResults.get(i).z > depth){
				depth = (int) Math.round(correctedResults.get(i).z);
			}
		}
		for(int Ch = 1; Ch <= Channels; Ch++){
			int idx 			= locatedParticles.size() - 1;

			double frameBin = Math.round( 				// Bin size for drift correction based on total number of frames and user input fraction. 
					locatedParticles.get(idx).frame * 	// Last frame that was used.
					(double) (1.0/(nBins[Ch-1]-1)));						// User input fraction.
			
			ArrayList<Particle> filteredResults =  new ArrayList<Particle>(); // output arraylist.
			int[] timeIndex = new int[(nBins[Ch-1])];
			double[] timeIndexDouble = new double[(nBins[Ch-1])];
			
			int count = 0;					
			// Check which particles are within user set parameters.
			for (int i = 0; i < locatedParticles.size(); i++){
				if (locatedParticles.get(i).include == 1 &&
						locatedParticles.get(i).channel == Ch && 
						count < nBins[Ch-1]){
					filteredResults.add(locatedParticles.get(i)); 	// Add particles that match user set parameters.										
					if (filteredResults.get(filteredResults.size()-1).frame > frameBin*count){	// First time data from a new bin is added, register index.						
						timeIndex[count] = filteredResults.size() - 1; // Get the index for the first entry with the new bin.
						timeIndexDouble[count] = filteredResults.get(filteredResults.size()-1).frame;
						count++;
					}
				}
			}
			if (count == 0){
				System.out.println(frameBin);
				ij.IJ.log("not enough particles, no shift correction possible");
				return;
			}
			timeIndex[timeIndex.length-1] =  filteredResults.size(); 									// Final entry.
			timeIndexDouble[timeIndex.length-1] = filteredResults.get(filteredResults.size()-1).frame; 	// Final entry.
			double[] lambdax = new double[nBins[Ch-1]];								// Holds drift estimate between all bins in x.
			double[] lambday = new double[nBins[Ch-1]];								// Holds drift estimate between all bins in y.
			double[] lambdaz = new double[nBins[Ch-1]];								// Holds drift estimate between all bins in y.
			lambdax[0] = 0;																				// Drift for first bin is 0.
			lambday[0] = 0;																				// Drift for first bin is 0.
			lambdaz[0] = 0;																				// Drift for first bin is 0.
			int maxTime =(int) timeIndexDouble[timeIndexDouble.length-1]; 								// Last frame included.
			double[][] lambda = new double[maxTime][3];													// Holds interpolated drift corrections in x and y.
			int okBins = 0;																				// If all bins are ok, this will still be 0.
			for (int i = 1; i < timeIndex.length ; i++){ 												// Loop over all bins.
				if ((timeIndex[i] - timeIndex[i-1])<minParticles[Ch-1]){									// If the bin lacks enough points to meet user minimum criteria.
					okBins++;				
				}
			}


			int processors 					= Runtime.getRuntime().availableProcessors();				// Number of processor cores on this system.
			ExecutorService exec 			= Executors.newFixedThreadPool(processors);					// Set up parallel computing using all cores.
			List<Callable<float[]>> tasks 	= new ArrayList<Callable<float[]>>();						// Preallocate.

			boolean ToFewReached = false;
			if (okBins == 0){ 														// If all bins were ok.
				for (int i = 1; i < nBins[Ch-1]-1; i++){ 				// Loop over all bins.
					if(!ToFewReached){
						ArrayList<Particle> Data1 	= new ArrayList<Particle>(); 		// Target particles.			
						int addedFrames1 			= 0;								// Number of particles added to the bin.
						for (int j = timeIndex[i]; j < timeIndex[i+1];j++){
							if (addedFrames1 < nParticles[Ch-1] &&
									filteredResults.get(j).frame < frameBin*(i+1)){
								Data1.add(filteredResults.get(j));
								addedFrames1++;
							}
						}			
						ArrayList<Particle> Data2 	= new ArrayList<Particle>(); 		// Change these particles so that the correlation function is maximized.
						int addedFrames2 			= 0;								// Number of particles added to the bin.
						for (int j = timeIndex[i-1]; j < timeIndex[i];j++){
							if (addedFrames2 < nParticles[Ch-1] &&
									filteredResults.get(j).frame < frameBin*i ){
								Data2.add(filteredResults.get(j));
								addedFrames2++;
							}
						}
						
						
						final ArrayList<Particle> Beta = hasNeighbors(Data1, Data2, (double) maxDistance[0]);
						final ArrayList<Particle> Alpha = hasNeighbors(Beta, Data1, (double) maxDistance[0]);
						System.out.println("Alpha " + Data1.size() + " from round " + i);
						System.out.println("Beta " + Data2.size() + " from round " + i);
						if(Alpha.size() < minParticles[Ch-1] &&
								Beta.size() < minParticles[Ch-1]){
							ij.IJ.log("not enough particles, no shift correction possible");
						//	System.out.println(Alpha.size() + " in alpha and " + Beta.size() + " in beta from " + i);
							ToFewReached = true;
						} else if (!GPU){
							final int[] boundryFinal = {boundry[0][Ch-1], boundry[1][Ch-1]};
							final int[] maxDistanceFinal = maxDistance;
							Callable<float[]> c = new Callable<float[]>() {													// Computation to be done.							
								@Override
								public float[] call() throws Exception {		
									return DriftCompensation.findDrift (Alpha, Beta, boundryFinal,  maxDistanceFinal);// Actual call for each parallel process.
								}
							};
							tasks.add(c);	
						} // Parallel CPU bound.
							
						/*
						if(GPU){
							// TODO: run ptx code here.
							  
							 } // GPU bound. */

					} // verification of correct number of particles found.

				} // loop over bins, setting up calculations.
				if (!GPU){
					try {
						List<Future<float[]>> parallelComputeSmall = exec.invokeAll(tasks);		// Execute computation.
						float[] Corr;
						lambdax[0] = 0;
	 					lambday[0] = 0;
	 					lambdaz[0] = 0;
						for (int i = 1; i <= parallelComputeSmall.size(); i++){							// Loop over and transfer results.
							try {
								Corr = parallelComputeSmall.get(i-1).get();	
							
								
									lambdax[i] = Corr[1] + lambdax[i - 1];											// Update best guess at x shift.
									lambday[i] = Corr[2] + lambday[i - 1];											// Update best guess at y shift.
									lambdaz[i] = Corr[3] + lambdaz[i - 1];											// Update best guess at z shift.
								
	
							} catch (ExecutionException e) {
								e.printStackTrace();
							}
						}
					} catch (InterruptedException e) {
	
						e.printStackTrace();
					}
					finally {
						exec.shutdown();	// Shut down connection to cores.
					}	
				} // Parallel CPU bound.
				int countx = lambda.length-1;
				int county = lambda.length-1;
				int countz = lambda.length-1;

				for (int j =  nBins[Ch-1]; j >0; j--){
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

//				System.out.println(lambda[lambda.length-1][0]);
				double[] timeV = new double[lambda.length];
				for (int i = 0; i < timeV.length;i++){
					timeV[i] = i;
				}		

				for (int index = 0; index < locatedParticles.size(); index++ ){

					if(locatedParticles.get(index).channel == Ch){
						Particle tempPart 	= new Particle();
						tempPart.frame	 	= locatedParticles.get(index).frame;
						tempPart.r_square 	= locatedParticles.get(index).r_square;
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
				if (Ch == Channels){
					TableIO.Store(correctedResults);
				}

			}else
				ij.IJ.log("No drift correction possible, not enough particles in each bin.");			
		} // Channel loop ends.		
		
	}


	public static void ChannelAlign(int[][] boundry, int[] nParticles, int[] minParticles, boolean GPU){
		int[] maxDistance = {2500,2500,2500}; // everything beyond 50 nm apart after shift will not affect outcome.
		ArrayList<Particle> locatedParticles = TableIO.Load(); // Get current table data.
		if (locatedParticles.size() == 0){ // If no particles.
			ij.IJ.log("No data to align.");
			return;
		}

		double Channels = locatedParticles.get(locatedParticles.size()-1).channel;
		for (int i = 0; i < locatedParticles.size(); i ++){
			if (locatedParticles.get(i).channel>Channels){
				Channels = locatedParticles.get(i).channel;
			}
		}
		if (Channels == 1){ // If only 1 channel.
			ij.IJ.log("Single channel data, no second channel to align against.");
			return;
		}
		for (int Ch = 2; Ch <= Channels; Ch++){
			ArrayList<Particle> Data1 	= new ArrayList<Particle>(); 		// Target particles.			
			int addedFrames1 			= 0;								// Number of particles added to the bin.
			int index = 0;
			while (addedFrames1 < nParticles[Ch-2] && index < locatedParticles.size()){
				if (locatedParticles.get(index).channel == Ch-1 &&
						locatedParticles.get(index).include == 1){
					Data1.add(locatedParticles.get(index));					
					addedFrames1++;
				}
				index++;
			}


			ArrayList<Particle> Data2 	= new ArrayList<Particle>(); 		// Change these particles so that the correlation function is maximized.
			int addedFrames2 			= 0;								// Number of particles added to the bin.
			index = 0;
			while (addedFrames2 < nParticles[Ch-1] && index < locatedParticles.size()){
				if (locatedParticles.get(index).channel == Ch &&
						locatedParticles.get(index).include == 1){
					Data2.add(locatedParticles.get(index));					
					addedFrames2++;
				}
				index++;
			}
			
			ArrayList<Particle> Beta = hasNeighbors(Data1, Data2, (double) maxDistance[0]);
			ArrayList<Particle> Alpha = hasNeighbors(Beta, Data1, (double) maxDistance[0]);
			if(Alpha.size() < minParticles[Ch-2]){
				ij.IJ.log("not enough particles, no alignment possible");
				return;
			}
			if(Beta.size() < minParticles[Ch-1]){
				ij.IJ.log("not enough particles, no alignment possible");
				return;
			}
			float[] lambdaCh = {0,0,0,0}; // initiate.
			if(GPU){
				// TODO: run ptx code here.
			}else{
				//AutoCorr DriftCalc 		= new AutoCorr(Data1, Data2, stepSize, boundry, maxDistance);
				//lambdaCh 				= DriftCalc.optimize();
			//	double convergence = 1E-3;
			//	int maxIterations = 1000;
				//System.out.println(Alpha.size() + " and " + Beta.size());
				//lambdaCh = DriftCompensation.findDrift(Alpha,Beta,boundry,maxDistance,convergence,maxIterations);
				int[] boundryCh = {boundry[0][Ch-1], boundry[1][Ch-1]}; 
				lambdaCh = DriftCompensation.findDrift (Alpha, Beta, boundryCh,  maxDistance);// Actual call for each parallel process.
				ij.IJ.log("Channel " + Ch + " shifted by " + lambdaCh[1]+  " x " + lambdaCh[2] + " x " + lambdaCh[3] + " nm.");
			}
			for(int i = 0; i < locatedParticles.size(); i++){
				if (locatedParticles.get(i).channel == Ch){
					locatedParticles.get(i).x = locatedParticles.get(i).x - lambdaCh[1];
					locatedParticles.get(i).y = locatedParticles.get(i).y - lambdaCh[2];
					locatedParticles.get(i).z = locatedParticles.get(i).z - lambdaCh[3];
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
		ij.IJ.log("Channels aligned.");
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
	static void plot(double[] values, double[] values2, double[] values3, double[] x_axis) {
		Plot newPlot = new Plot("Drift corrections","frame","drift [nm]");
		newPlot.setColor(Color.GREEN);
		newPlot.addPoints(x_axis,values, Plot.LINE);
		newPlot.setColor(Color.RED);
		newPlot.addPoints(x_axis,values2, Plot.LINE);
		newPlot.setColor(Color.BLUE);
		newPlot.addPoints(x_axis,values3, Plot.LINE);
		newPlot.addLegend("X \n Y \n Z");
		newPlot.show();		
	}
	public static ArrayList<Particle> hasNeighbors(ArrayList<Particle> Alpha, ArrayList<Particle> Beta, double maxDistance)
	{	
		ArrayList<Particle> Include = new ArrayList<Particle>();		
		boolean[] retainBeta = new boolean[Beta.size()];
		for (int i = 0; i < Alpha.size(); i++) // loop over all entries in Alpha.
		{
			double x = Alpha.get(i).x;
			double y = Alpha.get(i).y;
			double z = Alpha.get(i).z;
			
			for (int j = 0; j < Beta.size(); j++)
			{
				double distance = Math.sqrt(
						(x-Beta.get(j).x)*(x-Beta.get(j).x) +
						(y-Beta.get(j).y)*(y-Beta.get(j).y)+
						(z-Beta.get(j).z)*(z-Beta.get(j).z) );
				if (distance < maxDistance)
					retainBeta[j] = true;												
			}						
		}
		for (int i = 0; i < Beta.size(); i++)
		{
			if(retainBeta[i])
				Include.add(Beta.get(i));
		}
		
		return Include;
	}
}
