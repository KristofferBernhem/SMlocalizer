/* Copyright 2017 Kristoffer Bernhem.
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
/**
 *
 * @author kristoffer.bernhem@gmail.com
 */

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;


public class ImageCrossCorr3D {
	static byte[][][] reference;
	static byte[][][] target;
	static double refSquare;
	static double tarMean;
	static double refMean;

	public ImageCrossCorr3D(byte[][][] constructReference, byte[][][] constructTarget, double constructRefSquare, double constructRefMean, double constructTarMean)
	{
		ImageCrossCorr3D.reference 	= constructReference;		
		ImageCrossCorr3D.target 	= constructTarget;
		ImageCrossCorr3D.refSquare	= constructRefSquare;
		ImageCrossCorr3D.refMean 	= constructRefMean;
		ImageCrossCorr3D.tarMean 	= constructTarMean;
	}

	public static ArrayList<Particle> run(ArrayList<Particle> inputParticles, int[] nBins, int[][] boundry, int[] dimensions,int pixelSize, int pixelSizeZ, boolean optimize) // main function call. Will per channel autocorrelate between time binned data.
	{
		ArrayList<Particle> shiftedParticles = new ArrayList<Particle>();

		int nChannels 	= inputParticles.get(inputParticles.size()-1).channel; // data is sorted on a per channel basis.
		int idx 		= 0; // index for inputParticles.
		int zOffset 	= 0;
		for (int i = 0; i < inputParticles.size();i++)
		{
			if (zOffset > Math.floor(inputParticles.get(i).z))
				zOffset = (int) Math.floor(inputParticles.get(i).z);
		}
		zOffset = -zOffset;

		for (int ch = 1; ch <= nChannels; ch++)
		{	
			int tempIdx 	= idx; // hold value
			int maxShift 	= boundry[0][ch-1];
			int maxShiftZ 	= boundry[1][ch-1];

			while (idx < inputParticles.size() && inputParticles.get(idx).channel == ch)
			{				
				idx++;
			} // loop whilst... Find final entry for this channel.
			idx--;

			int maxFrame = inputParticles.get(idx).frame; // final frame included in this channel.

			int binsize = (int) (maxFrame/nBins[ch-1]);		// size of each bin (in number of frames).		

			byte[][][] referenceFrame 	= new byte[dimensions[0]][dimensions[1]][dimensions[2]]; // create the reference array.
			byte[][][] targetFrame 	= new byte[dimensions[0]][dimensions[1]][dimensions[2]]; 		// create the target (shifted) array.	//		
			idx = tempIdx;	// return idx to startpoint of this loop.
			int currBin = 2;	// start with bin 2.

			double targetMean = 0;	// mean value of target array.
			double referenceMean = 0;	// mean value of reference array.
			while(inputParticles.get(idx).frame<=binsize) // populate target first due to while loop design.
			{

				if (inputParticles.get(idx).include == 1)	// only include particles that are ok from the users parameter choice.
				{
					targetFrame[(int)(inputParticles.get(idx).x/pixelSize)][(int)(inputParticles.get(idx).y/pixelSize)][(int)((inputParticles.get(idx).z+zOffset)/pixelSizeZ)] += 1;	// increase value by one if there is a particle within the voxel.
					targetMean++;	// keep track of total number of added particles (for mean calculation).
				}
				idx++;	// step forward.
			}

			targetMean /= (dimensions[0]*dimensions[1]*dimensions[2]);	// calculate the mean.
			double[][] optimalShift = new double[4][nBins[ch-1]];	// this vector will hold all shift values.
			while (currBin <= nBins[ch-1])							// loop over all bins.
			{
				referenceMean = targetMean;							// last loops target is now the reference.
				targetMean = 0;										// new target array, set old values to 0.
				for (int i = 0; i < targetFrame.length; i++)		// loop over the entire target array.
				{
					for(int j = 0; j < targetFrame[0].length; j++)
					{
						for (int k = 0; k < targetFrame[0][0].length; k++)
						{
							referenceFrame[i][j][k] = targetFrame[i][j][k];	// transfer data from target to reference.
							targetFrame[i][j][k] = 0;						// zero the target array.
						}
					}
				}
				if (currBin < nBins[ch-1])	// if this is not the final bin.
				{
					while(inputParticles.get(idx).frame<binsize*currBin)
					{
						if (inputParticles.get(idx).include == 1)
						{
							targetFrame[(int)(inputParticles.get(idx).x/pixelSize)][(int)(inputParticles.get(idx).y/pixelSize)][(int)((inputParticles.get(idx).z+zOffset)/pixelSizeZ)] +=1;
							targetMean++;
						}

						idx++;
					}
				}else // if this is the final bin.
				{
					while(idx < inputParticles.size() && inputParticles.get(idx).channel==ch) // final batch, cover rounding errors.
					{
						if (inputParticles.get(idx).include == 1)
						{
							targetFrame[(int)(inputParticles.get(idx).x/pixelSize)][(int)(inputParticles.get(idx).y/pixelSize)][(int)((inputParticles.get(idx).z+zOffset)/pixelSizeZ)] +=1;
							targetMean++;
						}
						idx++;
					}
				}

				targetMean /= (dimensions[0]*dimensions[1]*dimensions[2]); // calculate new target mean.
				double refSquare = 0;										// this value needs to be calculated for every shift combination, pulled out and made accessible to save computational time.
				for (int i = 0; i < dimensions[0]; i++)
				{
					for (int j = 0; j < dimensions[1]; j++)
					{
						for (int k = 0; k < dimensions[2]; k++)
						{
							refSquare += (referenceFrame[i][j][k] - referenceMean)*( referenceFrame[i][j][k] - referenceMean); // calculate square pixel to mean difference.
						}
					}
				}
				//System.out.println(dimensions[2]);
				refSquare = Math.sqrt(refSquare);	// square root of difference.
				int[] shift = new int[3];			// will hold shift values in x-y-z.
				double[][] r = new double[4][8*maxShift*maxShift*maxShiftZ];	// result vector, xCorr-shiftX-shiftY-shiftZ.
				ImageCrossCorr3D xCorr = new ImageCrossCorr3D(referenceFrame,targetFrame,refSquare,referenceMean,targetMean); // create instance for crosscorrelation calculations between the current bins.
				ArrayList<int[]> allShifts 	= new ArrayList<int[]>();	// vector to hold all shift combinations for parallel computational setup.
				for (shift[0] = -maxShift; shift[0]< maxShift; shift[0]++)
				{
					for (shift[1] = -maxShift; shift[1]< maxShift; shift[1]++)
					{
						if (dimensions[2]> 1)
						{
							for (shift[2] = - maxShiftZ; shift[2] < maxShiftZ; shift[2]++)
							{
								int[] shiftAdd = {shift[0],shift[1],shift[2]};					
								allShifts.add(shiftAdd);
							}	
						}else
						{
							int[] shiftAdd = {shift[0],shift[1],0};					
							allShifts.add(shiftAdd);
						}

					}
				}				
				List<Callable<double[]>> tasks = new ArrayList<Callable<double[]>>();	// Preallocate.
				for (final int[] object : allShifts) {									// Loop over and setup computation.
					Callable<double[]> d = new Callable<double[]>() {					// Computation to be done.
						@Override
						public double[] call() throws Exception {
							return xCorr.xCorr3d(object);								// Actual call for each parallel process.
						}
					};
					tasks.add(d);														// Que this task.
				} // setup parallel computing, distribute all allShifts objects. 				
				int processors 			= Runtime.getRuntime().availableProcessors();	// Number of processor cores on this system.
				ExecutorService exec 	= Executors.newFixedThreadPool(processors);		// Set up parallel computing using all cores.
				//			double[] rPlot = new double[8*maxShift*maxShift*maxShiftZ];
				try {

					List<Future<double[]>> parallelCompute = exec.invokeAll(tasks);				// Execute computation.
					for (int i = 0; i < parallelCompute.size(); i++){							// Loop over and transfer results.
						try {										
							r[0][i] = parallelCompute.get(i).get()[0];							// Add computed results to r.
							r[1][i] = parallelCompute.get(i).get()[1];							// Add computed results to r.
							r[2][i] = parallelCompute.get(i).get()[2];							// Add computed results to r.
							r[3][i] = parallelCompute.get(i).get()[3];							// Add computed results to r.
							//					rPlot[i] = r[0][i];
						} catch (ExecutionException e) {
							e.printStackTrace();
						}
					}
				} catch (InterruptedException e) {

					e.printStackTrace();
				}
				finally {
					exec.shutdown();
				}		
				//	correctDrift.plot(rPlot);
				optimalShift[0][currBin-1] = 0;		// ensure that the current bin is set to 0.
				optimalShift[1][currBin-1] = 0;		// ensure that the current bin is set to 0.
				optimalShift[2][currBin-1] = 0;		// ensure that the current bin is set to 0.
				optimalShift[3][currBin-1] = 0;		// ensure that the current bin is set to 0.				

				for (int i = 0; i < r[0].length; i++) // loop over all results.
				{			
					if (optimalShift[0][currBin-1] < r[0][i]) // if we got a higher correlation then previously encountered.
					{
						optimalShift[0][currBin-1] = r[0][i]; // store values.
						optimalShift[1][currBin-1] = r[1][i]; // store values.
						optimalShift[2][currBin-1] = r[2][i]; // store values.
						optimalShift[3][currBin-1] = r[3][i]; // store values.
					}else if(optimalShift[0][currBin-1] == r[0][i] && // if we got the same correlation as previously but from a smaller shift.
							optimalShift[1][currBin-1] + optimalShift[2][currBin-1] + optimalShift[3][currBin-1]> r[1][i] + r[2][i] + r[3][i] && 
							optimalShift[1][currBin-1] >= r[1][i] &&
							optimalShift[2][currBin-1] >= r[2][i] &&
							optimalShift[3][currBin-1] >= r[3][i])
					{
						optimalShift[0][currBin-1] = r[0][i]; // store values.
						optimalShift[1][currBin-1] = r[1][i]; // store values.
						optimalShift[2][currBin-1] = r[2][i]; // store values.
						optimalShift[3][currBin-1] = r[3][i]; // store values.
					}
				}

				if (optimize) // reduce pixelsize and run again.
				{

					int increase = 2;
					byte[][][] referenceFrameFine 	= new byte[dimensions[0]*increase][dimensions[1]*increase][dimensions[2]*increase]; // finer pixelsize reference array.
					byte[][][] targetFrameFine 		= new byte[dimensions[0]*increase][dimensions[1]*increase][dimensions[2]*increase]; // finer pixelsize target array.
					int idxFine 					= tempIdx;	// temp index variable, start at top of this bin.
					double referenceMeanFine 		= 0;		// new mean calculation.
					double targetMeanFine 			= 0;		// new mean calculation.
					while(inputParticles.get(idxFine).frame<(binsize*(currBin-1))) //populate reference array.
					{
						if (inputParticles.get(idxFine).include == 1)
						{
							referenceFrameFine[(int)(inputParticles.get(idxFine).x/(pixelSize/increase))][(int)(inputParticles.get(idxFine).y/(pixelSize/increase))][(int)((inputParticles.get(idxFine).z+zOffset)/(pixelSizeZ/increase))] += 1;
							referenceMeanFine++;
						}
						idxFine++;
					}
					while(inputParticles.get(idxFine).frame<(binsize*currBin) && 
							idxFine < inputParticles.size() && inputParticles.get(idxFine).channel==ch) // populate target array.
					{
						if (inputParticles.get(idxFine).include == 1)
						{
							targetFrameFine[(int)(inputParticles.get(idxFine).x/(pixelSize/increase))][(int)(inputParticles.get(idxFine).y/(pixelSize/increase))][(int)((inputParticles.get(idxFine).z+zOffset)/(pixelSizeZ/increase))] += 1;
							targetMeanFine++;
						}
						idxFine++;
					}
					referenceMeanFine 	/= (dimensions[0]*dimensions[2]*dimensions[2]*increase*increase*increase); // calculate mean.
					targetMeanFine 		/= (dimensions[0]*dimensions[2]*dimensions[2]*increase*increase*increase); // calculate mean.
					double refSquareFine = 0;	// square difference to mean for reference frame.
					for (int i = 0; i < dimensions[0]*increase; i++)
					{
						for (int j = 0; j < dimensions[1]*increase; j++)
						{
							for (int k = 0; k < dimensions[2]*increase; k++)
							{
								refSquareFine += (referenceFrameFine[i][j][k] - referenceMeanFine)*( referenceFrameFine[i][j][k] - referenceMeanFine);
							}
						}
					}
					refSquareFine = Math.sqrt(refSquareFine);	// square root of square difference of mean to reference frame.	
					ImageCrossCorr3D xCorrFine = new ImageCrossCorr3D(referenceFrameFine,targetFrameFine,refSquareFine,referenceMeanFine,targetMeanFine); // create instance for crosscorrelation calculations between the current bins.					
					double[][] rFine = new double[4][8*increase*increase*increase];	// precast result array.
					ArrayList<int[]> allShiftsFine 	= new ArrayList<int[]>();		// hold all shift combinations for parallel computing.

					for (shift[0] = (int) ((optimalShift[1][currBin-1])*increase - increase); shift[0]< (int) ((optimalShift[1][currBin-1])*increase + increase); shift[0]++)
					{
						for (shift[1] = (int) ((optimalShift[2][currBin-1])*increase - increase); shift[1]< (int) ((optimalShift[2][currBin-1])*increase + increase); shift[1]++)
						{
							if (dimensions[2]> 1)
							{
								for (shift[2] = (int) ((optimalShift[3][currBin-1])*increase - increase); shift[2]< (int) ((optimalShift[3][currBin-1])*increase + increase); shift[2]++)
								{
									int[] shiftAdd = {shift[0],shift[1],shift[2]};				
									allShiftsFine.add(shiftAdd);								
								}	
							}else
							{
								int[] shiftAdd = {shift[0],shift[1],0};					
								allShiftsFine.add(shiftAdd);
							}
						}
					}			
					List<Callable<double[]>> tasksFine = new ArrayList<Callable<double[]>>();	// Preallocate.
					for (final int[] object : allShiftsFine) {									// Loop over and setup computation.
						Callable<double[]> d = new Callable<double[]>() {						// Computation to be done.
							@Override
							public double[] call() throws Exception {
								return xCorrFine.xCorr3d(object);								// Actual call for each parallel process.
							}
						};
						tasksFine.add(d);														// Que this task.
					} // setup parallel computing, distribute all allShiftsFine objects. 					

					ExecutorService execFine 	= Executors.newFixedThreadPool(processors);		// Set up parallel computing using all cores.
					try {

						List<Future<double[]>> parallelCompute = execFine.invokeAll(tasksFine);				// Execute computation.
						for (int i = 0; i < parallelCompute.size(); i++){							// Loop over and transfer results.
							try {										
								rFine[0][i] = parallelCompute.get(i).get()[0];							// Add computed results to rFine.
								rFine[1][i] = parallelCompute.get(i).get()[1];							// Add computed results to rFine.
								rFine[2][i] = parallelCompute.get(i).get()[2];							// Add computed results to rFine.
								rFine[3][i] = parallelCompute.get(i).get()[3];							// Add computed results to rFine.								

							} catch (ExecutionException e) {
								e.printStackTrace();
							}
						}
					} catch (InterruptedException e) {

						e.printStackTrace();
					}
					finally {
						execFine.shutdown();
					}
					optimalShift[0][currBin-1] = 0;	// reset values for this bin.
					optimalShift[1][currBin-1] = 0;	// reset values for this bin.
					optimalShift[2][currBin-1] = 0;	// reset values for this bin.
					optimalShift[3][currBin-1] = 0;	// reset values for this bin.

					for (int i = 0; i < rFine[0].length; i++) // loop over all shift combinations.
					{
						if (optimalShift[0][currBin-1] < rFine[0][i]) // if the current shift combination yielded a higher correlation.
						{
							optimalShift[0][currBin-1] = rFine[0][i];
							optimalShift[1][currBin-1] = rFine[1][i]/increase;
							optimalShift[2][currBin-1] = rFine[2][i]/increase;
							optimalShift[3][currBin-1] = rFine[3][i]/increase;
						}else if(optimalShift[0][currBin-1] == rFine[0][i] && // if the current shift combination yielded the same correlation but with smaller shift.
								optimalShift[1][currBin-1] + optimalShift[2][currBin-1] + optimalShift[3][currBin-1]> rFine[1][i]/increase + rFine[2][i]/increase + rFine[3][i]/increase &&
								optimalShift[1][currBin-1] >= rFine[1][i]/increase &&
								optimalShift[2][currBin-1] >= rFine[2][i]/increase &&
								optimalShift[3][currBin-1] >= rFine[3][i]/increase)
						{
							optimalShift[0][currBin-1] = rFine[0][i];
							optimalShift[1][currBin-1] = rFine[1][i]/increase;
							optimalShift[2][currBin-1] = rFine[2][i]/increase;
							optimalShift[3][currBin-1] = rFine[3][i]/increase;
						}
					}

				}
				currBin++;
			}			// bin loop.

			/*
			 * Interpolate and modify particles.
			 */
			double[] xAxis = new double[optimalShift[0].length];
			for(int j = 1; j < optimalShift[0].length; j++)
			{
				xAxis[j] = j*binsize;
				optimalShift[1][j] += optimalShift[1][j-1];
				optimalShift[2][j] += optimalShift[2][j-1];
				optimalShift[3][j] += optimalShift[3][j-1];
				//				System.out.println(pixelSize*optimalShift[1][j-1] + " to " +pixelSize*optimalShift[1][j]);
				//				System.out.println(pixelSize*optimalShift[2][j-1] + " to " +pixelSize*optimalShift[2][j]);
				//				System.out.println(optimalShift[3][j-1] + " to " +pixelSizeZ*optimalShift[3][j]);
				//				System.out.println(optimalShift[0][j]);
			}
			xAxis[0] = 0;
			xAxis[xAxis.length-1] = inputParticles.get(idx-1).frame;
			double[] xDrift = new double[xAxis.length];
			double[] yDrift = new double[xAxis.length];
			double[] zDrift = new double[xAxis.length];
			for (int i =1; i < xDrift.length; i++)
			{
				xDrift[i] = pixelSize*optimalShift[1][i];
				yDrift[i] = pixelSize*optimalShift[2][i];
				zDrift[i] = pixelSizeZ*optimalShift[3][i];
			}
			correctDrift.plot(xDrift, yDrift, zDrift, xAxis);


			int bin = 0;
			int i = tempIdx; 
			while (i < inputParticles.size() && inputParticles.get(i).channel == ch)// && inputParticles.get(i).frame < (bin+1)*binsize) // apply shifts.
			{
				Particle tempPart 	= new Particle();
				tempPart.frame	 	= inputParticles.get(i).frame;
				tempPart.r_square 	= inputParticles.get(i).r_square;
				tempPart.photons 	= inputParticles.get(i).photons;
				tempPart.include 	= inputParticles.get(i).include;
				tempPart.precision_x= inputParticles.get(i).precision_x;
				tempPart.precision_y= inputParticles.get(i).precision_y;
				tempPart.precision_z= inputParticles.get(i).precision_z;
				tempPart.sigma_x 	= inputParticles.get(i).sigma_x;
				tempPart.sigma_y 	= inputParticles.get(i).sigma_y;
				tempPart.channel 	= inputParticles.get(i).channel;
				if ((bin+1)*binsize <= inputParticles.get(i).frame)										
					bin++;						

				if (bin == nBins[ch-1]) // load last fragment of data.
					bin--;

				tempPart.x = inputParticles.get(i).x + pixelSize*optimalShift[1][bin];
				tempPart.y = inputParticles.get(i).y + pixelSize*optimalShift[2][bin];				
				tempPart.z = inputParticles.get(i).z + pixelSizeZ*optimalShift[3][bin];

				if(tempPart.x >= 0){
					if(tempPart.y >= 0){

						shiftedParticles.add(tempPart);
					}
				}	
				//	System.out.println(tempPart.x + " " + tempPart.y + " " + tempPart.z);
				i++;				
			}
		}  // loop over channels.
		ij.IJ.log("drift correction complete");
		return shiftedParticles;
	}


	public static ArrayList<Particle> runChannel(ArrayList<Particle> inputParticles, int[][] boundry, int[] dimensions,int pixelSize, int pixelSizeZ, boolean optimize) // main function call. Will per channel autocorrelate between time binned data.
	{
		ArrayList<Particle> shiftedParticles = new ArrayList<Particle>();

		int nChannels 	= inputParticles.get(inputParticles.size()-1).channel; // data is sorted on a per channel basis.
		int zOffset 	= 0;
		for (int i = 0; i < inputParticles.size();i++)
		{
			if (zOffset > Math.floor(inputParticles.get(i).z))
				zOffset = (int) Math.floor(inputParticles.get(i).z);
		}
		zOffset = -zOffset;
		if (nChannels >= 2)
		{
			int idx 		= 0; // index for inputParticles.
			byte[][][] referenceFrame 	= new byte[dimensions[0]][dimensions[1]][dimensions[2]]; // create the reference array.
			byte[][][] targetFrame 	= new byte[dimensions[0]][dimensions[1]][dimensions[2]]; 		// create the target (shifted) array.

			double[][] optimalShift = new double[4][nChannels];	// this vector will hold all shift values.
			double targetMean = 0;	// mean value of target array.
			double referenceMean = 0;	// mean value of reference array.
			while(inputParticles.get(idx).channel == 1) // populate target first due to while loop design.
			{
				if (inputParticles.get(idx).include == 1)	// only include particles that are ok from the users parameter choice.
				{
					targetFrame[(int)(inputParticles.get(idx).x/pixelSize)][(int)(inputParticles.get(idx).y/pixelSize)][(int)((inputParticles.get(idx).z+zOffset)/pixelSizeZ)] += 1;	// increase value by one if there is a particle within the voxel.
					targetMean++;	// keep track of total number of added particles (for mean calculation).
				}
				idx++;	// step forward.
			}
			targetMean /= (dimensions[0]*dimensions[1]*dimensions[2]);	// calculate the mean.

			for (int ch = 2; ch <= nChannels; ch++) // loop over ch2:final channel
			{
				//int tempIdx 	= idx; // start of current channel.
				int maxShift 	= boundry[0][ch-1];
				int maxShiftZ 	= boundry[1][ch-1];
				referenceMean = targetMean;							// last loops target is now the reference.
				targetMean = 0;										// new target array, set old values to 0.
				for (int i = 0; i < targetFrame.length; i++)		// loop over the entire target array.
				{
					for(int j = 0; j < targetFrame[0].length; j++)
					{
						for (int k = 0; k < targetFrame[0][0].length; k++)
						{
							referenceFrame[i][j][k] = targetFrame[i][j][k];	// transfer data from target to reference.
							targetFrame[i][j][k] = 0;						// zero the target array.
						}
					}
				}

				while(idx < inputParticles.size() && inputParticles.get(idx).channel == ch)
				{
					if (inputParticles.get(idx).include == 1)
					{
						targetFrame[(int)(inputParticles.get(idx).x/pixelSize)][(int)(inputParticles.get(idx).y/pixelSize)][(int)((inputParticles.get(idx).z+zOffset)/pixelSizeZ)] +=1;
						targetMean++;
					}

					idx++;
				}
				targetMean /= (dimensions[0]*dimensions[1]*dimensions[2]);	// calculate the mean.
				double refSquare = 0;										// this value needs to be calculated for every shift combination, pulled out and made accessible to save computational time.
				for (int i = 0; i < dimensions[0]; i++)
				{
					for (int j = 0; j < dimensions[1]; j++)
					{
						for (int k = 0; k < dimensions[2]; k++)
						{
							refSquare += (referenceFrame[i][j][k] - referenceMean)*( referenceFrame[i][j][k] - referenceMean); // calculate square pixel to mean difference.
						}
					}
				}
				refSquare = Math.sqrt(refSquare);	// square root of difference.
				int[] shift = new int[3];			// will hold shift values in x-y-z.
				double[][] r = new double[4][8*maxShift*maxShift*maxShiftZ];	// result vector, xCorr-shiftX-shiftY-shiftZ.
				ImageCrossCorr3D xCorr = new ImageCrossCorr3D(referenceFrame,targetFrame,refSquare,referenceMean,targetMean); // create instance for crosscorrelation calculations between the current bins.
				ArrayList<int[]> allShifts 	= new ArrayList<int[]>();	// vector to hold all shift combinations for parallel computational setup.
				for (shift[0] = -maxShift; shift[0]< maxShift; shift[0]++)
				{
					for (shift[1] = -maxShift; shift[1]< maxShift; shift[1]++)
					{
						if (dimensions[2]> 1)
						{
							for (shift[2] = - maxShiftZ; shift[2] < maxShiftZ; shift[2]++)
							{
								int[] shiftAdd = {shift[0],shift[1],shift[2]};									
								allShifts.add(shiftAdd);
							}
						}else
						{
							int[] shiftAdd = {shift[0],shift[1],0};					
							allShifts.add(shiftAdd);
						}
					}
				}				
				List<Callable<double[]>> tasks = new ArrayList<Callable<double[]>>();	// Preallocate.
				for (final int[] object : allShifts) {									// Loop over and setup computation.
					Callable<double[]> d = new Callable<double[]>() {					// Computation to be done.
						@Override
						public double[] call() throws Exception {
							return xCorr.xCorr3d(object);								// Actual call for each parallel process.
						}
					};
					tasks.add(d);														// Que this task.
				} // setup parallel computing, distribute all allShifts objects. 				
				int processors 			= Runtime.getRuntime().availableProcessors();	// Number of processor cores on this system.
				ExecutorService exec 	= Executors.newFixedThreadPool(processors);		// Set up parallel computing using all cores.
				//			double[] rPlot = new double[8*maxShift*maxShift*maxShiftZ];
				try {

					List<Future<double[]>> parallelCompute = exec.invokeAll(tasks);				// Execute computation.
					for (int i = 0; i < parallelCompute.size(); i++){							// Loop over and transfer results.
						try {										
							r[0][i] = parallelCompute.get(i).get()[0];							// Add computed results to r.
							r[1][i] = parallelCompute.get(i).get()[1];							// Add computed results to r.
							r[2][i] = parallelCompute.get(i).get()[2];							// Add computed results to r.
							r[3][i] = parallelCompute.get(i).get()[3];							// Add computed results to r.
							//					rPlot[i] = r[0][i];
						} catch (ExecutionException e) {
							e.printStackTrace();
						}
					}
				} catch (InterruptedException e) {

					e.printStackTrace();
				}
				finally {
					exec.shutdown();
				}	
				//						correctDrift.plot(rPlot);
				optimalShift[0][ch-1] = 0;		// ensure that the current bin is set to 0.
				optimalShift[1][ch-1] = 0;		// ensure that the current bin is set to 0.
				optimalShift[2][ch-1] = 0;		// ensure that the current bin is set to 0.
				optimalShift[3][ch-1] = 0;		// ensure that the current bin is set to 0.				

				for (int i = 0; i < r[0].length; i++) // loop over all results.
				{			
					if (optimalShift[0][ch-1] < r[0][i]) // if we got a higher correlation then previously encountered.
					{
						optimalShift[0][ch-1] = r[0][i]; // store values.
						optimalShift[1][ch-1] = r[1][i]; // store values.
						optimalShift[2][ch-1] = r[2][i]; // store values.
						optimalShift[3][ch-1] = r[3][i]; // store values.
					}else if(optimalShift[0][ch-1] == r[0][i] && // if we got the same correlation as previously but from a smaller shift.
							optimalShift[1][ch-1] + optimalShift[2][ch-1] + optimalShift[3][ch-1]> r[1][i] + r[2][i] + r[3][i] && 
							optimalShift[1][ch-1] >= r[1][i] &&
							optimalShift[2][ch-1] >= r[2][i] &&
							optimalShift[3][ch-1] >= r[3][i])
					{
						optimalShift[0][ch-1] = r[0][i]; // store values.
						optimalShift[1][ch-1] = r[1][i]; // store values.
						optimalShift[2][ch-1] = r[2][i]; // store values.
						optimalShift[3][ch-1] = r[3][i]; // store values.
					}
				}
				/*			System.out.println(optimalShift[1][0] + " to " +optimalShift[1][1]);
				System.out.println(optimalShift[2][0] + " to " +optimalShift[2][1]);
				System.out.println(optimalShift[3][0] + " to " +optimalShift[3][1]);
				System.out.println(optimalShift[0][1]);
				 */		if (optimize) // reduce pixelsize and run again.
				 {
					 int increase = 2;
					 byte[][][] referenceFrameFine 	= new byte[dimensions[0]*increase][dimensions[1]*increase][dimensions[2]*increase]; // finer pixelsize reference array.
					 byte[][][] targetFrameFine 		= new byte[dimensions[0]*increase][dimensions[1]*increase][dimensions[2]*increase]; // finer pixelsize target array.
					 int idxFine 					= 0;	// start loop from teh beginning.
					 double referenceMeanFine 		= 0;		// new mean calculation.
					 double targetMeanFine 			= 0;		// new mean calculation.
					 while(inputParticles.get(idxFine).channel <= ch-1) //populate reference array.
					 {
						 if (inputParticles.get(idxFine).include == 1 && inputParticles.get(idxFine).channel == ch-1)
						 {
							 referenceFrameFine[(int)(inputParticles.get(idxFine).x/(pixelSize/increase))][(int)(inputParticles.get(idxFine).y/(pixelSize/increase))][(int)((inputParticles.get(idxFine).z+zOffset)/(pixelSizeZ/increase))] += 1;
							 referenceMeanFine++;
						 }
						 idxFine++;
					 }
					 while(idxFine < inputParticles.size() && inputParticles.get(idxFine).channel==ch) // populate target array.
					 {
						 if (inputParticles.get(idxFine).include == 1)
						 {
							 targetFrameFine[(int)(inputParticles.get(idxFine).x/(pixelSize/increase))][(int)(inputParticles.get(idxFine).y/(pixelSize/increase))][(int)((inputParticles.get(idxFine).z+zOffset)/(pixelSizeZ/increase))] += 1;
							 targetMeanFine++;
						 }
						 idxFine++;
					 }
					 referenceMeanFine 	/= (dimensions[0]*dimensions[2]*dimensions[2]*increase*increase*increase); // calculate mean.
					 targetMeanFine 		/= (dimensions[0]*dimensions[2]*dimensions[2]*increase*increase*increase); // calculate mean.
					 double refSquareFine = 0;	// square difference to mean for reference frame.
					 for (int i = 0; i < dimensions[0]*increase; i++)
					 {
						 for (int j = 0; j < dimensions[1]*increase; j++)
						 {
							 for (int k = 0; k < dimensions[2]*increase; k++)
							 {
								 refSquareFine += (referenceFrameFine[i][j][k] - referenceMeanFine)*( referenceFrameFine[i][j][k] - referenceMeanFine);
							 }
						 }
					 }
					 refSquareFine = Math.sqrt(refSquareFine);	// square root of square difference of mean to reference frame.	
					 ImageCrossCorr3D xCorrFine = new ImageCrossCorr3D(referenceFrameFine,targetFrameFine,refSquareFine,referenceMeanFine,targetMeanFine); // create instance for crosscorrelation calculations between the current bins.					
					 double[][] rFine = new double[4][8*increase*increase*increase];	// precast result array.
					 ArrayList<int[]> allShiftsFine 	= new ArrayList<int[]>();		// hold all shift combinations for parallel computing.

					 for (shift[0] = (int) ((optimalShift[1][ch-1])*increase - increase); shift[0]< (int) ((optimalShift[1][ch-1])*increase + increase); shift[0]++)
					 {
						 for (shift[1] = (int) ((optimalShift[2][ch-1])*increase - increase); shift[1]< (int) ((optimalShift[2][ch-1])*increase + increase); shift[1]++)
						 {
							 if (dimensions[2]> 1)
							 {
								 for (shift[2] = (int) ((optimalShift[3][ch-1])*increase - increase); shift[2]< (int) ((optimalShift[3][ch-1])*increase + increase); shift[2]++)
								 {
									 int[] shiftAdd = {shift[0],shift[1],shift[2]};				
									 allShiftsFine.add(shiftAdd);
								 }
							 }else
							 {
								 int[] shiftAdd = {shift[0],shift[1],0};					
								 allShiftsFine.add(shiftAdd);
							 }
						 }
					 }			
					 List<Callable<double[]>> tasksFine = new ArrayList<Callable<double[]>>();	// Preallocate.
					 for (final int[] object : allShiftsFine) {									// Loop over and setup computation.
						 Callable<double[]> d = new Callable<double[]>() {						// Computation to be done.
							 @Override
							 public double[] call() throws Exception {
								 return xCorrFine.xCorr3d(object);								// Actual call for each parallel process.
							 }
						 };
						 tasksFine.add(d);														// Que this task.
					 } // setup parallel computing, distribute all allShiftsFine objects. 					

					 ExecutorService execFine 	= Executors.newFixedThreadPool(processors);		// Set up parallel computing using all cores.
					 try {

						 List<Future<double[]>> parallelCompute = execFine.invokeAll(tasksFine);				// Execute computation.
						 for (int i = 0; i < parallelCompute.size(); i++){							// Loop over and transfer results.
							 try {										
								 rFine[0][i] = parallelCompute.get(i).get()[0];							// Add computed results to rFine.
								 rFine[1][i] = parallelCompute.get(i).get()[1];							// Add computed results to rFine.
								 rFine[2][i] = parallelCompute.get(i).get()[2];							// Add computed results to rFine.
								 rFine[3][i] = parallelCompute.get(i).get()[3];							// Add computed results to rFine.								

							 } catch (ExecutionException e) {
								 e.printStackTrace();
							 }
						 }
					 } catch (InterruptedException e) {

						 e.printStackTrace();
					 }
					 finally {
						 execFine.shutdown();
					 }
					 optimalShift[0][ch-1] = 0;	// reset values for this bin.
					 optimalShift[1][ch-1] = 0;	// reset values for this bin.
					 optimalShift[2][ch-1] = 0;	// reset values for this bin.
					 optimalShift[3][ch-1] = 0;	// reset values for this bin.

					 for (int i = 0; i < rFine[0].length; i++) // loop over all shift combinations.
					 {
						 if (optimalShift[0][ch-1] < rFine[0][i]) // if the current shift combination yielded a higher correlation.
						 {
							 optimalShift[0][ch-1] = rFine[0][i];
							 optimalShift[1][ch-1] = rFine[1][i]/increase;
							 optimalShift[2][ch-1] = rFine[2][i]/increase;
							 optimalShift[3][ch-1] = rFine[3][i]/increase;
						 }else if(optimalShift[0][ch-1] == rFine[0][i] && // if the current shift combination yielded the same correlation but with smaller shift.
								 optimalShift[1][ch-1] + optimalShift[2][ch-1] + optimalShift[3][ch-1]> rFine[1][i]/increase + rFine[2][i]/increase + rFine[3][i]/increase &&
								 optimalShift[1][ch-1] >= rFine[1][i]/increase &&
								 optimalShift[2][ch-1] >= rFine[2][i]/increase &&
								 optimalShift[3][ch-1] >= rFine[3][i]/increase)
						 {
							 optimalShift[0][ch-1] = rFine[0][i];
							 optimalShift[1][ch-1] = rFine[1][i]/increase;
							 optimalShift[2][ch-1] = rFine[2][i]/increase;
							 optimalShift[3][ch-1] = rFine[3][i]/increase;
						 }
					 }

				 }

			} // loop over channel 2:end
			for(int j = 1; j < optimalShift[0].length; j++)
			{
				optimalShift[1][j] += optimalShift[1][j-1];
				optimalShift[2][j] += optimalShift[2][j-1];
				optimalShift[3][j] += optimalShift[3][j-1];
				ij.IJ.log("Channel " + (j+1) + " shifted by " + pixelSize*optimalShift[1][j]+  " x " + pixelSize*optimalShift[2][j] + " x " + pixelSizeZ*optimalShift[3][j] + " nm.");
			}
			/*		System.out.println(optimalShift[1][0] + " to " +pixelSize*optimalShift[1][1]);
			System.out.println(optimalShift[2][0] + " to " +pixelSize*optimalShift[2][1]);
			System.out.println(optimalShift[3][0] + " to " +pixelSizeZ*optimalShift[3][1]);
			System.out.println(optimalShift[0][1]);
			 */			
			int i = 0; 
			while (i < inputParticles.size()) // apply shifts.
			{
				Particle tempPart 	= new Particle();
				tempPart.frame	 	= inputParticles.get(i).frame;
				tempPart.r_square 	= inputParticles.get(i).r_square;
				tempPart.photons 	= inputParticles.get(i).photons;
				tempPart.include 	= inputParticles.get(i).include;
				tempPart.precision_x= inputParticles.get(i).precision_x;
				tempPart.precision_y= inputParticles.get(i).precision_y;
				tempPart.precision_z= inputParticles.get(i).precision_z;
				tempPart.sigma_x 	= inputParticles.get(i).sigma_x;
				tempPart.sigma_y 	= inputParticles.get(i).sigma_y;
				tempPart.channel 	= inputParticles.get(i).channel;

				tempPart.x = inputParticles.get(i).x + pixelSize*optimalShift[1][inputParticles.get(i).channel-1];
				tempPart.y = inputParticles.get(i).y + pixelSize*optimalShift[2][inputParticles.get(i).channel-1];				
				tempPart.z = inputParticles.get(i).z + pixelSizeZ*optimalShift[3][inputParticles.get(i).channel-1];

				if(tempPart.x >= 0){
					if(tempPart.y >= 0){

						shiftedParticles.add(tempPart);
					}
				}								
				i++;
			}			
			ij.IJ.log("Channels aligned.");		
			return shiftedParticles;
		} // check for multichannel image.
		else
			return inputParticles;
	} // runChannel

	public static void main(String[] args) {

		ArrayList<Particle> result = new ArrayList<Particle>();
		int frame = 1;

		Random r = new Random();
		for (int i = 0; i < 5000; i++)
		{
			Particle p = new Particle();
			p.include = 1;
			p.x = r.nextDouble()*1000;
			p.y = r.nextDouble()*1000;
			//		p.z = r.nextDouble()*400-300;
			if (p.x < 100)
				p.x = 100;
			if (p.y < 100)
				p.y = 100;
			//		if (p.z < 100)
			//			p.z = 100;
			p.channel = 1;
			p.frame= frame;
			result.add(p);
			frame++;
		}
		for (int i = 0; i < 5000; i++)
		{
			Particle p = result.get(i);
			Particle p2 = new Particle();
			p2.include = 1;
			p2.x = p.x + 60;
			p2.y = p.y - 50;
			//		p2.z = p.z;
			p2.channel = 1;
			p2.frame = frame;
			result.add(p2);
			frame++;
		}

		int[][] maxShift = new int[2][2]; //xy-z per channel

		int[] nBins = {2};
		int pixelSize = 10;
		int pixelSizeZ = 20;
		maxShift[0][0] = 100/pixelSize;
		maxShift[1][0] = 100/pixelSizeZ;
		maxShift[0][1] = 100/pixelSize;
		maxShift[1][1] = 100/pixelSizeZ;
		int[] size = {12800/pixelSize, 12800/pixelSize, 1000/pixelSizeZ};
		size[2] = 1; // if sending in 2D data, send in with zdim = 1.
		long time = System.nanoTime();
		result = run(result, nBins, maxShift,size ,pixelSize,pixelSizeZ,true);
		//result = runChannel(result,  maxShift,size ,pixelSize,pixelSizeZ,true);
		time = System.nanoTime() - time;
		System.out.println(time*1E-9);	
	}


	public double[] xCorr3d (int[] shift) // calculate crosscorrelation between two 2D arrays for the given x-y shift.
	{
		int startX = 0;
		int endX = 0;
		int startY = 0;
		int endY = 0;
		int startZ = 0;
		int endZ = 0;
		if (shift[0]>= 0)
		{
			startX = 0 + shift[0];
			endX = reference.length;
		}else
		{
			startX = 0;
			endX = reference.length + shift[0];
		}
		if (shift[1]>= 0)
		{
			startY = 0 + shift[1];
			endY = reference[0].length;			

		}else
		{			
			startY = 0;
			endY = reference[0].length + shift[1];
		}
		if (shift[2] >= 0)
		{
			startZ = 0 + shift[2];
			endZ = reference[0][0].length;
		}else
		{
			startZ = 0;
			endZ = reference[0][0].length + shift[2];
		}

		double union = 0;
		double shiftSquare = 0;		
		for (int xi = startX; xi < endX; xi++)
		{
			for (int yi = startY; yi < endY; yi++)
			{		
				for (int zi = startZ; zi < endZ; zi++)
				{
					union += (reference[xi][yi][zi] - refMean)*(target[xi-shift[0]][yi-shift[1]][zi-shift[2]]-tarMean);				
					shiftSquare += (target[xi-shift[0]][yi-shift[1]][zi-shift[2]]-tarMean)*(target[xi-shift[0]][yi-shift[1]][zi-shift[2]]-tarMean);
				}
			}			
		}	
		double[] r = {((union / (refSquare*Math.sqrt(shiftSquare)))),shift[0],shift[1],shift[2]};
		return r;
	}

}