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
	static int[][][] reference;
	static int[][][] target;
	static double refSquare;
	static double tarMean;
	static double refMean;

	public ImageCrossCorr3D(int[][][] constructReference, int[][][] constructTarget, double constructRefSquare, double constructRefMean, double constructTarMean)
	{
		ImageCrossCorr3D.reference 	= constructReference;		
		ImageCrossCorr3D.target 	= constructTarget;
		ImageCrossCorr3D.refSquare	= constructRefSquare;
		ImageCrossCorr3D.refMean 	= constructRefMean;
		ImageCrossCorr3D.tarMean 	= constructTarMean;
	}

	public static ArrayList<Particle> run(ArrayList<Particle> inputParticles, int[] nBins, int[][] boundry, int[] dimensions,int pixelSize, int pixelSizeZ) // main function call. Will per channel autocorrelate between time binned data.
	{
		ArrayList<Particle> shiftedParticles = new ArrayList<Particle>();


		int nChannels 	= inputParticles.get(inputParticles.size()-1).channel; // data is sorted on a per channel basis.
		int idx 		= 0; // index for inputParticles.
		double zOffset 	= 0;
		for (int i = 0; i < inputParticles.size();i++)
		{
			if (zOffset > Math.floor(inputParticles.get(i).z))
				zOffset = Math.floor(inputParticles.get(i).z);
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
			int currBin = 2;	// start with bin 2.
			idx = tempIdx;
			double[][] optimalShift = new double[4][nBins[ch-1]];	// this vector will hold all shift values.
			while (currBin <= nBins[ch-1])							// loop over all bins.
			{


				idx = tempIdx;
				int[] firstBin = {idx,1};
				int[] secondBin = {idx,1};

				while (idx < inputParticles.size() &&
						inputParticles.get(idx).channel == ch &&
						inputParticles.get(idx).frame <= (currBin-1) * binsize)
				{
					firstBin[1] = idx;
					idx++;
				}
				secondBin[0] = idx;
				while (idx < inputParticles.size() &&
						inputParticles.get(idx).channel == ch &&
						inputParticles.get(idx).frame <= currBin * binsize)
				{
					secondBin[1] = idx;
					idx++;
				}
				if (currBin == nBins[ch-1])
				{
					while (idx < inputParticles.size() &&
							inputParticles.get(idx).channel == ch)
					{
						secondBin[1] = idx;
						idx++;
					}
				}

				int[][] c = getOverlapCenter(inputParticles, firstBin, secondBin, dimensions, pixelSize,  pixelSizeZ, maxShift, maxShiftZ,zOffset);
				int[] croppedDimensions = {(int)Math.ceil((c[0][1]-c[0][0])/pixelSize),(int)Math.ceil((c[1][1]-c[1][0])/pixelSize),(int)Math.ceil((c[2][1]-c[2][0])/pixelSizeZ)};
				idx = tempIdx;	// return idx to startpoint of this loop.
				tempIdx = secondBin[0];

				int[][][] referenceFrame 	= new int[croppedDimensions[0]][croppedDimensions[1]][croppedDimensions[2]]; // create the reference array.
				int[][][] targetFrame 		= new int[croppedDimensions[0]][croppedDimensions[1]][croppedDimensions[2]]; 		// create the target (shifted) array.	//					
				double targetMean = 0;	// mean value of target array.
				double referenceMean = 0;	// mean value of reference array.

				while(inputParticles.get(idx).frame<=(currBin - 1) * binsize) // populate target first due to while loop design.
				{
					if (inputParticles.get(idx).include == 1)	// only include particles that are ok from the users parameter choice.
					{
						if (inputParticles.get(idx).x >= c[0][0] && inputParticles.get(idx).x < c[0][1] &&
								inputParticles.get(idx).y >= c[1][0] && inputParticles.get(idx).y < c[1][1] &&
								inputParticles.get(idx).z >= c[2][0] && inputParticles.get(idx).z < c[2][1])
						{
							int x = (int)Math.floor((inputParticles.get(idx).x-c[0][0])/pixelSize);
							int y = (int)Math.floor((inputParticles.get(idx).y-c[1][0])/pixelSize);
							int z = (int)Math.floor((inputParticles.get(idx).z-c[2][0])/pixelSizeZ);
							referenceFrame[x][y][z]++;
							referenceMean++;	// keep track of total number of added particles (for mean calculation).
						}
					}
					idx++;	// step forward.
				}
				referenceMean /= (croppedDimensions[0]*croppedDimensions[1]*croppedDimensions[2]);	// calculate the mean.
				if (currBin < nBins[ch-1])	// if this is not the final bin.
				{
					while(inputParticles.get(idx).frame<= currBin * binsize) // populate target first due to while loop design.
					{

						if (inputParticles.get(idx).include == 1)	// only include particles that are ok from the users parameter choice.
						{
							if (inputParticles.get(idx).x >= c[0][0] && inputParticles.get(idx).x < c[0][1] &&
									inputParticles.get(idx).y >= c[1][0] && inputParticles.get(idx).y < c[1][1] &&
									inputParticles.get(idx).z >= c[2][0] && inputParticles.get(idx).z < c[2][1])
							{
								int x = (int)Math.floor((inputParticles.get(idx).x-c[0][0])/pixelSize);
								int y = (int)Math.floor((inputParticles.get(idx).y-c[1][0])/pixelSize);
								int z = (int)Math.floor((inputParticles.get(idx).z-c[2][0])/pixelSizeZ);
								targetFrame[x][y][z]++;
								targetMean++;	// keep track of total number of added particles (for mean calculation).
							}
						}
						idx++;	// step forward.
					}
				}
				else
				{
					while(idx < inputParticles.size() && inputParticles.get(idx).channel==ch) // final batch, cover rounding errors.
					{

						if (inputParticles.get(idx).include == 1)	// only include particles that are ok from the users parameter choice.
						{
							if (inputParticles.get(idx).x >= c[0][0] && inputParticles.get(idx).x < c[0][1] &&
									inputParticles.get(idx).y >= c[1][0] && inputParticles.get(idx).y < c[1][1] &&
									inputParticles.get(idx).z >= c[2][0] && inputParticles.get(idx).z < c[2][1])
							{
								int x = (int)Math.floor((inputParticles.get(idx).x-c[0][0])/pixelSize);
								int y = (int)Math.floor((inputParticles.get(idx).y-c[1][0])/pixelSize) ;
								int z = (int)Math.floor((inputParticles.get(idx).z-c[2][0])/pixelSizeZ);
								targetFrame[x][y][z]++;
								targetMean++;	// keep track of total number of added particles (for mean calculation).
							}
						}
						idx++;	// step forward.
					}
				} // last bin.
				targetMean /= (croppedDimensions[0]*croppedDimensions[1]*croppedDimensions[2]); // calculate new target mean.
				/*	double refSquare = 0;										// this value needs to be calculated for every shift combination, pulled out and made accessible to save computational time.
				for (int i = 0; i < croppedDimensions[0]; i++)
				{
					for (int j = 0; j < croppedDimensions[1]; j++)
					{
						for (int k = 0; k < croppedDimensions[2]; k++)
						{
							refSquare += (referenceFrame[i][j][k] - referenceMean)*( referenceFrame[i][j][k] - referenceMean); // calculate square pixel to mean difference.
						}
					}
				}
				refSquare = Math.sqrt(refSquare);	// square root of difference.*/
				double refSquare = 0;										// this value needs to be calculated for every shift combination, pulled out and made accessible to save computational time.
				double tarSquare = 0;										// this value needs to be calculated for every shift combination, pulled out and made accessible to save computational time.


				for (int i = 0; i < croppedDimensions[0]; i++)
				{
					for (int j = 0; j < croppedDimensions[1]; j++)
					{
						for (int k = 0; k < croppedDimensions[2]; k++)
						{
							refSquare += (referenceFrame[i][j][k] - referenceMean)*( referenceFrame[i][j][k] - referenceMean); // calculate square pixel to mean difference.
							tarSquare += (targetFrame[i][j][k] - targetMean)*( targetFrame[i][j][k] - targetMean); // calculate square pixel to mean difference.
						}
					}
				}

				if (refSquare > 0 && tarSquare > 0)
				{

					refSquare = Math.sqrt(refSquare);	// square root of difference.
					tarSquare = Math.sqrt(tarSquare);	// square root of difference.
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

					int[] center = {0,0,0};
					double[] noDrift = xCorr.xCorr3d(center);
					//				System.out.println(noDrift[0]/(tarSquare*refSquare));
					//	correctDrift.plot(rPlot);
					optimalShift[0][currBin-1] = noDrift[0]/(tarSquare*refSquare);		// ensure that the current bin is set to the non shifted correlation.
					optimalShift[1][currBin-1] = 0;		// ensure that the current bin is set to 0.
					optimalShift[2][currBin-1] = 0;		// ensure that the current bin is set to 0.
					optimalShift[3][currBin-1] = 0;		// ensure that the current bin is set to 0.				

					for (int i = 0; i < r[0].length; i++) // loop over all results.
					{			
						if (r[0][i]/(tarSquare*refSquare) > 0.2 && optimalShift[0][currBin-1] < r[0][i]/(tarSquare*refSquare)) // if we got a higher correlation then previously encountered.
						{
							optimalShift[0][currBin-1] = r[0][i]/(tarSquare*refSquare); // store values.
							optimalShift[1][currBin-1] = r[1][i]; // store values.
							optimalShift[2][currBin-1] = r[2][i]; // store values.
							optimalShift[3][currBin-1] = r[3][i]; // store values.
						}else if(optimalShift[0][currBin-1] == r[0][i] && // if we got the same correlation as previously but from a smaller shift.
								optimalShift[1][currBin-1] + optimalShift[2][currBin-1] + optimalShift[3][currBin-1]> r[1][i] + r[2][i] + r[3][i] && 
								optimalShift[1][currBin-1] >= r[1][i] &&
								optimalShift[2][currBin-1] >= r[2][i] &&
								optimalShift[3][currBin-1] >= r[3][i])
						{
							optimalShift[0][currBin-1] = r[0][i]/(tarSquare*refSquare); // store values.
							optimalShift[1][currBin-1] = r[1][i]; // store values.
							optimalShift[2][currBin-1] = r[2][i]; // store values.
							optimalShift[3][currBin-1] = r[3][i]; // store values.
						}
					}
				} // end check for any entry in bin.
				else
				{
					optimalShift[0][currBin-1] = 0;
					optimalShift[1][currBin-1] = 0;
					optimalShift[2][currBin-1] = 0;
					optimalShift[3][currBin-1] = 0;
				}
				//	System.out.println(currBin + ": " +optimalShift[0][currBin-1]);
				currBin++;
			} // bin loop.

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
				/*		System.out.println(pixelSize*optimalShift[1][j-1] + " to " +pixelSize*optimalShift[1][j]);
								System.out.println(pixelSize*optimalShift[2][j-1] + " to " +pixelSize*optimalShift[2][j]);
								System.out.println(optimalShift[3][j-1] + " to " +pixelSizeZ*optimalShift[3][j]);
								System.out.println(optimalShift[0][j]);*/
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
			int i = 0; 
			while (inputParticles.size() > i && inputParticles.get(i).channel > ch)
				i++;
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

				if ((bin+1)*binsize + 1 <= inputParticles.get(i).frame)		
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


	public static ArrayList<Particle> runChannel(ArrayList<Particle> inputParticles, int[][] boundry, int[] dimensions,int pixelSize, int pixelSizeZ) // main function call. Will per channel autocorrelate between time binned data.
	{
		ArrayList<Particle> shiftedParticles = new ArrayList<Particle>();
		int nChannels 	= inputParticles.get(inputParticles.size()-1).channel; // data is sorted on a per channel basis.
		double zOffset 	= 0;
		for (int i = 0; i < inputParticles.size();i++)
		{
			if (zOffset > Math.floor(inputParticles.get(i).z))
				zOffset = Math.floor(inputParticles.get(i).z);
		}
		zOffset = -zOffset;
		if (nChannels >= 2)
		{			
			double[][] optimalShift = new double[4][nChannels];	// this vector will hold all shift values.
			int idx 		= 0; // index for inputParticles.

			int[] firstBin = {idx,1};
			while (idx < inputParticles.size() &&
					inputParticles.get(idx).channel == 1)
			{
				firstBin[1] = idx;
				idx++;
			}	
			//int tempIdx 	= idx; // hold value

			for (int ch = 2; ch <= nChannels; ch++)
			{

				int maxShift 	= boundry[0][ch-1];
				int maxShiftZ 	= boundry[1][ch-1];
				//		idx = tempIdx;
				int[] secondBin = {idx,1};


				/*	while (idx < inputParticles.size() && inputParticles.get(idx).channel < ch)
					idx++;
				secondBin[0] = idx;*/
				while (idx < inputParticles.size() &&
						inputParticles.get(idx).channel == ch)
				{
					secondBin[1] = idx;
					idx++;
				}
				/*
				 * Make 256x256x20 image or less, going for voxel size of 2*maxDrift x 2*maxDrift * 2*maxDriftZ. 
				 * Find pixel with largest correlation with the next bin and extract a 10x10x20 nm voxel 256x256x20 pixel large ROI and apply the drift correction from this ROI to all particles.
				 */

				int[][] c = getOverlapCenter(inputParticles, firstBin, secondBin, dimensions, pixelSize,  pixelSizeZ, maxShift, maxShiftZ,zOffset);
				int[] croppedDimensions = {(int)Math.ceil((c[0][1]-c[0][0])/pixelSize),(int)Math.ceil((c[1][1]-c[1][0])/pixelSize),(int)Math.ceil((c[2][1]-c[2][0])/pixelSizeZ)};
				idx = firstBin[0];	// return idx to startpoint of this loop.

				int[][][] referenceFrame 	= new int[croppedDimensions[0]][croppedDimensions[1]][croppedDimensions[2]]; // create the reference array.
				int[][][] targetFrame 		= new int[croppedDimensions[0]][croppedDimensions[1]][croppedDimensions[2]]; 		// create the target (shifted) array.
				double targetMean = 0;	// mean value of target array.
				double referenceMean = 0;	// mean value of reference array.
				while(inputParticles.get(idx).channel == 1) // populate target first due to while loop design.
				{
					if (inputParticles.get(idx).include == 1)	// only include particles that are ok from the users parameter choice.
					{
						if (inputParticles.get(idx).x >= c[0][0] && inputParticles.get(idx).x <= c[0][1] &&
								inputParticles.get(idx).y >= c[1][0] && inputParticles.get(idx).y <= c[1][1] &&
								inputParticles.get(idx).z >= c[2][0] && inputParticles.get(idx).z <= c[2][1])
						{
							int x = (int)Math.floor((inputParticles.get(idx).x-c[0][0])/pixelSize);
							int y = (int)Math.floor((inputParticles.get(idx).y-c[1][0])/pixelSize);
							int z = (int)Math.floor((inputParticles.get(idx).z-c[2][0])/pixelSizeZ);
							referenceFrame[x][y][z]++;
							referenceMean++;	// keep track of total number of added particles (for mean calculation).
						}
					}
					idx++;	// step forward.
				}
				referenceMean /= (croppedDimensions[0]*croppedDimensions[1]*croppedDimensions[2]);	// calculate the mean.
				idx = secondBin[0];
				if (ch < nChannels)	// if this is not the final bin.
				{
					while(inputParticles.get(idx).channel == ch) // populate target first due to while loop design.
					{

						if (inputParticles.get(idx).include == 1)	// only include particles that are ok from the users parameter choice.
						{
							if (inputParticles.get(idx).x >= c[0][0] && inputParticles.get(idx).x <= c[0][1] &&
									inputParticles.get(idx).y >= c[1][0] && inputParticles.get(idx).y <= c[1][1] &&
									inputParticles.get(idx).z >= c[2][0] && inputParticles.get(idx).z <= c[2][1])
							{
								int x = (int)Math.floor((inputParticles.get(idx).x-c[0][0])/pixelSize);
								int y = (int)Math.floor((inputParticles.get(idx).y-c[1][0])/pixelSize) ;
								int z = (int)Math.floor((inputParticles.get(idx).z-c[2][0])/pixelSizeZ);
								targetFrame[x][y][z]++;
								targetMean++;	// keep track of total number of added particles (for mean calculation).
							}
						}
						idx++;	// step forward.
					}
				}
				else
				{
					while(idx < inputParticles.size()) // final batch, cover rounding errors.
					{

						if (inputParticles.get(idx).include == 1)	// only include particles that are ok from the users parameter choice.
						{
							if (inputParticles.get(idx).x >= c[0][0] && inputParticles.get(idx).x <= c[0][1] &&
									inputParticles.get(idx).y >= c[1][0] && inputParticles.get(idx).y <= c[1][1] &&
									inputParticles.get(idx).z >= c[2][0] && inputParticles.get(idx).z <= c[2][1])
							{
								int x = (int)((inputParticles.get(idx).x-c[0][0])/pixelSize) -1;
								int y = (int)((inputParticles.get(idx).y-c[1][0])/pixelSize) -1 ;
								int z = (int)((inputParticles.get(idx).z-c[2][0])/pixelSizeZ) -1;
								targetFrame[x][y][z]++;
								targetMean++;	// keep track of total number of added particles (for mean calculation).
							}
						}
						idx++;	// step forward.
					}
				} // last bin.

				targetMean /= (croppedDimensions[0]*croppedDimensions[1]*croppedDimensions[2]);	// calculate the mean.
				double refSquare = 0;										// this value needs to be calculated for every shift combination, pulled out and made accessible to save computational time.

				for (int i = 0; i < croppedDimensions[0]; i++)
				{
					for (int j = 0; j < croppedDimensions[1]; j++)
					{
						for (int k = 0; k < croppedDimensions[2]; k++)
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

			} // channel loop.

			for(int j = 1; j < optimalShift[0].length; j++)
			{
				ij.IJ.log("Channel " + (j+1) + " shifted by " + pixelSize*optimalShift[1][j]+  " x " + pixelSize*optimalShift[2][j] + " x " + pixelSizeZ*optimalShift[3][j] + " nm.");
			}

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
		for (int i = 0; i < 50000; i++)
		{
			Particle p = new Particle();
			p.include = 1;
			p.x = r.nextDouble()*1000;
			p.y = r.nextDouble()*1000;
			p.x = 500;
			p.y = 700;
			p.z = 0;
			//		p.z = r.nextDouble()*400-300;
			/*		if (p.x < 100)
				p.x = 100;
			if (p.y < 100)
				p.y = 100;
			 */	//		if (p.z < 100)
			//			p.z = 100;
			p.channel = 1;
			p.frame = frame;
			result.add(p);
			frame++;
		}
		for (int i = 0; i < 50000; i++)
		{
			Particle p = result.get(i);
			Particle p2 = new Particle();
			p2.include = 1;
			p2.x = p.x + 20;
			p2.y = p.y -100 ;
			p2.z = p.z + 40;
			p2.channel = 1;
			p2.frame = frame;
			result.add(p2);
			frame++;
		}
		for (int i = 0; i < 50000; i++)
		{
			Particle p = result.get(i);
			Particle p2 = new Particle();
			p2.include = 1;
			p2.x = p.x+ 10;
			p2.y = p.y- 10;
			p2.z = p.z+ 80;
			p2.channel = 1;
			p2.frame = frame;
			result.add(p2);
			frame++;
		}
		for (int i = 0; i < 50000; i++)
		{
			Particle p = result.get(i);
			Particle p2 = new Particle();
			p2.include = 1;
			p2.x = p.x+ 100;
			p2.y = p.y- 10;
			p2.z = p.z- 40;
			p2.channel = 1;
			p2.frame = frame;
			result.add(p2);
			frame++;
		}
		int[][] maxShift = new int[2][4]; //xy-z per channel

		int[] nBins = {4};
		int pixelSize = 10;
		int pixelSizeZ = 10;
		maxShift[0][0] = 200/pixelSize;
		maxShift[1][0] = 200/pixelSizeZ;
		maxShift[0][1] = 200/pixelSize;
		maxShift[1][1] = 200/pixelSizeZ;
		maxShift[0][2] = 200/pixelSize;
		maxShift[1][2] = 200/pixelSizeZ;
		maxShift[0][3] = 200/pixelSize;
		maxShift[1][3] = 200/pixelSizeZ;
		int xTimes = 10;
		int[] size = {xTimes*1280/pixelSize, xTimes*1280/pixelSize, 1000/pixelSizeZ};
		//	size[2] = 1; // if sending in 2D data, send in with zdim = 1.
		long time = System.nanoTime();
		result = run(result, nBins, maxShift,size ,pixelSize,pixelSizeZ);
		double error = 0;

		for (int i = 50000; i < 100000; i++){
			error += (result.get(0).x-result.get(i).x);
			if (result.get(i).x != 500)
				System.out.println(i + " ; " + result.get(i).x);
		}


		System.out.println(error);
		//result = runChannel(result,  maxShift,size ,pixelSize,pixelSizeZ);
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
					if (reference[xi][yi][zi] > 0)
					{
						//						if (target[xi-shift[0]][yi-shift[1]][zi-shift[2]] > 0)
						//						{
						union += (reference[xi][yi][zi] - refMean)*(target[xi-shift[0]][yi-shift[1]][zi-shift[2]]-tarMean);				
						shiftSquare += (target[xi-shift[0]][yi-shift[1]][zi-shift[2]]-tarMean)*(target[xi-shift[0]][yi-shift[1]][zi-shift[2]]-tarMean);
						//					}
					}
				}
			}			
		}	

		//double[] r = {((union / (refSquare*Math.sqrt(shiftSquare)))),shift[0],shift[1],shift[2]};
		double[] r = {union,shift[0],shift[1],shift[2]};
		return r;
	}
	/*
	 * Get centers of maximum correlation from the two bins. 
	 */
	public static int[][] getOverlapCenter(ArrayList<Particle> inputParticles, int[] firstBin, int[] secondBin, int[] dimensions,int pixelSize, int pixelSizeZ, int maxShift, int maxShiftZ, double zOffset)
	{
		int[][] range = new int[3][2];
		int xyPixelSize = pixelSize*2*maxShift;
		int zPixelSize = pixelSizeZ*2*maxShiftZ;
		int[] newDimensions = {dimensions[0]/(2*maxShift), dimensions[1]/(2*maxShift), dimensions[2]/(2*maxShiftZ)};
		if (newDimensions[2] == 0)
			newDimensions[2] = 1;
		int[][][] referenceFrame = new int[(int)Math.ceil(newDimensions[0])][(int)Math.ceil(newDimensions[1])][(int)Math.ceil(newDimensions[2])];
		int[][][] targetFrame = new int[(int)Math.ceil(newDimensions[0])][(int)Math.ceil(newDimensions[1])][(int)Math.ceil(newDimensions[2])];
		int idx = firstBin[0];
		double referenceMean = 0;
		double targetMean = 0;
		while (idx <= firstBin[1])
		{
			if (inputParticles.get(idx).include==1)
			{
				int x = (int)inputParticles.get(idx).x/xyPixelSize;
				int y = (int)inputParticles.get(idx).y/xyPixelSize;
				int z = (int) ((inputParticles.get(idx).z+ zOffset)/zPixelSize);
				referenceFrame[x][y][z] += 1;
				referenceMean++;
				if (Math.abs(x - referenceFrame.length) < 2 &&
						Math.abs(y - referenceFrame[0].length) < 2 &&
						Math.abs(z - referenceFrame[0][0].length + zOffset) < 2)
				{
					referenceFrame[x][y][z] += 1; // weight towards centern.				
					referenceMean++;
				}
			}
			idx++;
		}
		idx = secondBin[0];
		while (idx <= secondBin[1])
		{
			if (inputParticles.get(idx).include==1)
			{
				int x = (int)inputParticles.get(idx).x/xyPixelSize;
				int y = (int)inputParticles.get(idx).y/xyPixelSize;
				int z = (int)((inputParticles.get(idx).z+ zOffset)/zPixelSize);
				targetFrame[x][y][z] += 1;
				targetMean++;
				if (Math.abs(x - targetFrame.length) < 2 &&
						Math.abs(y - targetFrame[0].length) < 2 &&
						Math.abs(z - targetFrame[0][0].length + zOffset) < 2)
				{
					targetFrame[x][y][z] += 1; // weight towards centern.				
					targetMean++;
				}
			}
			idx++;
		}
		referenceMean /= newDimensions[0]*newDimensions[1]*newDimensions[2];
		targetMean /= newDimensions[0]*newDimensions[1]*newDimensions[2];
		double referenceSquare = 0;
		double targetSquare = 0;
		double union = 0;
		for (int x = 0; x < referenceFrame.length; x++)
		{
			for (int y = 0; y < referenceFrame[0].length; y++)
			{
				for (int z = 0; z < referenceFrame[0][0].length; z++)
				{
					referenceSquare += (referenceFrame[x][y][z] - referenceMean)*(referenceFrame[x][y][z] - referenceMean);
					targetSquare += (targetFrame[x][y][z] - targetMean)*(targetFrame[x][y][z] - targetMean);
				}				
			}
		}
		referenceSquare = Math.sqrt(referenceSquare);
		targetSquare = Math.sqrt(targetSquare);
		for (int x = 0; x < referenceFrame.length; x++)
		{
			for (int y = 0; y < referenceFrame[0].length; y++)
			{
				for (int z = 0; z < referenceFrame[0][0].length; z++)
				{
					if (((referenceFrame[x][y][z] - referenceMean)*(targetFrame[x][y][z] - targetMean))/(referenceSquare*targetSquare) > union)
					{
						union = ((referenceFrame[x][y][z] - referenceMean)*(targetFrame[x][y][z] - targetMean))/(referenceSquare*targetSquare);
						range[0][0] = x*xyPixelSize-128*pixelSize;
						range[1][0] = y*xyPixelSize-128*pixelSize;
						range[2][0] = z*zPixelSize-15*pixelSizeZ;
						range[0][1] = x*xyPixelSize+128*pixelSize;
						range[1][1] = y*xyPixelSize+128*pixelSize;
						range[2][1] = z*zPixelSize+15*pixelSizeZ;
					}
				}				
			}
		}
		// lower bounds:
		if(range[0][0] < 0)
			range[0][0] = 0;
		if(range[1][0] < 0)
			range[1][0] = 0;
		if(range[2][0] < -zOffset)
			range[2][0] = (int) -zOffset;
		// upper bounds:
		if(range[0][1] > dimensions[0]*pixelSize)
			range[0][1] = dimensions[0]*pixelSize;
		if(range[1][1] > dimensions[1]*pixelSize)
			range[1][1] = dimensions[1]*pixelSize;
		if(range[2][1] > dimensions[2]*pixelSizeZ)
			range[2][1] = dimensions[2]*pixelSizeZ;
		return range;
	}
}
