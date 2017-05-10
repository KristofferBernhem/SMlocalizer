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


/*
 * TODO: error when z < 0. shifting not working as intended.
 */
import static jcuda.driver.JCudaDriver.cuCtxCreate;
import static jcuda.driver.JCudaDriver.cuCtxSynchronize;
import static jcuda.driver.JCudaDriver.cuDeviceGet;
import static jcuda.driver.JCudaDriver.cuInit;
import static jcuda.driver.JCudaDriver.cuLaunchKernel;
import static jcuda.driver.JCudaDriver.cuMemFree;
import static jcuda.driver.JCudaDriver.cuMemcpyDtoH;
import static jcuda.driver.JCudaDriver.cuModuleGetFunction;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.JCudaDriver;
import static jcuda.driver.JCudaDriver.cuModuleLoadDataEx;

public class ImageCrossCorr3DGPU {


	public static ArrayList<Particle> run(ArrayList<Particle> inputParticles, int[] nBins, int[][] boundry, int[] dimensions,int pixelSize, int pixelSizeZ) // main function call. Will per channel autocorrelate between time binned data.
	{
		ArrayList<Particle> shiftedParticles = new ArrayList<Particle>();
		JCudaDriver.setExceptionsEnabled(true);
		cuInit(0);
		CUdevice device = new CUdevice();
		cuDeviceGet(device, 0);
		CUcontext context = new CUcontext();
		cuCtxCreate(context, 0, device);
		// Load the PTX that contains the kernel.
		CUmodule module = new CUmodule();

		String ptxFileName = "driftCorr.ptx";
		byte ptxFile[] = CUDA.loadData(ptxFileName);

		cuModuleLoadDataEx(module, Pointer.to(ptxFile), 
				0, new int[0], Pointer.to(new int[0]));

		// Obtain a handle to the kernel function.
		CUfunction function = new CUfunction();
		cuModuleGetFunction(function, module, "runAdd");
		int nChannels 	= inputParticles.get(inputParticles.size()-1).channel; // data is sorted on a per channel basis.
		int idx 		= 0; // index for inputParticles.
		double zOffset 	= 0;
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
			int reduce = 2;
			int addZ = 1;
			if (dimensions[2] == 1){
				reduce = 1;
				addZ = 0;
			}

			int[] shiftVector = new int[reduce*3*(1+2*maxShift)*(1+2*maxShift)*(addZ+maxShiftZ)];
			int counter = 0;
			for (int xShift = - maxShift; xShift <= maxShift; xShift++) // calculate the shifts.
			{
				for (int yShift = - maxShift; yShift <= maxShift; yShift++)
				{
					if (dimensions[2] > 1)
					{
						for (int zShift = -maxShiftZ; zShift <= maxShiftZ; zShift++)
						{
							shiftVector[counter]     = xShift;
							shiftVector[counter + 1] = yShift;
							shiftVector[counter + 2] = zShift;
							counter += 3;
						}
					}else
					{
						shiftVector[counter]     = xShift;
						shiftVector[counter + 1] = yShift;
						shiftVector[counter + 2] = 0;
						counter += 3;
					}
				}
			}
			
			while (idx < inputParticles.size() && inputParticles.get(idx).channel == ch)
			{				
				idx++;
			} // loop whilst... Find final entry for this channel.
			idx--;

			int maxFrame = inputParticles.get(idx).frame; // final frame included in this channel.
			int binsize = (int) (maxFrame/nBins[ch-1]);		// size of each bin (in number of frames).		
			int currBin = 2;	// start with bin 2.
			double[][] optimalShift = new double[4][nBins[ch-1]];	// this vector will hold all shift values.
			idx = tempIdx;	// return idx to startpoint of this loop.

			
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
				/*
				 * Make 256x256x20 image or less, going for voxel size of 2*maxDrift x 2*maxDrift * 2*maxDriftZ. 
				 * Find pixel with largest correlation with the next bin and extract a 10x10x20 nm voxel 256x256x20 pixel large ROI and apply the drift correction from this ROI to all particles.
				 */
			//	System.out.println("bins: " + currBin + " range: " + firstBin[0] + " x " + firstBin[1]+ " through: " + secondBin[0] + " x " + secondBin[1]);
				int[][] c = getOverlapCenter(inputParticles, firstBin, secondBin, dimensions, pixelSize,  pixelSizeZ, maxShift, maxShiftZ,zOffset);
				int[] croppedDimensions = {(int)Math.ceil((c[0][1]-c[0][0])/pixelSize),(int)Math.ceil((c[1][1]-c[1][0])/pixelSize),(int)Math.ceil((c[2][1]-c[2][0])/pixelSizeZ)};
				idx = tempIdx;	// return idx to startpoint of this loop.
	/*			System.out.println("from" + c[0][0] + " to " + c[0][1]);
				System.out.println("from" + c[1][0] + " to " + c[1][1]);
				System.out.println("from" + c[2][0] + " to " + c[2][1]);
				System.out.println("********");
*/
			//	System.out.println("bin: " + currBin  + " _ "+ croppedDimensions[0] + " to " + croppedDimensions[1]); 
				int[] referenceFrame 	= new int[croppedDimensions[0]*croppedDimensions[1]*croppedDimensions[2]]; // create the reference array.
				int[] targetFrame 		= new int[croppedDimensions[0]*croppedDimensions[1]*croppedDimensions[2]]; 		// create the target (shifted) array.	//					
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
							int linearIndex = (int)((int)Math.floor((inputParticles.get(idx).x-c[0][0])/pixelSize) + 
									(int)Math.floor((inputParticles.get(idx).y-c[1][0])/pixelSize) * croppedDimensions[0] + 
									(int)Math.floor((inputParticles.get(idx).z+zOffset-c[2][0])/pixelSizeZ) * croppedDimensions[0]*croppedDimensions[1]);
							referenceFrame[linearIndex] += 1;	// increase value by one if there is a particle within the voxel.
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
								int linearIndex = (int)((int)Math.floor((inputParticles.get(idx).x-c[0][0])/pixelSize) + 
										(int)Math.floor((inputParticles.get(idx).y-c[1][0])/pixelSize) * croppedDimensions[0] + 
										(int)Math.floor((inputParticles.get(idx).z+zOffset-c[2][0])/pixelSizeZ) * croppedDimensions[0]*croppedDimensions[1]);
								targetFrame[linearIndex] += 1;	// increase value by one if there is a particle within the voxel.
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
								int linearIndex = (int)((int)Math.floor((inputParticles.get(idx).x-c[0][0])/pixelSize) + 
										(int)Math.floor((inputParticles.get(idx).y-c[1][0])/pixelSize) * croppedDimensions[0] + 
										(int)Math.floor((inputParticles.get(idx).z+zOffset-c[2][0])/pixelSizeZ) * croppedDimensions[0]*croppedDimensions[1]);
								targetFrame[linearIndex] += 1;	// increase value by one if there is a particle within the voxel.
								targetMean++;	// keep track of total number of added particles (for mean calculation).
							}
						}
						idx++;	// step forward.
					}
				} // last bin.

				targetMean /= (croppedDimensions[0]*croppedDimensions[1]*croppedDimensions[2]);	// calculate the mean.
				
				double refSquare = 0;										// this value needs to be calculated for every shift combination, pulled out and made accessible to save computational time.
				double tarSquare = 0;										// this value needs to be calculated for every shift combination, pulled out and made accessible to save computational time.
				for (int i = 0; i < referenceFrame.length; i++)
				{
					refSquare += (referenceFrame[i] - referenceMean)*( referenceFrame[i] - referenceMean); // calculate square pixel to mean difference.					
					tarSquare += (targetFrame[i] - targetMean)*( targetFrame[i] - targetMean); // calculate square pixel to mean difference.					
				}				
				refSquare = Math.sqrt(refSquare);	// square root of difference.
				tarSquare = Math.sqrt(tarSquare);	// square root of difference.
				
			

				// Initialize the driver and create a context for the first device.
				double[] hostOutput = new double[shiftVector.length/3];
				float[] means 						= {(float) referenceMean, (float) targetMean};
				CUdeviceptr device_shiftVector 		= CUDA.copyToDevice(shiftVector);
				CUdeviceptr device_referenceFrame 	= CUDA.copyToDevice(referenceFrame);
				CUdeviceptr device_targetFrame 		= CUDA.copyToDevice(targetFrame);
				CUdeviceptr device_meanVector 		= CUDA.copyToDevice(means);
				CUdeviceptr device_dimensions 		= CUDA.copyToDevice(croppedDimensions);	
				CUdeviceptr deviceOutput 			= CUDA.allocateOnDevice((double)hostOutput.length);				

				int frameLength 			= referenceFrame.length;
				int shiftVectorLength 		= shiftVector.length;
				int meanVectorLength 		= 2;
				int dimensionsLength 		= 3;
				int outputLength			= hostOutput.length;

				Pointer kernelParameters 	= Pointer.to(   
						Pointer.to(device_referenceFrame),
						Pointer.to(new int[]{frameLength}),
						Pointer.to(device_targetFrame),
						Pointer.to(new int[]{frameLength}),
						Pointer.to(device_shiftVector),
						Pointer.to(new int[]{shiftVectorLength}),
						Pointer.to(device_meanVector),
						Pointer.to(new int[]{meanVectorLength}),
						Pointer.to(device_dimensions),
						Pointer.to(new int[]{dimensionsLength}),
						Pointer.to(deviceOutput),
						Pointer.to(new int[]{outputLength})
						);


				int gridSizeX = (int)Math.ceil(Math.sqrt(croppedDimensions[0]*croppedDimensions[1]*croppedDimensions[2]));///frameBatch); 	// update.
				int gridSizeY = gridSizeX;														// update.
				int blockSizeX = 1;
				int blockSizeY = 1;

				cuLaunchKernel(function,
						gridSizeX,  gridSizeY, 1, 	// Grid dimension
						blockSizeX, blockSizeY, 1,  // Block dimension
						0, null,               		// Shared memory size and stream
						kernelParameters, null 		// Kernel- and extra parameters
						);
				cuCtxSynchronize();
				cuMemcpyDtoH(Pointer.to(hostOutput), deviceOutput,
						hostOutput.length * Sizeof.DOUBLE);

				cuMemFree(device_referenceFrame);
				cuMemFree(device_targetFrame);
				cuMemFree(device_shiftVector);
				cuMemFree(device_meanVector);
				cuMemFree(device_dimensions);
				cuMemFree(deviceOutput);
				
				double sum = 0;
				for (int i = 0; i < shiftVector.length-3; i += 3)
				{
					if (shiftVector[i+1] == 0 &&
							shiftVector[i+2] == 0 &&
							shiftVector[i+3] == 0 )
						sum =  shiftVector[i];
				}
				//System.out.println(currBin + ": " + sum);
				int idxZero= 0;
				for (int i = 0; i < hostOutput.length; i++)
				{
					
					if (hostOutput[i]/(tarSquare*refSquare) > sum)
					{
					//	System.out.println(hostOutput[i]+ " x:" + shiftVector[i*3]+ " y:" + shiftVector[i*3+1]+ " z:" + shiftVector[i*3+2] + " from: " + i);
						sum = hostOutput[i]/(tarSquare*refSquare);
						idxZero = i;
					}
				}
				optimalShift[0][currBin-1] = sum;
		//		System.out.println(currBin + ": " +sum);
				
				if (optimalShift[0][currBin-1]  > 0.1)
				{

					optimalShift[1][currBin-1] = shiftVector[idxZero*3+0];
					optimalShift[2][currBin-1] = shiftVector[idxZero*3+1];
					optimalShift[3][currBin-1] = shiftVector[idxZero*3+2];							
				}
				tempIdx = secondBin[0];	// startpoint of next loop.
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
			int tempIdx 	= idx; // hold value
			int[] firstBin = {idx,1};
			while (idx < inputParticles.size() &&
					inputParticles.get(idx).channel == 1)
			{
				firstBin[1] = idx;
				idx++;
			}
			for (int ch = 2; ch <= nChannels; ch++)
			{
				int maxShift 	= boundry[0][ch-1];
				int maxShiftZ 	= boundry[1][ch-1];
				int reduce = 2;
				int addZ = 1;
				if (dimensions[2] == 1){
					reduce = 1;
					addZ = 0;
				}
	
				int[] shiftVector = new int[reduce*3*(1+2*maxShift)*(1+2*maxShift)*(addZ+maxShiftZ)];
				int counter = 0;
				for (int xShift = - maxShift; xShift <= maxShift; xShift++) // calculate the shifts.
				{
					for (int yShift = - maxShift; yShift <= maxShift; yShift++)
					{
						if (dimensions[2] > 1)
						{
							for (int zShift = -maxShiftZ; zShift <= maxShiftZ; zShift++)
							{
								shiftVector[counter]     = xShift;
								shiftVector[counter + 1] = yShift;
								shiftVector[counter + 2] = zShift;
								counter += 3;
							}
						}else
						{
							shiftVector[counter]     = xShift;
							shiftVector[counter + 1] = yShift;
							shiftVector[counter + 2] = 0;
							counter += 3;
						}
					}
				}
				idx = tempIdx;

				int[] secondBin = {idx,1};
				

				while (idx < inputParticles.size() && inputParticles.get(idx).channel < ch)
					idx++;
				secondBin[0] = idx;
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

				int[] referenceFrame 	= new int[croppedDimensions[0]*croppedDimensions[1]*croppedDimensions[2]]; // create the reference array.
				int[] targetFrame 		= new int[croppedDimensions[0]*croppedDimensions[1]*croppedDimensions[2]]; 		// create the target (shifted) array.	//					
				double targetMean = 0;	// mean value of target array.
				double referenceMean = 0;	// mean value of reference array.

				while(inputParticles.get(idx).channel == 1) // populate target first due to while loop design.
				{
					if (inputParticles.get(idx).include == 1)	// only include particles that are ok from the users parameter choice.
					{
						if (inputParticles.get(idx).x >= c[0][0] && inputParticles.get(idx).x < c[0][1] &&
								inputParticles.get(idx).y >= c[1][0] && inputParticles.get(idx).y < c[1][1] &&
								inputParticles.get(idx).z >= c[2][0] && inputParticles.get(idx).z < c[2][1])
						{
							int linearIndex = (int)((int)Math.floor((inputParticles.get(idx).x-c[0][0])/pixelSize) + 
									(int)Math.floor((inputParticles.get(idx).y-c[1][0])/pixelSize) * croppedDimensions[0] + 
									(int)Math.floor((inputParticles.get(idx).z+zOffset-c[2][0])/pixelSizeZ) * croppedDimensions[0]*croppedDimensions[1]);
							referenceFrame[linearIndex] += 1;	// increase value by one if there is a particle within the voxel.
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
							if (inputParticles.get(idx).x >= c[0][0] && inputParticles.get(idx).x < c[0][1] &&
									inputParticles.get(idx).y >= c[1][0] && inputParticles.get(idx).y < c[1][1] &&
									inputParticles.get(idx).z >= c[2][0] && inputParticles.get(idx).z < c[2][1])
							{
								int linearIndex = (int)((int)Math.floor((inputParticles.get(idx).x-c[0][0])/pixelSize) + 
										(int)Math.floor((inputParticles.get(idx).y-c[1][0])/pixelSize) * croppedDimensions[0] + 
										(int)Math.floor((inputParticles.get(idx).z+zOffset-c[2][0])/pixelSizeZ) * croppedDimensions[0]*croppedDimensions[1]);
								targetFrame[linearIndex] += 1;	// increase value by one if there is a particle within the voxel.
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
							if (inputParticles.get(idx).x >= c[0][0] && inputParticles.get(idx).x < c[0][1] &&
									inputParticles.get(idx).y >= c[1][0] && inputParticles.get(idx).y < c[1][1] &&
									inputParticles.get(idx).z >= c[2][0] && inputParticles.get(idx).z < c[2][1])
							{
								int linearIndex = (int)((int)Math.floor((inputParticles.get(idx).x-c[0][0])/pixelSize) + 
										(int)Math.floor((inputParticles.get(idx).y-c[1][0])/pixelSize) * croppedDimensions[0] + 
										(int)Math.floor((inputParticles.get(idx).z+zOffset-c[2][0])/pixelSizeZ) * croppedDimensions[0]*croppedDimensions[1]);
								targetFrame[linearIndex] += 1;	// increase value by one if there is a particle within the voxel.
								targetMean++;	// keep track of total number of added particles (for mean calculation).
							}
						}
						idx++;	// step forward.
					}
				} // last bin.

				targetMean /= (croppedDimensions[0]*croppedDimensions[1]*croppedDimensions[2]);	// calculate the mean.
				double refSquare = 0;										// this value needs to be calculated for every shift combination, pulled out and made accessible to save computational time.
				double tarSquare = 0;										// this value needs to be calculated for every shift combination, pulled out and made accessible to save computational time.

				for (int i = 0; i < referenceFrame.length; i++)
				{
					refSquare += (referenceFrame[i] - referenceMean)*( referenceFrame[i] - referenceMean); // calculate square pixel to mean difference.					
					tarSquare += (targetFrame[i] - targetMean)*( targetFrame[i] - targetMean); // calculate square pixel to mean difference.					
				}				
				refSquare = Math.sqrt(refSquare);	// square root of difference.
				tarSquare = Math.sqrt(tarSquare);	// square root of difference.

				JCudaDriver.setExceptionsEnabled(true);
				cuInit(0);

				CUdevice device = new CUdevice();
				cuDeviceGet(device, 0);
				CUcontext context = new CUcontext();
				cuCtxCreate(context, 0, device);
				// Load the PTX that contains the kernel.
				CUmodule module = new CUmodule();

				String ptxFileName = "driftCorr.ptx";
				byte ptxFile[] = CUDA.loadData(ptxFileName);

				cuModuleLoadDataEx(module, Pointer.to(ptxFile), 
						0, new int[0], Pointer.to(new int[0]));

				// Obtain a handle to the kernel function.
				CUfunction function = new CUfunction();
				cuModuleGetFunction(function, module, "runAdd");

				// Initialize the driver and create a context for the first device.
				double[] hostOutput = new double[shiftVector.length/3];
				float[] means 						= {(float) referenceMean, (float) targetMean};
				CUdeviceptr device_shiftVector 		= CUDA.copyToDevice(shiftVector);
				CUdeviceptr device_referenceFrame 	= CUDA.copyToDevice(referenceFrame);
				CUdeviceptr device_targetFrame 		= CUDA.copyToDevice(targetFrame);

				CUdeviceptr device_meanVector 		= CUDA.copyToDevice(means);
				CUdeviceptr device_dimensions 		= CUDA.copyToDevice(croppedDimensions);	
				CUdeviceptr deviceOutput 			= CUDA.allocateOnDevice((double)hostOutput.length);				

				int frameLength 			= referenceFrame.length;
				int shiftVectorLength 		= shiftVector.length;
				int meanVectorLength 		= 2;
				int dimensionsLength 		= 3;
				int outputLength			= hostOutput.length;

				Pointer kernelParameters 	= Pointer.to(   
						Pointer.to(device_referenceFrame),
						Pointer.to(new int[]{frameLength}),
						Pointer.to(device_targetFrame),
						Pointer.to(new int[]{frameLength}),
						Pointer.to(device_shiftVector),
						Pointer.to(new int[]{shiftVectorLength}),
						Pointer.to(device_meanVector),
						Pointer.to(new int[]{meanVectorLength}),
						Pointer.to(device_dimensions),
						Pointer.to(new int[]{dimensionsLength}),
						Pointer.to(deviceOutput),
						Pointer.to(new int[]{outputLength})
						);


				int gridSizeX = (int)Math.ceil(Math.sqrt(croppedDimensions[0]*croppedDimensions[1]*croppedDimensions[2]));///frameBatch); 	// update.
				int gridSizeY = gridSizeX;														// update.
				int blockSizeX = 1;
				int blockSizeY = 1;

				cuLaunchKernel(function,
						gridSizeX,  gridSizeY, 1, 	// Grid dimension
						blockSizeX, blockSizeY, 1,  // Block dimension
						0, null,               		// Shared memory size and stream
						kernelParameters, null 		// Kernel- and extra parameters
						);
				cuCtxSynchronize();
				cuMemcpyDtoH(Pointer.to(hostOutput), deviceOutput,
						hostOutput.length * Sizeof.DOUBLE);

				cuMemFree(device_referenceFrame);
				cuMemFree(device_targetFrame);
				cuMemFree(device_shiftVector);
				cuMemFree(device_meanVector);
				cuMemFree(device_dimensions);
				cuMemFree(deviceOutput);
				double sum = 0;
				int idxZero= 0;
				for (int i = 0; i < hostOutput.length; i++)
				{
					//					System.out.println(hostOutput[i]+ " x:" + shiftVector[i*3]+ " y:" + shiftVector[i*3+1]+ " z:" + shiftVector[i*3+2] + " from: " + i);
					if (hostOutput[i] > sum)
					{
						sum = hostOutput[i];
						idxZero = i;
					}
				}
				optimalShift[0][ch-1] = sum/(tarSquare*refSquare);
				if (optimalShift[0][ch-1]  > 0)
				{
					optimalShift[1][ch-1] = shiftVector[idxZero*3+0];
					optimalShift[2][ch-1] = shiftVector[idxZero*3+1];
					optimalShift[3][ch-1] = shiftVector[idxZero*3+2];							
				}
				tempIdx = secondBin[0];	// startpoint of next loop.
			
			

			} // loop over channel 2:end
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
				int z = (int) ((inputParticles.get(idx).z+ zOffset)/zPixelSize );
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
				int z = (int) ((inputParticles.get(idx).z+ zOffset)/zPixelSize );
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

		int[] nBins 	= {4};
		int pixelSize 	= 10;
		int pixelSizeZ 	= 10;
		maxShift[0][0] 	= 200/pixelSize;
		maxShift[1][0] 	= 200/pixelSize;
		maxShift[0][1] 	= 200/pixelSize;
		maxShift[1][1] 	= 200/pixelSizeZ;
		maxShift[0][2] 	= 200/pixelSize;
		maxShift[1][2] 	= 200/pixelSizeZ;
		maxShift[0][3] 	= 200/pixelSize;
		maxShift[1][3] 	= 200/pixelSizeZ;
		int xTimes = 1; // 256x256x20 pixels work great.
		int[] size = {xTimes*1280/pixelSize, xTimes*1280/pixelSize, 1000/pixelSizeZ};

		//size[2] = 1; // if sending in 2D data, send in with zdim = 1.
		long time = System.nanoTime();
		result = run(result, nBins, maxShift,size ,pixelSize,pixelSizeZ);

		//result = runChannel(result,  maxShift,size ,pixelSize,pixelSizeZ);
	double error = 0;
		
		for (int i = 50000; i < 100000; i++){
			error += (result.get(0).x-result.get(i).x);
			if (result.get(i).x != 500)
				System.out.println(i + " ; " + result.get(i).x);
		}
			
			
		System.out.println(error);
		time = System.nanoTime() - time;
		System.out.println(time*1E-9);	
		/*System.out.println(c[0][0] +  " x " + c[0][1]);
		System.out.println(c[1][0] +  " x " + c[1][1]);
		System.out.println(c[2][0] +  " x " + c[2][1]);*/
	}


}