import static jcuda.driver.JCudaDriver.cuCtxCreate;
import static jcuda.driver.JCudaDriver.cuCtxSynchronize;
import static jcuda.driver.JCudaDriver.cuDeviceGet;
import static jcuda.driver.JCudaDriver.cuInit;
import static jcuda.driver.JCudaDriver.cuLaunchKernel;
import static jcuda.driver.JCudaDriver.cuMemFree;
import static jcuda.driver.JCudaDriver.cuMemcpyDtoH;
import static jcuda.driver.JCudaDriver.cuModuleGetFunction;
import static jcuda.driver.JCudaDriver.cuModuleLoadDataEx;

import java.util.ArrayList;

import ij.ImagePlus;
import ij.plugin.filter.Analyzer;
import ij.process.ImageProcessor;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.JCudaDriver;
import jcuda.runtime.JCuda;


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




/**
*
* @author kristoffer.bernhem@gmail.com
*/
/*
 * ProcessMedianFit is called by the process call button and will in sequence load image data, median filter (background removal), locate events and fit them, returning a result table.
 */
public class processMedianFit {

	//@SuppressWarnings("deprecation")
	public static void run(final int[] W, ImagePlus image, int[] MinLevel, int pixelSize, int[] totalGain,double maxSigma, int gWindow, String modality)
	{
	//	int maxGrid = CUDA.getGrid(device);
	//	maxGrid = (int)(Math.log(maxGrid)/Math.log(2))+1;

		int maxGrid = 31;
		int columns 						= image.getWidth();
		int rows 							= image.getHeight();		
		int nChannels 						= image.getNChannels(); 	// Number of channels.
		int nFrames 						= image.getNFrames();
		if (nFrames == 1)
			nFrames 						= image.getNSlices();  
		double convCriteria = 1E-8; // how large improvement from one step to next we require.
		int maxIterations = 1000;  // stop if an individual fit reaches this number of iterati
		ArrayList<Particle> Results 		= new ArrayList<Particle>();		// Fitted results array list.

		int minPosPixels = gWindow*gWindow - 4;
		if (modality.equals("2D"))
		{
			minPosPixels = gWindow*gWindow - 4; // update to relevant numbers for this modality.
		}
		else if (modality.equals("PRILM"))
		{
			minPosPixels = gWindow*gWindow - 4; // update to relevant numbers for this modality.
		}
		else if (modality.equals("Biplane"))
		{
			// get calibrated values for gWindow.
			minPosPixels = gWindow*gWindow - 4; // update to relevant numbers for this modality.
		}
		else if (modality.equals("Double Helix"))
		{
			// get calibrated values for gWindow.
			minPosPixels = gWindow*gWindow - 4; // update to relevant numbers for this modality.
		}
		else if (modality.equals("Astigmatism"))
		{
			// get calibrated values for gWindow.
			minPosPixels = gWindow*gWindow - 4; // update to relevant numbers for this modality.					
		}
		// Initialize the driver and create a context for the first device.
		
		long GB = 1024*1024*1024;
		int frameSize = (4*columns*rows)*Sizeof.FLOAT;
		
								
		JCudaDriver.setExceptionsEnabled(true);
		// Initialize the driver and create a context for the first device.
		cuInit(0);

		
		
      
        CUdevice device = new CUdevice();
		cuDeviceGet(device, 0);
		CUcontext context = new CUcontext();
		cuCtxCreate(context, 0, device);
				
		long total[] = { 0 };
		long free[] = { 0 };
		JCuda.cudaMemGetInfo(free, total);

//		System.out.println("Total "+total[0]/GB+" free "+free[0]/GB);
	//	ij.IJ.log("Total "+total[0]/GB+" free "+free[0]/GB);
		long maxMemoryGPU = (long) (0.5*free[0]); 
		// Load the PTX that contains the kernel.
		CUmodule moduleMedian = new CUmodule();
		String ptxFileNameMedian = "medianFilter.ptx";
		byte ptxFileMedian[] = CUDA.loadData(ptxFileNameMedian);
		cuModuleLoadDataEx(moduleMedian, Pointer.to(ptxFileMedian), 
	            0, new int[0], Pointer.to(new int[0]));
		// Obtain a handle to the kernel function.
		CUfunction functionMedian = new CUfunction();
		cuModuleGetFunction(functionMedian, moduleMedian, "medianKernel");
		
		
		
		CUmodule moduleBSpline = new CUmodule();							
		String ptxFileNameBspline = "filterImage.ptx";				
		byte ptxFileBspline[] = CUDA.loadData(ptxFileNameBspline);
		cuModuleLoadDataEx(moduleBSpline, Pointer.to(ptxFileBspline), 
        0, new int[0], Pointer.to(new int[0]));				
		CUfunction functionBpline = new CUfunction();
		cuModuleGetFunction(functionBpline, moduleBSpline, "filterKernel");
		
		// prepare for locating centra for gaussfitting.
		// Load the PTX that contains the kernel.
		CUmodule moduleLM = new CUmodule();
		String ptxFileNameFindMaxima = "findMaxima.ptx";
		byte ptxFileFindMaxima[] = CUDA.loadData(ptxFileNameFindMaxima);
		//	cuModuleLoad(moduleLM, "findMaxima.ptx");					
		// Obtain a handle to the kernel function
		cuModuleLoadDataEx(moduleLM, Pointer.to(ptxFileFindMaxima), 
				0, new int[0], Pointer.to(new int[0]));
		CUfunction findMaximaFcn = new CUfunction();
		cuModuleGetFunction(findMaximaFcn, moduleLM, "run");	// findMaxima.ptx (run function).
		
		
		// gauss fit algorithm.
		// Load the PTX that contains the kernel.
		CUmodule moduleGFit = new CUmodule();
		String ptxFileNameGaussFit = "gFit.ptx";
		byte ptxFileGaussFit[] = CUDA.loadData(ptxFileNameGaussFit);
		cuModuleLoadDataEx(moduleGFit, Pointer.to(ptxFileGaussFit), 
				0, new int[0], Pointer.to(new int[0]));
		// Obtain a handle to the kernel function.
		CUfunction fittingFcn = new CUfunction();
		cuModuleGetFunction(fittingFcn, moduleGFit, "gaussFitter");  //gFit.pth (gaussFitter function).
		for(int Ch = 1; Ch <= nChannels; Ch++)
		{
			
			int staticMemory = (2*W[Ch-1]+1)*rows*columns*Sizeof.FLOAT;
			long framesPerBatch = (3*GB-staticMemory)/frameSize; // 3 GB memory allocation gives this numbers of frames. 					
			int nCenter =(( columns*rows/(gWindow*gWindow)) / 2); // ~ 80 possible particles for a 64x64 frame. Lets the program scale with frame size.
			int nMax = (int) (maxMemoryGPU/(4*columns*rows*Sizeof.INT + 4*nCenter*Sizeof.INT)); 	// the localMaxima GPU calculations require: (x*y*frame*(Sizeof.INT ) + frame*nCenters*Sizeof.FLOAT)/gb memory. with known x and y dimensions, determine maximum size of frame for each batch.
			int loadedFrames = 0;
			int startFrame = 1;					
			int endFrame = (int)framesPerBatch;					

			if (endFrame > nFrames)
				endFrame = nFrames;
			double lowXY = gWindow/2 - 2;
			if (lowXY < 1)
				lowXY = 1;
			double highXY = gWindow/2 +  2;
			double[] bounds = { // bounds for gauss fitting.
					0.6			, 1.4,				// amplitude.
					lowXY	, highXY,			// x.
					lowXY	, highXY,			// y.
					0.8			,  maxSigma,		// sigma x.
					0.8			,  maxSigma,		// sigma y.
					(-0.5*Math.PI) , (0.5*Math.PI),	// theta.
					-0.5		, 0.5				// offset.
			};

			
			CUdeviceptr deviceBounds 		= CUDA.copyToDevice(bounds);
			while (loadedFrames < nFrames)
			{							
				float[] timeVector = new float[(endFrame-startFrame+1) * rows * columns];
				float[] MeanFrame = new float[endFrame-startFrame+1]; 				// Will include frame mean value.
				ImageProcessor IP = image.getProcessor();
				int frameCounter = 0;

				for (int Frame = startFrame; Frame <= endFrame; Frame++)
				{			
					if (image.getNFrames() == 1)
					{
						image.setPosition(							
								Ch,			// channel.
								Frame,			// slice.
								1);		// frame.
					}
					else
					{														
						image.setPosition(
								Ch,			// channel.
								1,			// slice.
								Frame);		// frame.
					}
					IP = image.getProcessor();

					for (int i = 0; i < rows*columns; i ++)
					{
						timeVector[frameCounter + (endFrame-startFrame+1)*i] = IP.get(i);			
						MeanFrame[frameCounter] += IP.get(i);
					}
					MeanFrame[frameCounter] /= (columns*rows);
					loadedFrames++;
					frameCounter++;
				} // frame loop for mean calculations.
				
				int stepLength = nFrames/300;
				if (stepLength > 10)
					stepLength = 10;
				if(nFrames < 500)
					stepLength = 1;
				int nData = rows * columns;
				int blockSize = 256;
				int gridSize = (nData + blockSize - 1)/blockSize;
				gridSize = (int) (Math.log(gridSize)/Math.log(2) + 1);
				if (gridSize > maxGrid)
					gridSize = (int)( Math.pow(2, maxGrid));
				else
					gridSize = (int)( Math.pow(2, gridSize));
				CUdeviceptr device_Data 		= CUDA.copyToDevice(timeVector);
				CUdeviceptr device_meanVector 	= CUDA.copyToDevice(MeanFrame);
				CUdeviceptr deviceOutput 		= CUDA.allocateOnDevice((int)timeVector.length);
				CUdeviceptr device_window 		= CUDA.allocateOnDevice((float)((2 * W[Ch-1] + 1) * rows * columns)); // swap vector.
				int filterWindowLength 		= (2 * W[Ch-1] + 1) * rows * columns;
				int dataLength 				= timeVector.length;
				int meanVectorLength 		= MeanFrame.length;
				Pointer kernelParametersMedian 	= Pointer.to( 
						Pointer.to(new int[]{nData}),
						Pointer.to(new int[]{W[Ch]}),
						Pointer.to(device_window),
						Pointer.to(new int[]{filterWindowLength}),
						Pointer.to(new int[]{(meanVectorLength)}),
						Pointer.to(device_Data),
						Pointer.to(new int[]{dataLength}),
						Pointer.to(device_meanVector),
						Pointer.to(new int[]{meanVectorLength}),								
						Pointer.to(new int[]{stepLength}),
						Pointer.to(deviceOutput),
						Pointer.to(new int[]{dataLength})
						);
				cuLaunchKernel(functionMedian,
						gridSize,  1, 1, 	// Grid dimension
						blockSize, 1, 1,  // Block dimension
						0, null,               		// Shared memory size and stream
						kernelParametersMedian, null 		// Kernel- and extra parameters
						);
/*
				int blockSizeX 	= 1;
				int blockSizeY 	= 1;				   
				int gridSizeX 	= columns;
				int gridSizeY 	= rows;
			//	ij.IJ.log("grids: " + gridSizeX + " x " + gridSizeY);
				cuLaunchKernel(functionMedian,
						gridSizeX,  gridSizeY, 1, 	// Grid dimension
						blockSizeX, blockSizeY, 1,  // Block dimension
						0, null,               		// Shared memory size and stream
						kernelParametersMedian, null 		// Kernel- and extra parameters
						);
	*/			cuCtxSynchronize();
				cuMemFree(device_window);
				cuMemFree(device_Data);  
				cuMemFree(device_meanVector);

				// B-spline filter image:

				double[] filterKernel = { 0.0015257568383789054, 0.003661718759765626, 0.02868598630371093, 0.0036617187597656254, 0.0015257568383789054, 
                        0.003661718759765626, 0.008787890664062511, 0.06884453115234379, 0.00878789066406251, 0.003661718759765626, 
                        0.02868598630371093, 0.06884453115234379, 0.5393295900878906, 0.06884453115234378, 0.02868598630371093,
                        0.0036617187597656254, 0.00878789066406251, 0.06884453115234378, 0.008787890664062508, 0.0036617187597656254, 
                        0.0015257568383789054, 0.003661718759765626, 0.02868598630371093, 0.0036617187597656254, 0.0015257568383789054}; // 5x5 bicubic Bspline filter.
				
				int bSplineDataLength = timeVector.length;
				nData = timeVector.length/(columns*rows);
				CUdeviceptr deviceOutputBSpline 		= CUDA.allocateOnDevice(bSplineDataLength);
				CUdeviceptr deviceFilterKernel 			= CUDA.copyToDevice(filterKernel); // filter to applied to each pixel.				
				Pointer kernelParametersBspline 		= Pointer.to(   
						Pointer.to(new int[]{nData}),
						Pointer.to(deviceOutput),					// input data is output from medianFilter function call.
						Pointer.to(new int[]{bSplineDataLength}),	// length of vector
						Pointer.to(new int[]{(rows)}), 			// width
						Pointer.to(new int[]{(columns)}),				// height
						Pointer.to(deviceFilterKernel),				// Transfer filter kernel.
						Pointer.to(new int[]{(int)filterKernel.length}),								
						Pointer.to(new int[]{(int)(Math.sqrt(filterKernel.length))}), // width of filter kernel.
						Pointer.to(deviceOutputBSpline),			// result vector.
						Pointer.to(new int[]{bSplineDataLength})
						);

				blockSize = 256;
				gridSize = (nData + blockSize - 1)/blockSize;
				gridSize = (int) (Math.log(gridSize)/Math.log(2) + 1);
				if (gridSize > maxGrid)
					gridSize = (int)( Math.pow(2, maxGrid));
				else
					gridSize = (int)( Math.pow(2, gridSize));
				cuLaunchKernel(functionBpline,
						gridSize,  1, 1, 	// Grid dimension
						blockSize, 1, 1,  // Block dimension
						0, null,               		// Shared memory size and stream
						kernelParametersBspline, null 		// Kernel- and extra parameters
						);
				
				
			/*	gridSizeX = (int)Math.ceil(Math.sqrt(timeVector.length/(columns*rows))); 	// update.
				gridSizeY = gridSizeX;														// update.
				cuLaunchKernel(functionBpline,
						gridSizeX,  gridSizeY, 1, 	// Grid dimension
						blockSizeX, blockSizeY, 1,  // Block dimension
						0, null,               		// Shared memory size and stream
						kernelParametersBspline, null 		// Kernel- and extra parameters
						);
				*/cuCtxSynchronize();
				
				int hostOutput[] = new int[timeVector.length];

				cuMemcpyDtoH(Pointer.to(hostOutput), deviceOutputBSpline,
						bSplineDataLength * Sizeof.INT);
				// clean up GPU.
										
						
				cuMemFree(deviceOutput);  			
				cuMemFree(deviceFilterKernel);   
				int[] limits = findLimits.run(hostOutput, columns, rows, Ch); // get limits.				

				nData = bSplineDataLength/(columns*rows);
				blockSize = 256;
				gridSize = (nData + blockSize - 1)/blockSize;
				gridSize = (int) (Math.log(gridSize)/Math.log(2) + 1);
				if (gridSize > maxGrid)
					gridSize = (int)( Math.pow(2, maxGrid));
				else
					gridSize = (int)( Math.pow(2, gridSize));
				CUdeviceptr deviceLimits 	= CUDA.copyToDevice(limits);
				CUdeviceptr deviceCenter 	= CUDA.allocateOnDevice((int)(nMax*nCenter));
				Pointer kernelParameters 		= Pointer.to(   
						Pointer.to(new int[]{nData}),
						Pointer.to(deviceOutputBSpline),
						Pointer.to(new int[]{bSplineDataLength}),
						Pointer.to(new int[]{columns}),		 				       
						Pointer.to(new int[]{rows}),
						Pointer.to(new int[]{gWindow}),
						Pointer.to(deviceLimits),
						Pointer.to(new int[]{limits.length}),
						Pointer.to(new int[]{minPosPixels}),
						Pointer.to(new int[]{nCenter}),
						Pointer.to(deviceCenter),
						Pointer.to(new int[]{nMax*nCenter}));

				cuLaunchKernel(findMaximaFcn,
						gridSize,  1, 1, 	// Grid dimension
						blockSize, 1, 1,  // Block dimension
						0, null,               		// Shared memory size and stream
						kernelParameters, null 		// Kernel- and extra parameters
						);
				
		/*		blockSizeX 	= 1;
				blockSizeY 	= 1;				   
				gridSizeX 	= (int) Math.ceil((Math.sqrt(nMax)));
				gridSizeY 	= gridSizeX;

				cuLaunchKernel(findMaximaFcn,
						gridSizeX,  gridSizeY, 1, 	// Grid dimension
						blockSizeX, blockSizeY, 1,  // Block dimension
						0, null,               		// Shared memory size and stream
						kernelParameters, null 		// Kernel- and extra parameters
						);
				*/cuCtxSynchronize();

				int hostCenter[] = new int[nMax*nCenter];
				// Pull data from device.
				cuMemcpyDtoH(Pointer.to(hostCenter), deviceCenter,
						nMax*nCenter * Sizeof.INT);

				// Free up memory allocation on device, housekeeping.
				cuMemFree(deviceCenter);
				cuMemFree(deviceOutputBSpline);
				cuMemFree(deviceLimits);
				/*
				 * Depending on where in the frame loop we are, ignore edges if in the middle or include edge if required, median filtering yields differences.
				 */
				
				
				/******************************************************************************
				 * Transfer data for gauss fitting.
				 ******************************************************************************/
				int newN = 0;
				for (int j = 0; j < hostCenter.length; j++) // loop over all possible centers.
				{
					if (hostCenter[j]> 0) // if the center was added.
					{									
						newN++;
					}
				}
				int loaded = 0;
				int startIdx = 0;
				while (loaded < newN)
				{
					int maxLoad = 100000;
					if ((newN - loaded) < maxLoad)
						maxLoad = newN - loaded;
					int[] locatedCenter = new int[maxLoad]; // cleaned vector with indexes of centras.
					int[] locatedFrame = new int[maxLoad]; // cleaned vector with indexes of centras.
					int counter = 0;
					int j = startIdx;
					
					boolean fill = true;
					while (fill)
					{
						if (startFrame == 1 && endFrame < nFrames) // if the first frame is included in this batch but not the last.
						{
					//		System.out.println(" version 1" );
							if (hostCenter[j]> 0  && hostCenter[j]/(columns*rows) < endFrame - W[Ch-1])
							{
								locatedCenter[counter] = hostCenter[j];													
								locatedFrame[counter] = hostCenter[j]/(columns*rows);							
								counter++;
							}	
						}else if (startFrame == 1 && endFrame == nFrames) // if both first and last frame is included in the batch.
						{
						//	System.out.println(" version 2" );
							if (hostCenter[j]> 0)
							{
								locatedCenter[counter] = hostCenter[j];													
								locatedFrame[counter] = hostCenter[j]/(columns*rows);							
								counter++;
							}		
						}else if (startFrame > 1 && endFrame < nFrames) // if central fragment.
						{
							//System.out.println(" version 3" );
							if (hostCenter[j]> 0 && hostCenter[j]/(columns*rows) > W[Ch-1] && hostCenter[j]/(columns*rows) < endFrame - W[Ch-1])
							{
								locatedCenter[counter] = hostCenter[j];													
								locatedFrame[counter] = hostCenter[j]/(columns*rows);							
								counter++;
							}
						}else // only option left is that this is the final fragment, exlcuding the beginning.
						{
//							System.out.println(" version 4" );
							if (hostCenter[j]> 0 && hostCenter[j]/(columns*rows) > W[Ch-1])
							{
								locatedCenter[counter] = hostCenter[j];													
								locatedFrame[counter] = hostCenter[j]/(columns*rows);							
								counter++;
							}
						}
					

						j ++;
						if (counter == maxLoad || j == hostCenter.length)
							fill = false;										
					}
					startIdx = j;
					double[] P = new double[counter*7];
					double[] stepSize = new double[counter*7];							
					int[] gaussVector = new int[counter*gWindow*gWindow];

					for (int i = 0; i < counter; i++)
					{
						P[i*7] = hostOutput[locatedCenter[i]];
						P[i*7+1] = 2;
						P[i*7+2] = 2;
						P[i*7+3] = 1.5;
						P[i*7+4] = 1.5;
						P[i*7+6] = 0;
						P[i*7+6] = 0;
						stepSize[i * 7] = 0.1;// amplitude
						stepSize[i * 7 + 1] = 0.25*100/pixelSize; // x center.
						stepSize[i * 7 + 2] = 0.25*100/pixelSize; // y center.
						stepSize[i * 7 + 3] = 0.25*100/pixelSize; // sigma x.
						stepSize[i * 7 + 4] = 0.25*100/pixelSize; // sigma y.
						stepSize[i * 7 + 5] = 0.19625; // Theta.
						stepSize[i * 7 + 6] = 0.01; // offset.   
						int k = locatedCenter[i] - (gWindow / 2) * (columns + 1); // upper left corner.
						j = 0;
						int loopC = 0;
						while (k <= locatedCenter[i] + (gWindow / 2) * (columns + 1)) // loop over all relevant pixels. use this loop to extract data based on single indexing defined centers.
						{
							gaussVector[i * gWindow * gWindow + j] = hostOutput[k]; // add data.
							k++;
							loopC++;
							j++;
							if (loopC == gWindow)
							{
								k += (columns - gWindow);
								loopC = 0;
							}
						} // data pulled.
					}

					CUdeviceptr deviceGaussVector 	= 		CUDA.copyToDevice(gaussVector);					
					CUdeviceptr deviceP 			= 		CUDA.copyToDevice(P);
					CUdeviceptr deviceStepSize 		= 		CUDA.copyToDevice(stepSize);							

					/******************************************************************************
					 * Gauss fitting.
					 ******************************************************************************/
			/*		blockSizeX 	= 1;
					blockSizeY 	= 1;
					gridSizeX 	= 1000+(int) Math.ceil((Math.sqrt(counter)));
					gridSizeY 	= gridSizeX;
*/
					nData = counter;
					blockSize = 256;
					gridSize = (nData + blockSize - 1)/blockSize;
					gridSize = (int) (Math.log(gridSize)/Math.log(2) + 1);
					if (gridSize > maxGrid)
						gridSize = (int)( Math.pow(2, maxGrid));
					else
						gridSize = (int)( Math.pow(2, gridSize));
					Pointer kernelParametersGaussFit 		= Pointer.to(   
							Pointer.to(new int[]{nData}),
							Pointer.to(deviceGaussVector),
							Pointer.to(new int[]{counter * gWindow * gWindow}),
							Pointer.to(deviceP),																											
							Pointer.to(new double[]{counter*7}),
							Pointer.to(new short[]{(short) gWindow}),
							Pointer.to(deviceBounds),
							Pointer.to(new double[]{bounds.length}),
							Pointer.to(deviceStepSize),																											
							Pointer.to(new double[]{counter*7}),
							Pointer.to(new double[]{convCriteria}),
							Pointer.to(new int[]{maxIterations}));	
					cuLaunchKernel(fittingFcn,
							gridSize,  1, 1, 	// Grid dimension
							blockSize, 1, 1,  // Block dimension
							0, null,               		// Shared memory size and stream
							kernelParametersGaussFit, null 		// Kernel- and extra parameters
							);
	/*				cuLaunchKernel(fittingFcn,
							gridSizeX,  gridSizeY, 1, 	// Grid dimension
							blockSizeX, blockSizeY, 1,  // Block dimension
							0, null,               		// Shared memory size and stream
							kernelParametersGaussFit, null 		// Kernel- and extra parameters
							);
		*/			cuCtxSynchronize(); 

					double gaussFittingresults[] = new double[counter*7];

					// Pull data from device.
					cuMemcpyDtoH(Pointer.to(gaussFittingresults), deviceP,
							counter*7 * Sizeof.DOUBLE);
					// Free up memory allocation on device, housekeeping.
					cuMemFree(deviceGaussVector);   
					cuMemFree(deviceP);    
					cuMemFree(deviceStepSize);	
					for (int n = 0; n < counter; n++) //loop over all particles
					{	    	
						Particle Localized = new Particle();
						Localized.include 		= 1;
						Localized.channel 		= Ch;
						Localized.frame   		= startFrame + locatedFrame[n];//+ locatedCenter[n]/(columns*rows);
						Localized.r_square 		= gaussFittingresults[n*7+6];
						Localized.x				= pixelSize*(0.5 + gaussFittingresults[n*7+1] + (locatedCenter[n]%columns) - Math.round((gWindow)/2));
						Localized.y				= pixelSize*(0.5 + gaussFittingresults[n*7+2] + ((locatedCenter[n]/columns)%rows) - Math.round((gWindow)/2));
						Localized.z				= pixelSize*0;	// no 3D information.
						Localized.sigma_x		= pixelSize*gaussFittingresults[n*7+3];
						Localized.sigma_y		= pixelSize*gaussFittingresults[n*7+4];
						Localized.photons		= (int) (gaussFittingresults[n*7]/totalGain[Ch-1]);
						Localized.precision_x 	= Localized.sigma_x/Math.sqrt(Localized.photons);
						Localized.precision_y 	= Localized.sigma_y/Math.sqrt(Localized.photons);
						Results.add(Localized);
					}	
					loaded += maxLoad;
				}
		
				startFrame = endFrame-W[Ch-1]; // include W more frames to ensure that border errors from median calculations dont occur ore often then needed.
				endFrame += framesPerBatch;					
				if (endFrame > nFrames)
					endFrame = nFrames;

			} // while loadedChannels < nFrames
			// Free up memory allocation on device, housekeeping.
			cuMemFree(deviceBounds);
		} // Channel loop.				
//		image.updateAndDraw();	

		ArrayList<Particle> cleanResults = new ArrayList<Particle>();
		for (int i = 0; i < Results.size(); i++)
		{
			if (Results.get(i).x > 0 &&
					Results.get(i).y > 0 &&
					Results.get(i).z >= 0 &&
					Results.get(i).sigma_x > 0 &&
					Results.get(i).sigma_y > 0 &&
					Results.get(i).precision_x > 0 &&
					Results.get(i).precision_y > 0 &&
					Results.get(i).photons > 0 && 
					Results.get(i).r_square > 0)
				cleanResults.add(Results.get(i));

		}
		/*
		 * 
		 * Remove duplicates that can occur if two center pixels has the exact same value. Select the best fit if this is the case.
		 */
		int currFrame = -1;
		int i  = 0;
		int j = 0;
		int Ch = 1;

		int pixelDistance = 2*pixelSize*pixelSize;
		while( i < cleanResults.size())
		{				
			if (cleanResults.get(i).channel > Ch)
				pixelDistance = 2*pixelSize*pixelSize;
			if( cleanResults.get(i).frame > currFrame)
			{
				currFrame = cleanResults.get(i).frame;					
				j = i+1;

			}
			while (j < cleanResults.size() && cleanResults.get(j).frame == currFrame)
			{
				if (((cleanResults.get(i).x - cleanResults.get(j).x)*(cleanResults.get(i).x - cleanResults.get(j).x) + 
						(cleanResults.get(i).y - cleanResults.get(j).y)*(cleanResults.get(i).y - cleanResults.get(j).y)) < pixelDistance)
				{					
					if (cleanResults.get(i).r_square > cleanResults.get(j).r_square)
					{
						cleanResults.get(j).include = 0;
					}else
					{
						cleanResults.get(i).include = 0;					
					}
				}
				j++;
			}

			i++;
			j = i+1;
		}
		for (i = cleanResults.size()-1; i >= 0; i--)
		{
			if (cleanResults.get(i).include == 0)
				cleanResults.remove(i);
		}
		
		if (modality.equals("2D"))
		{
			cleanResults = BasicFittingCorrections.compensate(cleanResults); // change 2D data to 3D data based on calibration data.
		}
		else if (modality.equals("PRILM"))
		{
			cleanResults = PRILMfitting.fit(cleanResults); // change 2D data to 3D data based on calibration data.
		}
		else if (modality.equals("Biplane"))
		{
			cleanResults = BiplaneFitting.fit(cleanResults,pixelSize,totalGain); // change 2D data to 3D data based on calibration data.
			columns /= 2;
		}
		else if (modality.equals("Double Helix"))
		{
			cleanResults = DoubleHelixFitting.fit(cleanResults); // change 2D data to 3D data based on calibration data.
		}
		else if (modality.equals("Astigmatism"))
		{
			cleanResults =AstigmatismFitting.fit(cleanResults); // change 2D data to 3D data based on calibration data.
		}
		ij.measure.ResultsTable tab = Analyzer.getResultsTable();
		tab.reset();		
		tab.incrementCounter();
		tab.addValue("width", columns*pixelSize);
		tab.addValue("height", rows*pixelSize);
		TableIO.Store(cleanResults);
	} // end run
}
