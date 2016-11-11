import static jcuda.driver.JCudaDriver.cuCtxCreate;
import static jcuda.driver.JCudaDriver.cuCtxSynchronize;
import static jcuda.driver.JCudaDriver.cuDeviceGet;
import static jcuda.driver.JCudaDriver.cuInit;
import static jcuda.driver.JCudaDriver.cuLaunchKernel;
import static jcuda.driver.JCudaDriver.cuMemFree;
import static jcuda.driver.JCudaDriver.cuMemcpyDtoH;
import static jcuda.driver.JCudaDriver.cuModuleGetFunction;
import static jcuda.driver.JCudaDriver.cuModuleLoad;

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


/*
 * ProcessMedianFit is called by the process call button and will in sequence load image data, median filter (background removal), locate events and fit them, returning a result table.
 */
public class processMedianFit {

	public static void run(final int[] W, ImagePlus image, int selectedModel, int[] MinLevel, double[] sqDistance, int[] gWindow, int[] inputPixelSize, int[] minPosPixels, int[] totalGain)
	{
		int columns 						= image.getWidth();
		int rows 							= image.getHeight();		
		int nChannels 						= image.getNChannels(); 	// Number of channels.
		int nFrames 						= image.getNFrames();
		if (nFrames == 1)
			nFrames 						= image.getNSlices();  
		double convCriteria = 1E-8; // how large improvement from one step to next we require.
		int maxIterations = 1000;  // stop if an individual fit reaches this number of iterati
		ArrayList<Particle> Results 		= new ArrayList<Particle>();		// Fitted results array list.
		if(selectedModel == 2) // GPU
		{			
			// Initialize the driver and create a context for the first device.
			cuInit(0);
			CUdevice device = new CUdevice();
			cuDeviceGet(device, 0);
			CUcontext context = new CUcontext();
			cuCtxCreate(context, 0, device);
			// Load the PTX that contains the kernel.
			CUmodule moduleMedianFilter = new CUmodule();
			cuModuleLoad(moduleMedianFilter, "medianFilter.ptx");
			// Obtain a handle to the kernel function.
			CUfunction functionMedianFilter = new CUfunction();
			cuModuleGetFunction(functionMedianFilter, moduleMedianFilter, "medianKernel");
			// gauss fit algorithm.
			// Load the PTX that contains the kernel.
			CUmodule module = new CUmodule();
			cuModuleLoad(module, "gFit.ptx");
			// Obtain a handle to the kernel function.
			CUfunction fittingFcn = new CUfunction();
			cuModuleGetFunction(fittingFcn, module, "gaussFitter");  //gFit.pth (gaussFitter function).

			// prepare for locating centra for gaussfitting.
			// Load the PTX that contains the kernel.
			CUmodule moduleLM = new CUmodule();
			cuModuleLoad(moduleLM, "findMaxima.ptx");					
			// Obtain a handle to the kernel function
			CUfunction findMaximaFcn = new CUfunction();
			cuModuleGetFunction(findMaximaFcn, moduleLM, "run");	// findMaxima.ptx (run function).
			CUmodule modulePrepGauss = new CUmodule();
			cuModuleLoad(modulePrepGauss, "prepareGaussian.ptx");					
			// Obtain a handle to the kernel function
			CUfunction prepareGaussFcn = new CUfunction();
			cuModuleGetFunction(prepareGaussFcn, modulePrepGauss, "run");	// prepareGaussian.ptx (run function).
			long GB = 1024*1024*1024;
			int frameSize = (2*columns*rows + 1)*Sizeof.FLOAT;
			for(int Ch = 1; Ch <= nChannels; Ch++)
			{
				int nCenter =(( columns*rows/(gWindow[Ch-1]*gWindow[Ch-1])) / 2); // ~ 80 possible particles for a 64x64 frame. Lets the program scale with frame size.
				int staticMemory = (2*W[Ch-1]+1*rows*columns)*Sizeof.FLOAT;
				long framesPerBatch = (3*GB-frameSize)/staticMemory; // 3 GB memory allocation gives this numbers of frames. 
				int loadedFrames = 0;
				int startFrame = 1;
				int endFrame = (int)framesPerBatch;					
				if (endFrame > nFrames)
					endFrame = nFrames;
				CUdeviceptr device_window 		= CUDA.allocateOnDevice((float)((2 * W[Ch-1] + 1) * rows * columns)); // swap vector.
				float[] bounds = { // bounds for gauss fitting.
						0.5F			, 1.5F,				// amplitude.
						1	,(float)(gWindow[Ch-1]-1),			// x.
						1	, (float)(gWindow[Ch-1]-1),			// y.
						0.7F			, (float) (gWindow[Ch-1] / 2.0),		// sigma x.
						0.7F			, (float) (gWindow[Ch-1] / 2.0),		// sigma y.
						(float) (-0.5*Math.PI) ,(float) (0.5*Math.PI),	// theta.
						-0.5F		, 0.5F				// offset.
				};
				CUdeviceptr deviceBounds 		= CUDA.copyToDevice(bounds);	
				while (loadedFrames < nFrames-1)
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

					
					CUdeviceptr device_Data 		= CUDA.copyToDevice(timeVector);
					CUdeviceptr device_meanVector 	= CUDA.copyToDevice(MeanFrame);
					CUdeviceptr deviceOutput 		= CUDA.allocateOnDevice((int)timeVector.length);

					int filteWindowLength 		= (2 * W[Ch-1] + 1) * rows * columns;
					int testDataLength 			= timeVector.length;
					int meanVectorLength 		= MeanFrame.length;
					Pointer kernelParametersMedianFilter 	= Pointer.to(   
							Pointer.to(new int[]{W[Ch]}),
							Pointer.to(device_window),
							Pointer.to(new int[]{filteWindowLength}),
							Pointer.to(new int[]{(meanVectorLength)}),
							Pointer.to(device_Data),
							Pointer.to(new int[]{testDataLength}),
							Pointer.to(device_meanVector),
							Pointer.to(new int[]{meanVectorLength}),
							Pointer.to(deviceOutput),
							Pointer.to(new int[]{testDataLength})
							);
					int blockSizeX 	= 1;
					int blockSizeY 	= 1;				   
					int gridSizeX 	= columns;
					int gridSizeY 	= rows;
					cuLaunchKernel(functionMedianFilter,
							gridSizeX,  gridSizeY, 1, 	// Grid dimension
							blockSizeX, blockSizeY, 1,  // Block dimension
							0, null,               		// Shared memory size and stream
							kernelParametersMedianFilter, null 		// Kernel- and extra parameters
							);
					cuCtxSynchronize();

					// Pull data from device.
					int hostOutput[] = new int[timeVector.length];
					cuMemcpyDtoH(Pointer.to(hostOutput), deviceOutput,
							timeVector.length * Sizeof.INT);

					// Free up memory allocation on device, housekeeping.
					   
					cuMemFree(device_Data);    
					cuMemFree(deviceOutput);
					cuMemFree(device_meanVector);
					
					/********************************************************************************
					 * 
					 * 								Localize events
					 * 
					 ********************************************************************************/
					
					CUdeviceptr deviceData 	= CUDA.copyToDevice(hostOutput);
					CUdeviceptr deviceCenter = CUDA.allocateOnDevice((int)(meanVectorLength*nCenter));

					Pointer kernelParameters 		= Pointer.to(   
							Pointer.to(deviceData),
							Pointer.to(new int[]{hostOutput.length}),
							Pointer.to(new int[]{columns}),		 				       
							Pointer.to(new int[]{rows}),
							Pointer.to(new int[]{gWindow[Ch-1]}),
							Pointer.to(new int[]{MinLevel[Ch-1]}),
							Pointer.to(new double[]{sqDistance[Ch-1]}),
							Pointer.to(new int[]{minPosPixels[Ch-1]}),
							Pointer.to(new int[]{nCenter}),
							Pointer.to(deviceCenter),
							Pointer.to(new int[]{meanVectorLength*nCenter}));

					blockSizeX 	= 1;
					blockSizeY 	= 1;				   
					gridSizeX 	= (int) Math.ceil((Math.sqrt(meanVectorLength)));
					gridSizeY 	= gridSizeX;

					cuLaunchKernel(findMaximaFcn,
							gridSizeX,  gridSizeY, 1, 	// Grid dimension
							blockSizeX, blockSizeY, 1,  // Block dimension
							0, null,               		// Shared memory size and stream
							kernelParameters, null 		// Kernel- and extra parameters
							);
					cuCtxSynchronize();

					int hostCenter[] = new int[meanVectorLength*nCenter];
					// Pull data from device.
					cuMemcpyDtoH(Pointer.to(hostCenter), deviceCenter,
							meanVectorLength*nCenter * Sizeof.INT);

					// Free up memory allocation on device, housekeeping.
					cuMemFree(deviceCenter);
					
					cuMemFree(deviceData);
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

					int[] locatedCenter = new int[newN]; // cleaned vector with indexes of centras.
					int counter = 0;
					for (int j = 0; j < hostCenter.length; j++) // loop over all possible centers.
					{
						if (hostCenter[j]> 0 && counter < newN) // if the center was added.
						{
							locatedCenter[counter] = hostCenter[j];								
							counter++;
						}
							
					} // locatedCenter now populated.
					float[] P = new float[newN*7];
					float[] stepSize = new float[newN*7];							
					int[] gaussVector = new int[newN*gWindow[Ch-1]*gWindow[Ch-1]];
					
					for (int i = 0; i < newN; i++)
					{
						P[i*7] = hostOutput[locatedCenter[i]];
						P[i*7+1] = 2;
						P[i*7+2] = 2;
						P[i*7+3] = 2;
						P[i*7+4] = 2;
						P[i*7+6] = 0;
						P[i*7+6] = 0;
		                stepSize[i * 7] = 0.1F;// amplitude
		                stepSize[i * 7 + 1] = 0.25F; // x center.
		                stepSize[i * 7 + 2] = 0.25F; // y center.
		                stepSize[i * 7 + 3] = 0.25F; // sigma x.
		                stepSize[i * 7 + 4] = 0.25F; // sigma y.
		                stepSize[i * 7 + 5] = 0.19625F; // Theta.
		                stepSize[i * 7 + 6] = 0.01F; // offset.   
		                int k = locatedCenter[i] - (gWindow[Ch-1] / 2) * (columns + 1); // upper left corner.
		                int j = 0;
		                int loopC = 0;
		                while (k <= locatedCenter[i] + (gWindow[Ch-1] / 2) * (columns + 1)) // loop over all relevant pixels. use this loop to extract data based on single indexing defined centers.
		                {
		                    gaussVector[i * gWindow[Ch-1] * gWindow[Ch-1] + j] = hostOutput[k]; // add data.
		                    k++;
		                    loopC++;
		                    j++;
		                    if (loopC == gWindow[Ch-1])
		                    {
		                        k += (columns - gWindow[Ch-1]);
		                        loopC = 0;
		                    }
		                } // data pulled.
					}
					CUdeviceptr deviceGaussVector 	= 		CUDA.copyToDevice(gaussVector);					
					CUdeviceptr deviceP 			= 		CUDA.copyToDevice(P);
					CUdeviceptr deviceStepSize 		= 		CUDA.copyToDevice(stepSize);
					
					/********************************************************************************
					 * 
					 * 								Fit events
					 * 
					 ********************************************************************************/
					
					gridSizeX = (int) Math.ceil((Math.sqrt(newN)));
					gridSizeY 	= gridSizeX;
					Pointer kernelParametersGaussFit 		= Pointer.to(   
							Pointer.to(deviceGaussVector),
							Pointer.to(new int[]{newN * gWindow[Ch-1] * gWindow[Ch-1]}),
							Pointer.to(deviceP),																											
							Pointer.to(new float[]{newN*7}),
							Pointer.to(new short[]{(short) gWindow[Ch-1]}),
							Pointer.to(deviceBounds),
							Pointer.to(new float[]{bounds.length}),
							Pointer.to(deviceStepSize),																											
							Pointer.to(new float[]{newN*7}),
							Pointer.to(new double[]{convCriteria}),
							Pointer.to(new int[]{maxIterations}));	


					cuLaunchKernel(fittingFcn,
							gridSizeX,  gridSizeY, 1, 	// Grid dimension
							blockSizeX, blockSizeY, 1,  // Block dimension
							0, null,               		// Shared memory size and stream
							kernelParametersGaussFit, null 		// Kernel- and extra parameters
							);
					//cuCtxSynchronize(); 

					float hostParameterOutput[] = new float[newN*7];

					// Pull data from device.
					cuMemcpyDtoH(Pointer.to(hostParameterOutput), deviceP,
							newN*7 * Sizeof.FLOAT);
			//		for(int i = 0; i < hostOutput.length; i+=7)
			//			System.out.println(hostOutput[i]);
					// Free up memory allocation on device, housekeeping.
					cuMemFree(deviceGaussVector);   
					cuMemFree(deviceP);    
					cuMemFree(deviceStepSize);
					
					for (int n = 0; n < newN; n++) //loop over all particles
					{	    	
						Particle Localized = new Particle();
						Localized.include 		= 1;
						Localized.channel 		= Ch;
						Localized.frame   		= startFrame + locatedCenter[n]/(columns*rows);
						Localized.r_square 		= hostParameterOutput[n*7+6];
						Localized.x				= inputPixelSize[Ch-1]*(hostParameterOutput[n*7+1] + (locatedCenter[n]%columns) - Math.round((gWindow[Ch-1])/2));
						Localized.y				= inputPixelSize[Ch-1]*(hostParameterOutput[n*7+2] + ((locatedCenter[n]/columns)%rows) - Math.round((gWindow[Ch-1])/2));
						Localized.z				= inputPixelSize[Ch-1]*0;	// no 3D information.
						Localized.sigma_x		= inputPixelSize[Ch-1]*hostParameterOutput[n*7+3];
						Localized.sigma_y		= inputPixelSize[Ch-1]*hostParameterOutput[n*7+4];
						Localized.sigma_z		= inputPixelSize[Ch-1]*0; // no 3D information.
						Localized.photons		= (int) (hostParameterOutput[n*7]/totalGain[Ch-1]);
						Localized.precision_x 	= Localized.sigma_x/Math.sqrt(Localized.photons);
						Localized.precision_y 	= Localized.sigma_y/Math.sqrt(Localized.photons);
						Localized.precision_z 	= Localized.sigma_z/Math.sqrt(Localized.photons);
						Results.add(Localized);
					}		
					
					
					startFrame = endFrame-W[Ch-1]; // include W more frames to ensure that border errors from median calculations dont occur ore often then needed.
					endFrame += framesPerBatch;					
					if (endFrame > nFrames)
						endFrame = nFrames;
				} // while loadedChannels < nFrames
				// Free up memory allocation on device, housekeeping.
				cuMemFree(device_window);
			} // Channel loop.	
		
			ArrayList<Particle> cleanResults = new ArrayList<Particle>();
			for (int i = 0; i < Results.size(); i++)
			{
				if (Results.get(i).sigma_x > 0 &&
						Results.get(i).sigma_y > 0 &&
						Results.get(i).precision_x > 0 &&
						Results.get(i).precision_y > 0 &&
						Results.get(i).photons > 0 && 
						Results.get(i).r_square > 0)
					cleanResults.add(Results.get(i));

			}
			ij.measure.ResultsTable tab = Analyzer.getResultsTable();
			tab.reset();		
			tab.incrementCounter();
			tab.addValue("width", columns*inputPixelSize[0]);
			tab.addValue("height", rows*inputPixelSize[0]);
			TableIO.Store(cleanResults);

		} // GPU computing.
	}


}
