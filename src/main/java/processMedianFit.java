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
import ij.process.ImageProcessor;
import ij.process.ImageStatistics;
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
 * ProcessMedianFit is called by the process call button and will in a single load from the imagestack.
 * TODO: Change medianFilter.ptx to yield output organized per frame and not pixel for prepared input to gaussfit. 
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
		ArrayList<Particle> Results 		= new ArrayList<Particle>();		// Fitted results array list.
		if(selectedModel == 2) // GPU
		{
			// TODO look over kernel for errors. Change median from int to float. 
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

			for(int Ch = 1; Ch <= nChannels; Ch++)
			{
				/*****************************************************************************
				 * 
				 * 							Correct background
				 * 
				 *****************************************************************************/ 
				
				float[] timeVector = new float[nFrames * rows * columns];
				float[] MeanFrame = new float[nFrames]; 		// Will include frame mean value.
				ImageProcessor IP = image.getProcessor();
				for (int Frame = 1; Frame <= nFrames; Frame++)
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
						timeVector[(Frame-1) + nFrames*i] = IP.get(i);			
						MeanFrame[Frame-1] += IP.get(i); 
					}
					MeanFrame[Frame-1] /= columns*rows;
					
							
				} // frame loop for mean calculations.

				CUdeviceptr device_window 		= CUDA.allocateOnDevice((float)((2 * W[Ch] + 1) * rows * columns)); // swap vector.
				CUdeviceptr device_Data 		= CUDA.copyToDevice(timeVector);
				CUdeviceptr device_meanVector 	= CUDA.copyToDevice(MeanFrame);
				CUdeviceptr deviceMedianFiltImage 		= CUDA.allocateOnDevice((int)timeVector.length);

				int filteWindowLength 		= (2 * W[0] + 1) * rows * columns;
				int medianFiltImageLength 			= timeVector.length;
				int meanVectorLength 		= MeanFrame.length;
				Pointer kernelParameters 	= Pointer.to(   
						Pointer.to(new int[]{W[Ch]}),
						Pointer.to(device_window),
						Pointer.to(new int[]{filteWindowLength}),
						Pointer.to(new int[]{nFrames}),
						Pointer.to(device_Data),
						Pointer.to(new int[]{medianFiltImageLength}),
						Pointer.to(device_meanVector),
						Pointer.to(new int[]{meanVectorLength}),
						Pointer.to(deviceMedianFiltImage),
						Pointer.to(new int[]{medianFiltImageLength})
						);
				int blockSizeX 	= 1;
				int blockSizeY 	= 1;				   
				int gridSizeX 	= columns;
				int gridSizeY 	= rows;
				cuLaunchKernel(functionMedianFilter,
						gridSizeX,  gridSizeY, 1, 	// Grid dimension
						blockSizeX, blockSizeY, 1,  // Block dimension
						0, null,               		// Shared memory size and stream
						kernelParameters, null 		// Kernel- and extra parameters
						);
				cuCtxSynchronize();
				// Free up memory allocation on device, housekeeping.
				cuMemFree(device_window);   
				cuMemFree(device_Data);    
				
				/*
				 * Data already on device will be used for locating maxima.
				 */
				
				/*****************************************************************************
				 * 
				 * 						Locate particles and fit
				 * 
				 *****************************************************************************/ 
				
				
				
				// TODO: Change findMaxima to handle new input. Verify that xy does not need to be altered and give output as pixel idx. Create new function to pull out data around center pixels.
				
				
				double convCriteria = 1E-8; // how large improvement from one step to next we require.
				int maxIterations = 1000;  // stop if an individual fit reaches this number of iterations.

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

				int nCenter =2*(( columns*rows/(gWindow[Ch-1]*gWindow[Ch-1])) / 2); // ~ 80 possible particles for a 64x64 frame. Lets the program scale with frame size.
				double gb = 1024*1024*1024;
				double maxMemoryGPU = 3*gb; // TODO: get size of gpu memory.

				ArrayList<fitParameters> fitThese = new ArrayList<fitParameters>();
				int dataIdx = 0;
				int idx = 0;

				int loopStartFrame = 0;
				boolean processed = true;

				CUdeviceptr deviceCenter = CUDA.allocateOnDevice((int)(nFrames*nCenter));

				Pointer kernelParametersLocateMaxima 		= Pointer.to(   
						Pointer.to(deviceMedianFiltImage),
						Pointer.to(new int[]{medianFiltImageLength}),
						Pointer.to(new int[]{columns}),		 				       
						Pointer.to(new int[]{rows}),
						Pointer.to(new int[]{gWindow[Ch-1]}),
						Pointer.to(new int[]{MinLevel[Ch-1]}),
						Pointer.to(new double[]{sqDistance[Ch-1]}),
						Pointer.to(new int[]{minPosPixels[Ch-1]}),
						Pointer.to(new int[]{nCenter}),
						Pointer.to(deviceCenter),
						Pointer.to(new int[]{nFrames*nCenter}));

				blockSizeX 	= 1;
				blockSizeY 	= 1;				   
				gridSizeX 	= (int) Math.ceil((Math.sqrt(nFrames)));
				gridSizeY 	= gridSizeX;
				cuLaunchKernel(findMaximaFcn,
						gridSizeX,  gridSizeY, 1, 	// Grid dimension
						blockSizeX, blockSizeY, 1,  // Block dimension
						0, null,               		// Shared memory size and stream
						kernelParametersLocateMaxima, null 		// Kernel- and extra parameters
						);
				cuCtxSynchronize();

				int hostCenter[] = new int[nFrames*nCenter];

				// Pull data from device.
				cuMemcpyDtoH(Pointer.to(hostCenter), deviceCenter,
						nFrames*nCenter * Sizeof.INT);

				// Free up memory allocation on device, housekeeping.
				//cuMemFree(deviceData);   
				cuMemFree(deviceCenter);
				cuMemFree(deviceMedianFiltImage); // main image list.
				// add adding of fitThese.
				int centerIdx = 0;
				while (centerIdx <hostCenter.length)
				{
					if (hostCenter[centerIdx] >  0)
					{										
						int[] coord = {hostCenter[centerIdx],hostCenter[centerIdx+1]};;
						fitParameters fitObj = new fitParameters();
						fitObj.Center =  coord;
						fitObj.frame = loopStartFrame + centerIdx/ nCenter;
						fitObj.pixelsize = inputPixelSize[Ch-1];
						fitObj.windowWidth = gWindow[Ch-1];
						fitObj.totalGain = totalGain[Ch-1];
						fitObj.channel = Ch;										
						fitThese.add(fitObj);
					}
					centerIdx += 2;
				} // adding data to fitObj list.
			
				
				// TODO: add loop of above over frame to ensure that the gpu can handle data volume. Extend frame for windowwidth to cover edge artefacts and ignore duplicates.
				// pull out all data.

				int gIdx = 0;
				int N = fitThese.size(); // number of particles to be fitted.						
				float[] parameters = new float[N*7];
				float[] bounds = {
						0.5F			, 1.5F,				// amplitude.
						1	,(float)(gWindow[Ch-1]-1),			// x.
						1	, (float)(gWindow[Ch-1]-1),			// y.
						0.7F			, (float) (gWindow[Ch-1] / 2.0),		// sigma x.
						0.7F			, (float) (gWindow[Ch-1] / 2.0),		// sigma y.
						(float) (-0.5*Math.PI) ,(float) (0.5*Math.PI),	// theta.
						-0.5F		, 0.5F				// offset.
				};

				float scale = 100/inputPixelSize[Ch-1];
				float[] stepSize = new float[N*7];
				int gWindowSquare = gWindow[Ch-1]*gWindow[Ch-1];
				int[] gaussVector = new int[fitThese.size()*gWindowSquare]; // preallocate.
				int gCenter = gWindow[Ch-1]*(gWindow[Ch-1]-1)/2 + (gWindow[Ch-1]-1)/2;
				for (int n = 0; n < N; n++) //loop over all parameters to set up calculations:
				{
					int x0 = fitThese.get(n).Center[0] - gWindow[Ch-1]/2; // upper left corner of region.
					int y0 = fitThese.get(n).Center[1] - gWindow[Ch-1]/2; // upper left corner of region.		
					image.setPosition(
							Ch,			// channel.
							1,			// slice.
							fitThese.get(n).frame);		// frame.
					IP = image.getProcessor();

					for (int j = 0; j < gWindowSquare; j++ )
					{

						int xi = (j % gWindow[Ch-1]) + x0; // step through all points in the area.
						int yi = (j / gWindow[Ch-1]) + y0; // step through all points in the area.

						gaussVector[gIdx] = IP.get(xi, yi); // takes correct datapoints.
						gIdx++;
					}								
					// start parameters for fit:
					parameters[n*7] = gaussVector[n*gWindowSquare+ gCenter]; // read in center pixel.
					parameters[n*7+1] = 2; // x center, will be calculated as weighted centroid on GPU.
					parameters[n*7+2] = 2; // y center, will be calculated as weighted centroid on GPU.
					parameters[n*7+3] = 2.0F; // x sigma.
					parameters[n*7+4] = 2.0F; // y sigma.
					parameters[n*7+5] = 0; // theta.
					parameters[n*7+6] = 0; // offset.
					stepSize[n*7] 	 = 0.1F; // amplitude
					stepSize[n*7+1] 	= (float) (0.25*scale); // x
					stepSize[n*7+2] 	= (float) (0.25*scale); // y
					stepSize[n*7+3] 	= (float) (0.5*scale); // sigma x
					stepSize[n*7+4] 	= (float) (0.5*scale); // sigma y
					stepSize[n*7+5] 	= 0.1965F; //theta;
					stepSize[n*7+6] 	= 0.01F; // offset.

				}

				CUdeviceptr deviceGaussVector 	= CUDA.copyToDevice(gaussVector);
				CUdeviceptr deviceParameters 	= CUDA.copyToDevice(parameters);
				CUdeviceptr deviceStepSize 		= CUDA.copyToDevice(stepSize);
				CUdeviceptr deviceBounds 		= CUDA.copyToDevice(bounds);
				Pointer kernelParametersFit 		= Pointer.to(   
						Pointer.to(deviceGaussVector),
						Pointer.to(new int[]{gaussVector.length}),
						Pointer.to(deviceParameters),
						Pointer.to(new int[]{parameters.length}),
						Pointer.to(new int[]{gWindow[Ch-1]}),
						Pointer.to(deviceBounds),
						Pointer.to(new int[]{bounds.length}),
						Pointer.to(deviceStepSize),
						Pointer.to(new int[]{stepSize.length}),
						Pointer.to(new double[]{convCriteria}),
						Pointer.to(new int[]{maxIterations}));	

				blockSizeX 	= 1;
				blockSizeY 	= 1;				   
				gridSizeX 	= (int) Math.ceil(Math.sqrt(N));
				gridSizeY 	= gridSizeX;
				cuLaunchKernel(fittingFcn,
						gridSizeX,  gridSizeY, 1, 	// Grid dimension
						blockSizeX, blockSizeY, 1,  // Block dimension
						0, null,               		// Shared memory size and stream
						kernelParametersFit, null 		// Kernel- and extra parameters
						);
				cuCtxSynchronize();

				float hostOutput[] = new float[parameters.length];

				// Pull data from device.
				cuMemcpyDtoH(Pointer.to(hostOutput), deviceParameters,
						parameters.length * Sizeof.FLOAT);

				// Free up memory allocation on device, housekeeping.
				cuMemFree(deviceGaussVector);   
				cuMemFree(deviceParameters);    
				cuMemFree(deviceStepSize);
				cuMemFree(deviceBounds);
				for (int n = 0; n < N; n++) //loop over all particles
				{	    	
					Particle Localized = new Particle();
					Localized.include 		= 1;
					Localized.channel 		= fitThese.get(n).channel;
					Localized.frame   		= fitThese.get(n).frame;
					Localized.r_square 		= hostOutput[n*7+6];
					Localized.x				= inputPixelSize[Ch-1]*(hostOutput[n*7+1] + fitThese.get(n).Center[0] - Math.round((gWindow[Ch-1])/2));
					Localized.y				= inputPixelSize[Ch-1]*(hostOutput[n*7+2] + fitThese.get(n).Center[1] - Math.round((gWindow[Ch-1])/2));
					Localized.z				= inputPixelSize[Ch-1]*0;	// no 3D information.
					Localized.sigma_x		= inputPixelSize[Ch-1]*hostOutput[n*7+3];
					Localized.sigma_y		= inputPixelSize[Ch-1]*hostOutput[n*7+4];
					Localized.sigma_z		= inputPixelSize[Ch-1]*0; // no 3D information.
					Localized.photons		= (int) (hostOutput[n*7]/fitThese.get(n).totalGain);
					Localized.precision_x 	= Localized.sigma_x/Math.sqrt(Localized.photons);
					Localized.precision_y 	= Localized.sigma_y/Math.sqrt(Localized.photons);
					Localized.precision_z 	= Localized.sigma_z/Math.sqrt(Localized.photons);
					Results.add(Localized);
				}

			} // loop over all channels.

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

			TableIO.Store(cleanResults); // output.


		} // GPU computing.
	}


}
