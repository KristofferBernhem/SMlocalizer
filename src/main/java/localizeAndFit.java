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
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import ij.ImagePlus;
import ij.WindowManager;
import ij.process.ImageProcessor;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;

// TODO removal of duplicates not functional in ptx locate maxima.
public class localizeAndFit {

	public static ArrayList<Particle> run(int[] MinLevel, double[] sqDistance, int[] gWindow, int[] inputPixelSize, int[] minPosPixels, int[] totalGain , int selectedModel){				

		ImagePlus image 					= WindowManager.getCurrentImage();
		int columns 						= image.getWidth();
		int rows 							= image.getHeight();
		
		int nChannels 						= image.getNChannels(); 	// Number of channels.
		int nFrames 						= image.getNFrames();
		if (nFrames == 1)
			nFrames 						= image.getNSlices();  	
		ArrayList<Particle> Results 		= new ArrayList<Particle>();		// Fitted results array list.

		if (selectedModel == 1) // sequential
		{
			int[][][][] inputArray 				= TranslateIm.ReadIm();
			ArrayList<Particle> cleanResults = new ArrayList<Particle>();
			for (int Ch = 1; Ch <= nChannels; Ch++)							// Loop over all channels.
			{
				for (int Frame = 1; Frame <= nFrames;Frame++)					// Loop over all frames.
				{
					int[][] DataArray = new int[inputArray.length][inputArray[0].length];
					for (int x = 0; x < inputArray.length; x++)
					{
						for (int y = 0; y < inputArray[0].length; y++)
						{
							DataArray[x][y] = inputArray[x][y][Frame-1][Ch - 1];
						} // y loop.
					} // x loop.

					ArrayList<int[]> Center 	= LocalMaxima.FindMaxima(DataArray, gWindow[Ch-1], MinLevel[Ch-1], sqDistance[Ch-1], minPosPixels[Ch-1]); 	// Get possibly relevant center coordinates.

					for (int i = 0; i < Center.size(); i++)
					{
						int[] dataFit = new int[gWindow[Ch-1]*gWindow[Ch-1]];							// Container for data to be fitted.
						int[] Coord = Center.get(i);									// X and Y coordinates for center pixels to be fitted.

						for (int j = 0; j < gWindow[Ch-1]*gWindow[Ch-1]; j++)
						{
							int x =  Coord[0] - Math.round((gWindow[Ch-1])/2) +  (j % gWindow[Ch-1]);
							int y =  Coord[1] - Math.round((gWindow[Ch-1])/2) +  (j / gWindow[Ch-1]);
							dataFit[j] = inputArray[x][y][Frame-1][Ch - 1];
						} // pull out data for this fit.
						fitParameters fitThese = new fitParameters(Coord, 
								dataFit,
								Ch,
								Frame,
								inputPixelSize[Ch-1],
								gWindow[Ch-1],
								totalGain[Ch-1]);
						Particle tempParticle = ParticleFitter.Fitter(fitThese);
						if (tempParticle.sigma_x > 0 &&
								tempParticle.sigma_y > 0 &&
								tempParticle.precision_x > 0 &&
								tempParticle.precision_y > 0 &&
								tempParticle.photons > 0 && 
								tempParticle.r_square > 0)

							cleanResults.add(tempParticle);
					} // loop over all located centers from this frame.									
				} // loop over all frames.					
			} // loop over all channels.
			return cleanResults; // end parallel computation by returning results.
		}else // sequential 
			if (selectedModel == 0) // parallel
			{
				ImageProcessor IP = image.getProcessor();
				ArrayList<fitParameters> fitThese 	= new ArrayList<fitParameters>(); 	// arraylist to hold all fitting parameters.
				for (int Ch = 1; Ch <= nChannels; Ch++)							// Loop over all channels.
				{
					for (int Frame = 1; Frame <= nFrames;Frame++)					// Loop over all frames.
					{											
						image.setPosition(
								Ch,			// channel.
								1,			// slice.
								Frame);		// frame.
						IP = image.getProcessor();					
						int[][] Arr = IP.getIntArray();
						
						ArrayList<int[]> Center 	= LocalMaxima.FindMaxima(Arr, gWindow[Ch-1], MinLevel[Ch-1], sqDistance[Ch-1], minPosPixels[Ch-1]); 	// Get possibly relevant center coordinates.
						
						for (int i = 0; i < Center.size(); i++)
						{
							
							int[] dataFit = new int[gWindow[Ch-1]*gWindow[Ch-1]];			// Container for data to be fitted.
							int[] Coord = Center.get(i);									// X and Y coordinates for center pixels to be fitted.

							for (int j = 0; j < gWindow[Ch-1]*gWindow[Ch-1]; j++)
							{
								
								int x =  Coord[0] - Math.round((gWindow[Ch-1])/2) +  (j % gWindow[Ch-1]);
								int y =  Coord[1] - Math.round((gWindow[Ch-1])/2) +  (j / gWindow[Ch-1]);
								
								dataFit[j] = Arr[x][y];	// Faster then pulling from IP.get again.
							} // pull out data for this fit.

							fitThese.add(new fitParameters(Coord, 
									dataFit,
									Ch,
									Frame,
									inputPixelSize[Ch-1],
									gWindow[Ch-1],
									totalGain[Ch-1]));

						} // loop over all located centers from this frame.									
					} // loop over all frames.					
				} // loop over all channels.

				List<Callable<Particle>> tasks = new ArrayList<Callable<Particle>>();	// Preallocate.
				for (final fitParameters object : fitThese) {							// Loop over and setup computation.
					Callable<Particle> c = new Callable<Particle>() {					// Computation to be done.
						@Override
						public Particle call() throws Exception {
							return ParticleFitter.Fitter(object);						// Actual call for each parallel process.
						}
					};
					tasks.add(c);														// Que this task.
				} // setup parallel computing, distribute all fitThese objects. 

				int processors 			= Runtime.getRuntime().availableProcessors();	// Number of processor cores on this system.
				ExecutorService exec 	= Executors.newFixedThreadPool(processors);		// Set up parallel computing using all cores.
				try {

					List<Future<Particle>> parallelCompute = exec.invokeAll(tasks);				// Execute computation.    
					for (int i = 0; i < parallelCompute.size(); i++){							// Loop over and transfer results.
						try {
							Results.add(parallelCompute.get(i).get());							// Add computed results to Results arraylist.
						} catch (ExecutionException e) {
							e.printStackTrace();
						}
					}


				} catch (InterruptedException e) {

					e.printStackTrace();
				}
				finally {
					exec.shutdown();
				} // try executing parallel computation.


				//TableIO.Store(Results);												// Return and display results to user.
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
				return cleanResults; // end parallel computation by returning results.
			}else // end parallel. 
				if (selectedModel == 2) // GPU TODO: ADd image loading to GPU bound code. 
				{			
					double convCriteria = 1E-8; // how large improvement from one step to next we require.
					int maxIterations = 1000;  // stop if an individual fit reaches this number of iterations.
					ImageProcessor IP = image.getProcessor();
					// Initialize the driver and create a context for the first device.
					cuInit(0);
					CUdevice device = new CUdevice();
					cuDeviceGet(device, 0);
					CUcontext context = new CUcontext();
					cuCtxCreate(context, 0, device);
					
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
					for (int Ch = 1; Ch <= nChannels; Ch++)					// Loop over all channels.
					{						
						int nCenter =2*(( columns*rows/(gWindow[Ch-1]*gWindow[Ch-1])) / 2); // ~ 80 possible particles for a 64x64 frame. Lets the program scale with frame size.
						double gb = 1024*1024*1024;
						double maxMemoryGPU = 3*gb; // TODO: get size of gpu memory.
						int nMax = (int) (maxMemoryGPU/(columns*rows*Sizeof.INT + nCenter*Sizeof.INT)); 	// the localMaxima GPU calculations require: (x*y*frame*(Sizeof.INT ) + frame*nCenters*Sizeof.FLOAT)/gb memory. with known x and y dimensions, determine maximum size of frame for each batch.
						ArrayList<fitParameters> fitThese = new ArrayList<fitParameters>();
						int dataIdx = 0;
						int idx = 0;

						if (nMax > nFrames)
							nMax = nFrames;
						int[] data = new int[columns*rows*nMax];
						int loopStartFrame = 0;
						boolean processed = true;
						for (int Frame = 1; Frame <= nFrames; Frame ++)
						{					
							
							image.setPosition(
									Ch,			// channel.
									1,			// slice.
									Frame);		// frame.
							IP = image.getProcessor();
							if (processed)
							{
								loopStartFrame = Frame;
								processed = false;
							}
							// load as large chunks of data as possible on the gpu at each time.
							if (dataIdx < nMax)
							{
								for (int x = 0; x < columns; x++)
								{									
									for (int y = 0; y < rows; y++)
									{				
										data[idx] = IP.get(x, y);							
										idx++;
									} // y loop.
								} // x loop.
								dataIdx++; // count up number of added frames this round.		    			
							}
							if (dataIdx == nMax)
							{
								dataIdx = 0; // reset for next round.
								idx = 0; // reset for next round.		
								
								CUdeviceptr deviceData 	= CUDA.copyToDevice(data);
								CUdeviceptr deviceCenter = CUDA.allocateOnDevice((int)(nMax*nCenter));

								Pointer kernelParameters 		= Pointer.to(   
										Pointer.to(deviceData),
										Pointer.to(new int[]{data.length}),
										Pointer.to(new int[]{columns}),		 				       
										Pointer.to(new int[]{rows}),
										Pointer.to(new int[]{gWindow[Ch-1]}),
										Pointer.to(new int[]{MinLevel[Ch-1]}),
										Pointer.to(new double[]{sqDistance[Ch-1]}),
										Pointer.to(new int[]{minPosPixels[Ch-1]}),
										Pointer.to(new int[]{nCenter}),
										Pointer.to(deviceCenter),
										Pointer.to(new int[]{nMax*nCenter}));

								int blockSizeX 	= 1;
								int blockSizeY 	= 1;				   
								int gridSizeX 	= (int) Math.ceil((Math.sqrt(nMax)));
								int gridSizeY 	= gridSizeX;
								
								cuLaunchKernel(findMaximaFcn,
										gridSizeX,  gridSizeY, 1, 	// Grid dimension
										blockSizeX, blockSizeY, 1,  // Block dimension
										0, null,               		// Shared memory size and stream
										kernelParameters, null 		// Kernel- and extra parameters
										);
								cuCtxSynchronize();

								int hostCenter[] = new int[nMax*nCenter];

								// Pull data from device.
								cuMemcpyDtoH(Pointer.to(hostCenter), deviceCenter,
										nMax*nCenter * Sizeof.INT);

								// Free up memory allocation on device, housekeeping.
								cuMemFree(deviceData);   
								cuMemFree(deviceCenter);
								// add adding of fitThese.
								
								int centerIdx = 0;

								while (centerIdx < hostCenter.length)
								{
									if (hostCenter[centerIdx] >  0)
									{
										int[] coord = {hostCenter[centerIdx],hostCenter[centerIdx+1]};;
										fitParameters fitObj = new fitParameters();
										fitObj.Center =  coord;
										fitObj.frame  = loopStartFrame + centerIdx/ nCenter;
										fitObj.pixelsize = inputPixelSize[Ch-1];
										fitObj.windowWidth = gWindow[Ch-1];
										fitObj.totalGain = totalGain[Ch-1];
										fitObj.channel = Ch;																							
										fitThese.add(fitObj);									
									}
									centerIdx += 2;									
								}
								processed = true;
							}else if ( Frame == nFrames)
							{
								while (idx < data.length)
								{
									data[idx] = 0; // remove remaining entries.
									idx++;
								}
								CUdeviceptr deviceData 	= CUDA.copyToDevice(data);
								CUdeviceptr deviceCenter = CUDA.allocateOnDevice((int)(nMax*nCenter));

								Pointer kernelParameters 		= Pointer.to(   
										Pointer.to(deviceData),
										Pointer.to(new int[]{data.length}),
										Pointer.to(new int[]{columns}),		 				       
										Pointer.to(new int[]{rows}),
										Pointer.to(new int[]{gWindow[Ch-1]}),
										Pointer.to(new int[]{MinLevel[Ch-1]}),
										Pointer.to(new double[]{sqDistance[Ch-1]}),
										Pointer.to(new int[]{minPosPixels[Ch-1]}),
										Pointer.to(new int[]{nCenter}),
										Pointer.to(deviceCenter),
										Pointer.to(new int[]{nMax*nCenter}));

								int blockSizeX 	= 1;
								int blockSizeY 	= 1;				   
								int gridSizeX 	= (int) Math.ceil((Math.sqrt(nMax)));
								int gridSizeY 	= gridSizeX;
								cuLaunchKernel(findMaximaFcn,
										gridSizeX,  gridSizeY, 1, 	// Grid dimension
										blockSizeX, blockSizeY, 1,  // Block dimension
										0, null,               		// Shared memory size and stream
										kernelParameters, null 		// Kernel- and extra parameters
										);
								cuCtxSynchronize();

								int hostCenter[] = new int[nMax*nCenter];

								// Pull data from device.
								cuMemcpyDtoH(Pointer.to(hostCenter), deviceCenter,
										nMax*nCenter * Sizeof.INT);

								// Free up memory allocation on device, housekeeping.
								cuMemFree(deviceData);   
								cuMemFree(deviceCenter);
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
								}
							}
						} // frame loop.

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
						Pointer kernelParameters 		= Pointer.to(   
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

						int blockSizeX 	= 1;
						int blockSizeY 	= 1;				   
						int gridSizeX 	= (int) Math.ceil(Math.sqrt(N));
						int gridSizeY 	= gridSizeX;
						cuLaunchKernel(fittingFcn,
								gridSizeX,  gridSizeY, 1, 	// Grid dimension
								blockSizeX, blockSizeY, 1,  // Block dimension
								0, null,               		// Shared memory size and stream
								kernelParameters, null 		// Kernel- and extra parameters
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
					
					return cleanResults;
				} // end GPU computing.

		return Results;
	}

	/*
	 * Generate fitParameter objects by finding local maximas seperated by sqDistance of atleast MinLevel center pixel intensity. 
	 * Returns fitParameters for subsequent gaussian fitting.
	 */
	public static ArrayList<fitParameters> LocalizeEvents(ImageProcessor IP, int MinLevel, double sqDistance, int Window, int Frame, int Channel, int pixelSize, int minPosPixels, int totalGain){
		int[][] DataArray 		= IP.getIntArray();												// Array representing the frame.
		ArrayList<int[]> Center 	= LocalMaxima.FindMaxima(DataArray, Window, MinLevel, sqDistance,minPosPixels); 	// Get possibly relevant center coordinates.
		ArrayList<fitParameters> fitThese = new ArrayList<fitParameters>();
		for (int i = 0; i < Center.size(); i++){
			int[] dataFit = new int[Window*Window];							// Container for data to be fitted.
			int[] Coord = Center.get(i);									// X and Y coordinates for center pixels to be fitted.

			for (int j = 0; j < Window*Window; j++)
			{
				int x =  Coord[0] - Math.round((Window)/2) +  (j % Window);
				int y =  Coord[1] - Math.round((Window)/2) +  (j / Window);
				dataFit[j] = (int) IP.getf(x,y);
			}
			fitThese.add(new fitParameters(Coord, 
					dataFit,
					Channel,
					Frame,
					pixelSize,
					Window,
					totalGain));
		}
		return fitThese;																					// Results contain all particles located.
	} // end LocalizeEvents
}
