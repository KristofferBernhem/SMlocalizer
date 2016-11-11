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
import ij.plugin.filter.Analyzer;
import ij.process.ImageProcessor;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;

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


		if (selectedModel == 0) // parallel
		{
			ImageProcessor IP = image.getProcessor();
			ArrayList<fitParameters> fitThese 	= new ArrayList<fitParameters>(); 	// arraylist to hold all fitting parameters.
			for (int Ch = 1; Ch <= nChannels; Ch++)							// Loop over all channels.
			{
				for (int Frame = 1; Frame <= nFrames;Frame++)					// Loop over all frames.
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
			ij.measure.ResultsTable tab = Analyzer.getResultsTable();
			tab.reset();		
			tab.incrementCounter();
			tab.addValue("width", columns*inputPixelSize[0]);
			tab.addValue("height", rows*inputPixelSize[0]);
			tab.show("Results");
			return cleanResults; // end parallel computation by returning results.
		}else // end parallel. 
			if (selectedModel == 2) // GPU TODO: Add image loading to GPU bound code. 
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
				CUmodule modulePrepGauss = new CUmodule();
				cuModuleLoad(modulePrepGauss, "prepareGaussian.ptx");					
				// Obtain a handle to the kernel function
				CUfunction prepareGaussFcn = new CUfunction();
				cuModuleGetFunction(prepareGaussFcn, modulePrepGauss, "run");	// prepareGaussian.ptx (run function).
				
				
				
				for (int Ch = 1; Ch <= nChannels; Ch++)					// Loop over all channels.
				{						
					int nCenter =(( columns*rows/(gWindow[Ch-1]*gWindow[Ch-1])) / 2); // ~ 80 possible particles for a 64x64 frame. Lets the program scale with frame size.
					long gb = 1024*1024*1024;
					long maxMemoryGPU = 4*gb; // TODO: get size of gpu memory.
					int nMax = (int) (maxMemoryGPU/(columns*rows*Sizeof.INT + 4*nCenter*Sizeof.INT)); 	// the localMaxima GPU calculations require: (x*y*frame*(Sizeof.INT ) + frame*nCenters*Sizeof.FLOAT)/gb memory. with known x and y dimensions, determine maximum size of frame for each batch.
					//ArrayList<fitParameters> fitThese = new ArrayList<fitParameters>();
					int dataIdx = 0;
					int idx = 0;
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
					System.out.println(nMax);
					if (nMax > nFrames)
						nMax = nFrames;

					int[] data = new int[columns*rows*nMax];	
					boolean processed = true;
					int startFrame = 0;
					for (int Frame = 1; Frame <= nFrames; Frame ++)
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
						if(processed)
						{
							processed = false;
							startFrame = Frame;
						}
	

						// load as large chunks of data as possible on the gpu at each time.
						if (dataIdx < nMax)
						{
							for(int i = 0; i < columns*rows;i++) // loop over X then Y.
							{
								data[idx] = IP.get(i);
								idx++;
							}							
							dataIdx++; // count up number of added frames this round.		    			
						}
						if (dataIdx == nMax)
						{
							dataIdx = 0; // reset for next round.
							idx = 0; // reset for next round.		
							processed=true;
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
								P[i*7] = data[locatedCenter[i]];
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
				                    gaussVector[i * gWindow[Ch-1] * gWindow[Ch-1] + j] = data[k]; // add data.
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
							CUdeviceptr deviceP 	= 		CUDA.copyToDevice(P);
							CUdeviceptr deviceStepSize 	= 		CUDA.copyToDevice(stepSize);							
							
						
							/*
							 * This functions is being implemented, reducing computationtime by reducing device to host to device transfers. 
							 * 
							CUdeviceptr deviceLocatedCenter = CUDA.copyToDevice(locatedCenter);
							CUdeviceptr deviceGaussVector 	= CUDA.allocateOnDevice((int)(newN * gWindow[Ch-1] * gWindow[Ch-1])); // gWindow*gWindow entries per found center.
							CUdeviceptr deviceP 			= CUDA.allocateOnDevice((float)(newN * 7)); // 7 entries per found center.
							CUdeviceptr deviceStepSize 		= CUDA.allocateOnDevice((float)(newN * 7));  // 7 entries per found center.
							gridSizeX = (int) Math.ceil((Math.sqrt(newN)));
							gridSizeY 	= gridSizeX;
							
							
							Pointer kernelParametersPrepareGaussFit 		= Pointer.to(   
									Pointer.to(deviceData),
									Pointer.to(new int[]{data.length}),
									Pointer.to(deviceLocatedCenter),
									Pointer.to(new int[]{newN}),
									Pointer.to(new int[]{columns}),		 				       
									Pointer.to(new int[]{rows}),
									Pointer.to(new int[]{gWindow[Ch-1]}),
									Pointer.to(deviceGaussVector),
									Pointer.to(new int[]{newN * gWindow[Ch-1] * gWindow[Ch-1]}),
									Pointer.to(deviceP),																											
									Pointer.to(new float[]{newN*7}),
									Pointer.to(deviceStepSize),																											
									Pointer.to(new float[]{newN*7})
									);
							
							
							cuLaunchKernel(prepareGaussFcn,
									gridSizeX,  gridSizeY, 1, 	// Grid dimension
									blockSizeX, blockSizeY, 1,  // Block dimension
									0, null,               		// Shared memory size and stream
									kernelParametersPrepareGaussFit, null 		// Kernel- and extra parameters
									);
					//		cuCtxSynchronize();



							// clean device memory.
							cuMemFree(deviceData); // remove deviceData from device.
							cuMemFree(deviceLocatedCenter); // remove deviceLocatedCenter from device.
							*/
							// pull down data and relaunch.
						//	int hostOutput[] = new int[newN * gWindow[Ch-1] * gWindow[Ch-1]];

//							cuMemcpyDtoH(Pointer.to(hostOutput), deviceGaussVector,
//									newN*25 * Sizeof.INT);
							//for( int i = 0; i < hostOutput.length; i++)
								//System.out.println(hostOutput[i]);

							/******************************************************************************
							 * Gauss fitting.
							 ******************************************************************************/

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

							float hostOutput[] = new float[newN*7];

							// Pull data from device.
							cuMemcpyDtoH(Pointer.to(hostOutput), deviceP,
									newN*7 * Sizeof.FLOAT);
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
								Localized.r_square 		= hostOutput[n*7+6];
								Localized.x				= inputPixelSize[Ch-1]*(hostOutput[n*7+1] + (locatedCenter[n]%columns) - Math.round((gWindow[Ch-1])/2));
								Localized.y				= inputPixelSize[Ch-1]*(hostOutput[n*7+2] + ((locatedCenter[n]/columns)%rows) - Math.round((gWindow[Ch-1])/2));
								Localized.z				= inputPixelSize[Ch-1]*0;	// no 3D information.
								Localized.sigma_x		= inputPixelSize[Ch-1]*hostOutput[n*7+3];
								Localized.sigma_y		= inputPixelSize[Ch-1]*hostOutput[n*7+4];
								Localized.sigma_z		= inputPixelSize[Ch-1]*0; // no 3D information.
								Localized.photons		= (int) (hostOutput[n*7]/totalGain[Ch-1]);
								Localized.precision_x 	= Localized.sigma_x/Math.sqrt(Localized.photons);
								Localized.precision_y 	= Localized.sigma_y/Math.sqrt(Localized.photons);
								Localized.precision_z 	= Localized.sigma_z/Math.sqrt(Localized.photons);
								Results.add(Localized);
							}					
						}else if ( Frame == nFrames) // final part if chunks were loaded.
						{
							while (idx < data.length)
							{
								data[idx] = 0; // remove remaining entries.
								idx++;
							}
							processed = true;
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
							cuMemFree(deviceCenter);
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
								P[i*7] = data[locatedCenter[i]];
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
				                    gaussVector[i * gWindow[Ch-1] * gWindow[Ch-1] + j] = data[k]; // add data.
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
							CUdeviceptr deviceP 	= 		CUDA.copyToDevice(P);
							CUdeviceptr deviceStepSize 	= 		CUDA.copyToDevice(stepSize);							
							
						
							/*
							 * 
							 * This functions is being implemented, reducing computationtime by reducing device to host to device transfers.
							 * 
							 * 
							CUdeviceptr deviceLocatedCenter = CUDA.copyToDevice(locatedCenter);
							CUdeviceptr deviceGaussVector 	= CUDA.allocateOnDevice((int)(newN * gWindow[Ch-1] * gWindow[Ch-1])); // gWindow*gWindow entries per found center.
							CUdeviceptr deviceP 			= CUDA.allocateOnDevice((float)(newN * 7)); // 7 entries per found center.
							CUdeviceptr deviceStepSize 		= CUDA.allocateOnDevice((float)(newN * 7));  // 7 entries per found center.
							gridSizeX = (int) Math.ceil((Math.sqrt(newN)));
							gridSizeY 	= gridSizeX;
							
							
							Pointer kernelParametersPrepareGaussFit 		= Pointer.to(   
									Pointer.to(deviceData),
									Pointer.to(new int[]{data.length}),
									Pointer.to(deviceLocatedCenter),
									Pointer.to(new int[]{newN}),
									Pointer.to(new int[]{columns}),		 				       
									Pointer.to(new int[]{rows}),
									Pointer.to(new int[]{gWindow[Ch-1]}),
									Pointer.to(deviceGaussVector),
									Pointer.to(new int[]{newN * gWindow[Ch-1] * gWindow[Ch-1]}),
									Pointer.to(deviceP),																											
									Pointer.to(new float[]{newN*7}),
									Pointer.to(deviceStepSize),																											
									Pointer.to(new float[]{newN*7})
									);
							
							
							cuLaunchKernel(prepareGaussFcn,
									gridSizeX,  gridSizeY, 1, 	// Grid dimension
									blockSizeX, blockSizeY, 1,  // Block dimension
									0, null,               		// Shared memory size and stream
									kernelParametersPrepareGaussFit, null 		// Kernel- and extra parameters
									);
					//		cuCtxSynchronize();



							// clean device memory.
							cuMemFree(deviceData); // remove deviceData from device.
							cuMemFree(deviceLocatedCenter); // remove deviceLocatedCenter from device.
							*/
							// pull down data and relaunch.
						//	int hostOutput[] = new int[newN * gWindow[Ch-1] * gWindow[Ch-1]];

//							cuMemcpyDtoH(Pointer.to(hostOutput), deviceGaussVector,
//									newN*25 * Sizeof.INT);
							//for( int i = 0; i < hostOutput.length; i++)
								//System.out.println(hostOutput[i]);

							/******************************************************************************
							 * Gauss fitting.
							 ******************************************************************************/

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

							float hostOutput[] = new float[newN*7];

							// Pull data from device.
							cuMemcpyDtoH(Pointer.to(hostOutput), deviceP,
									newN*7 * Sizeof.FLOAT);
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
								Localized.r_square 		= hostOutput[n*7+6];
								Localized.x				= inputPixelSize[Ch-1]*(hostOutput[n*7+1] + (locatedCenter[n]%columns) - Math.round((gWindow[Ch-1])/2));
								Localized.y				= inputPixelSize[Ch-1]*(hostOutput[n*7+2] + ((locatedCenter[n]/columns)%rows) - Math.round((gWindow[Ch-1])/2));
								Localized.z				= inputPixelSize[Ch-1]*0;	// no 3D information.
								Localized.sigma_x		= inputPixelSize[Ch-1]*hostOutput[n*7+3];
								Localized.sigma_y		= inputPixelSize[Ch-1]*hostOutput[n*7+4];
								Localized.sigma_z		= inputPixelSize[Ch-1]*0; // no 3D information.
								Localized.photons		= (int) (hostOutput[n*7]/totalGain[Ch-1]);
								Localized.precision_x 	= Localized.sigma_x/Math.sqrt(Localized.photons);
								Localized.precision_y 	= Localized.sigma_y/Math.sqrt(Localized.photons);
								Localized.precision_z 	= Localized.sigma_z/Math.sqrt(Localized.photons);
								Results.add(Localized);
							}			
						}
					} // frame loop.
					cuMemFree(deviceBounds);
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
				ij.measure.ResultsTable tab = Analyzer.getResultsTable();
				tab.reset();		
				tab.incrementCounter();
				tab.addValue("width", columns*inputPixelSize[0]);
				tab.addValue("height", rows*inputPixelSize[0]);
				tab.show("Results");
				return cleanResults;
			} // end GPU computing.
		ij.measure.ResultsTable tab = Analyzer.getResultsTable();
		tab.reset();		
		tab.incrementCounter();
		tab.addValue("width", columns*inputPixelSize[0]);
		tab.addValue("height", rows*inputPixelSize[0]);
		tab.show("Results");
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
